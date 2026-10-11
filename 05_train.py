"""LoRA fine-tune a small chat model on the supervision-note dataset (loss on the assistant answer only).

Examples
  python 05_train.py --model Qwen/Qwen3.5-9B        --out ../runs/qwen35-9b-sn
  python 05_train.py --model google/gemma-4-E4B-it  --out ../runs/gemma4-e4b-sn      (gated: run `hf auth login` first)
  python 05_train.py --model Qwen/Qwen3.5-9B --dry-run          # tokenise only: length stats + what the loss covers
  python 05_train.py --model Qwen/Qwen3.5-9B --limit 64 --epochs 1 --out ../runs/smoke   # quick end-to-end check

Needs: torch, transformers (recent enough for the chosen model), peft, accelerate; bitsandbytes only for --load-4bit.
The same script trains the note -> Report-2 task: add  --data ../dataset/sn_to_report2
"""
import argparse, json, math, os, random, time

from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
SENTINEL = "␞ANSWER␞"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--data", default=os.path.join(HERE, "..", "dataset", "supervision"))
    p.add_argument("--out", default=os.path.join(HERE, "..", "runs", "sn-lora"))
    p.add_argument("--max-len", type=int, default=16384, help="examples longer than this many tokens are skipped")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--load-4bit", action="store_true", help="QLoRA: 4-bit base weights (for GPUs with little memory)")
    p.add_argument("--limit", type=int, default=0, help="use only the first N examples of each split (smoke test)")
    p.add_argument("--note-kind", default="", help="keep only this note_kind, e.g. supervision_note")
    p.add_argument("--save-steps", type=int, default=100, help="checkpoint every N optimiser steps (resumable)")
    p.add_argument("--max-mem-gb", type=float, default=0,
                   help="fail with an out-of-memory error above this much GPU memory instead of starving the machine")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def mem_available_gb():
    try:
        with open("/proc/meminfo") as f:
            return next(int(line.split()[1]) for line in f if line.startswith("MemAvailable")) / 2 ** 20
    except (OSError, StopIteration):
        return float("nan")


def load_split(path, limit, note_kind):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            if note_kind and ex.get("note_kind") != note_kind:
                continue
            rows.append(ex["messages"])
            if limit and len(rows) >= limit:
                break
    return rows


class Encoder:
    """Turns a chat into input_ids + labels, with labels masked (-100) everywhere except the assistant answer."""

    def __init__(self, tokenizer, max_len):
        self.tok, self.max_len = tokenizer, max_len
        # Whatever the template writes after an assistant message (e.g. <|im_end|> / <end_of_turn>) is the stop marker.
        probe = self._render([{"role": "user", "content": "x"}, {"role": "assistant", "content": SENTINEL}], False)
        self.answer_suffix = probe.split(SENTINEL, 1)[1].rstrip("\n")

    def _render(self, messages, add_generation_prompt):
        # enable_thinking is read by Qwen templates and ignored by the others
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                            enable_thinking=False)

    def __call__(self, messages):
        prompt = self._render(messages[:-1], True)
        answer = messages[-1]["content"] + self.answer_suffix
        prompt_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = self.tok(answer, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) + len(answer_ids) > self.max_len:
            return None
        return {"input_ids": prompt_ids + answer_ids, "labels": [-100] * len(prompt_ids) + answer_ids}


def encode_all(rows, encoder, name):
    out = [e for e in map(encoder, rows) if e is not None]
    lens = sorted(len(e["input_ids"]) for e in out)
    if lens:
        print(f"{name}: kept {len(out)}/{len(rows)} examples (<= {encoder.max_len} tokens) | "
              f"{sum(lens) / 1e6:.1f}M tokens | median {lens[len(lens) // 2]}, longest {lens[-1]}")
    return out


def main():
    args = parse_args()
    random.seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    encoder = Encoder(tok, args.max_len)

    train = encode_all(load_split(os.path.join(args.data, "train.jsonl"), args.limit, args.note_kind), encoder, "train")
    val = encode_all(load_split(os.path.join(args.data, "val.jsonl"), args.limit, args.note_kind), encoder, "val")
    train_tokens = sum(len(e["input_ids"]) for e in train)

    if args.dry_run:
        ex = train[0]
        n_prompt = ex["labels"].count(-100)
        print("answer stop marker:", repr(encoder.answer_suffix))
        print("--- end of prompt (not trained on) ---\n" + tok.decode(ex["input_ids"][max(0, n_prompt - 60):n_prompt]))
        print("--- start of answer (trained on) ---\n" + tok.decode(ex["input_ids"][n_prompt:n_prompt + 80]))
        print("--- end of answer ---\n" + tok.decode(ex["input_ids"][-25:]))
        return

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments

    if args.max_mem_gb:
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, args.max_mem_gb * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory))
    load_kwargs = {"dtype": torch.bfloat16, "device_map": {"": 0}}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    except ValueError:   # some multimodal checkpoints are only registered under the image-text class
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kwargs)

    if args.load_4bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    def collate(batch):
        width = max(len(e["input_ids"]) for e in batch)
        pad = lambda seq, value: seq + [value] * (width - len(seq))
        return {
            "input_ids": torch.tensor([pad(e["input_ids"], tok.pad_token_id) for e in batch]),
            "attention_mask": torch.tensor([pad([1] * len(e["input_ids"]), 0) for e in batch]),
            "labels": torch.tensor([pad(e["labels"], -100) for e in batch]),
        }

    class Throughput(TrainerCallback):
        """Prints measured tokens/second and the projected total time, so the estimate is replaced by a real number."""

        def on_train_begin(self, a, state, control, **kw):
            # after a resume global_step starts at the checkpoint's step, so measure from there
            self.start, self.start_step = time.time(), state.global_step

        def on_log(self, a, state, control, **kw):
            steps = state.global_step - self.start_step
            if steps <= 0 or not state.max_steps:
                return
            elapsed = time.time() - self.start
            done_tokens = train_tokens * args.epochs * steps / state.max_steps
            remaining = elapsed * (state.max_steps - state.global_step) / steps
            print(f"  ~{done_tokens / elapsed:,.0f} tokens/s | this session {elapsed / 3600:.2f} h | "
                  f"about {remaining / 3600:.1f} h remaining | peak GPU memory "
                  f"{torch.cuda.max_memory_allocated() / 2 ** 30:.0f} GB (reserved {torch.cuda.memory_reserved() / 2 ** 30:.0f} GB)"
                  f" | system memory available {mem_available_gb():.0f} GB")

    # warmup as a step count: newer transformers releases dropped the warmup_ratio argument
    total_steps = math.ceil(len(train) / (args.batch * args.grad_accum)) * args.epochs
    trainer = Trainer(
        model=model, train_dataset=train, eval_dataset=val, data_collator=collate, callbacks=[Throughput()],
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs, seed=args.seed,
            per_device_train_batch_size=args.batch, per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=max(1, int(0.03 * total_steps)),
            weight_decay=0.0, max_grad_norm=1.0,
            bf16=True, logging_steps=10, eval_strategy="epoch", save_strategy="steps", save_steps=args.save_steps,
            save_total_limit=2,
            report_to="none", dataloader_num_workers=0, remove_unused_columns=False))
    result = trainer.train(resume_from_checkpoint=bool(args.out and os.path.isdir(args.out) and any(
        d.startswith("checkpoint-") for d in os.listdir(args.out))))
    runtime = result.metrics.get("train_runtime", 0)
    if runtime:
        print(f"MEASURED: {train_tokens * args.epochs / runtime:,.0f} training tokens/s "
              f"({train_tokens * args.epochs / 1e6:.2f}M tokens in {runtime / 60:.1f} min, evaluation included)")
    final = os.path.join(args.out, "final")
    model.save_pretrained(final)
    tok.save_pretrained(final)
    print("adapter saved to", final)


if __name__ == "__main__":
    main()
