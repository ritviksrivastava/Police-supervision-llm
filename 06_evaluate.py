"""Generate answers for the test set with a fine-tuned adapter (or the plain base model) and score them.

Examples
  python 06_evaluate.py --model Qwen/Qwen3.5-9B --adapter ../runs/qwen35-9b-sn/final --out ../runs/qwen35-9b-sn/eval
  python 06_evaluate.py --model Qwen/Qwen3.5-9B --out ../runs/qwen35-9b-base-eval        # baseline: no adapter
  python 06_evaluate.py --out ../runs/qwen35-9b-sn/eval --score-only                     # re-score existing predictions

Writes into --out:
  predictions.jsonl   the model's answer for each test case   (generation resumes from here if interrupted)
  scored.jsonl        reference, prediction and per-case scores
  metrics.json        averages over the test set
  report.html         reference and prediction side by side, worst cases first
"""
import argparse, collections, html, json, os, re, time

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--adapter", default="", help="folder saved by 05_train.py (…/final); omit to test the base model")
    p.add_argument("--data", default=os.path.join(HERE, "..", "dataset", "supervision"))
    p.add_argument("--split", default="test")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0, help="evaluate only the first N cases")
    p.add_argument("--max-len", type=int, default=16384, help="skip cases whose prompt is longer than this many tokens")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--stop-on-repeat", action="store_true",
                   help="end an answer once it has repeated three earlier lines (a loop), and drop repeated lines")
    p.add_argument("--load-4bit", action="store_true")
    p.add_argument("--score-only", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------- generation

_LINE_NUM = re.compile(r"^\s*\d{1,2}\s*[.)]\s*")


LOOP_REPEATS = 3   # this many repeated lines means the model is looping, not just restating one direction


def drop_repeated_lines(text, min_len=20):
    """Return (text without lines that repeat an earlier line, number of lines dropped). Numbering is ignored."""
    seen, kept, dropped = set(), [], 0
    for line in text.split("\n"):
        key = re.sub(r"\s+", " ", _LINE_NUM.sub("", line)).strip()
        if len(key) >= min_len:
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
        kept.append(line)
    return "\n".join(kept), dropped


def generate(args, cases, pred_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

    class StopOnRepeat(StoppingCriteria):
        """Marks a sequence finished once its newly generated text contains a repeated line."""

        def __init__(self, tok, prompt_len, every=16):
            self.tok, self.prompt_len, self.every, self.calls = tok, prompt_len, every, 0

        def __call__(self, input_ids, scores, **kwargs):
            done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            self.calls += 1
            if self.calls % self.every:
                return done
            for i, row in enumerate(input_ids):
                text = self.tok.decode(row[self.prompt_len:], skip_special_tokens=True)
                # only complete lines count: the last one may still be being written
                done[i] = drop_repeated_lines(text.rsplit("\n", 1)[0])[1] >= LOOP_REPEATS
            return done

    done = set()
    if os.path.exists(pred_path):
        with open(pred_path, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f}
    todo = [c for c in cases if c["id"] not in done]
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(args.adapter or args.model)
    tok.padding_side = "left"          # batched generation needs the prompts right-aligned
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"dtype": torch.bfloat16, "device_map": {"": 0}}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    except ValueError:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.model, **kwargs)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    prompts = {}
    for c in todo:   # same prompt construction as 05_train.py
        text = tok.apply_chat_template(c["messages"][:-1], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        if n <= args.max_len:
            prompts[c["id"]] = (text, n)
    print(f"generating {len(prompts)} cases ({len(todo) - len(prompts)} skipped as too long, {len(done)} already done)")
    todo = sorted((c for c in todo if c["id"] in prompts), key=lambda c: prompts[c["id"]][1])   # similar lengths together

    start, n_new = time.time(), 0
    with open(pred_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), args.batch):
            batch = todo[i:i + args.batch]
            enc = tok([prompts[c["id"]][0] for c in batch], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            stops = (StoppingCriteriaList([StopOnRepeat(tok, enc["input_ids"].shape[1])])
                     if args.stop_on_repeat else None)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     repetition_penalty=args.repetition_penalty, pad_token_id=tok.pad_token_id,
                                     stopping_criteria=stops)
            for c, seq in zip(batch, out):
                new = seq[enc["input_ids"].shape[1]:]
                n_tokens = int((new != tok.pad_token_id).sum())
                n_new += n_tokens
                text = tok.decode(new, skip_special_tokens=True)
                looped = False
                if args.stop_on_repeat:
                    if tok.eos_token_id not in new.tolist():   # stopped early or cut off: last line is unfinished
                        text = text.rsplit("\n", 1)[0]
                    text, dropped = drop_repeated_lines(text)
                    looped = dropped >= LOOP_REPEATS
                f.write(json.dumps({"id": c["id"], "source_file": c.get("source_file", c.get("report_file")),
                                    "prediction": text.strip(), "stopped_on_repeat": looped,
                                    "hit_token_limit": not looped and n_tokens >= args.max_new_tokens},
                                   ensure_ascii=False) + "\n")
            f.flush()
            elapsed = time.time() - start
            print(f"  {min(i + args.batch, len(todo))}/{len(todo)} | {n_new / elapsed:.0f} new tokens/s | "
                  f"about {elapsed / (i + len(batch)) * (len(todo) - i - len(batch)) / 60:.0f} min left")


# ---------------------------------------------------------------- scoring

_HEAD = re.compile(r"(निष्कर्ष|त्रुटि|निर्देश)\s*[:ः]\s*-?")
_ITEM = re.compile(r"^\s*\d{1,2}\s*[.)]\s*")


def split_answer(text):
    """Return ({'निष्कर्ष': str, 'त्रुटि': str}, [directions])."""
    parts, marks = {}, list(_HEAD.finditer(text))
    for m, nxt in zip(marks, marks[1:] + [None]):
        parts.setdefault(m.group(1), text[m.end():nxt.start() if nxt else len(text)].strip())
    directions = [_ITEM.sub("", line).strip() for line in parts.get("निर्देश", "").splitlines()]
    return parts, [d for d in directions if d]


def chrf(hyp, ref, max_n=6, beta=2.0):
    """Character n-gram F-score (0-100). Suits Hindi better than word overlap: tolerant of spelling/inflection variants."""
    hyp, ref = re.sub(r"\s+", " ", hyp.strip()), re.sub(r"\s+", " ", ref.strip())
    if not hyp or not ref:
        return 100.0 if hyp == ref else 0.0
    scores = []
    for n in range(1, max_n + 1):
        h = collections.Counter(hyp[i:i + n] for i in range(len(hyp) - n + 1))
        r = collections.Counter(ref[i:i + n] for i in range(len(ref) - n + 1))
        if not h or not r:
            continue
        match = sum((h & r).values())
        prec, rec = match / sum(h.values()), match / sum(r.values())
        scores.append((1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec) if prec + rec else 0.0)
    return 100 * sum(scores) / len(scores) if scores else 0.0


def law_sections(conclusion):
    """Section numbers cited after 'धारा' in the finding, e.g. {'126(2)', '115(2)', '3(5)'}."""
    found = set()
    for m in re.finditer(r"धारा\s*[-–:]*\s*([0-9A-Za-z()/\s.,\-]+)", conclusion):
        found.update(t for t in re.findall(r"\d+[A-Za-z]?(?:\(\w+\))*", m.group(1)) if t)
    return found


def verdict(conclusion):
    """What the note concludes about the case, from its standard phrasing."""
    head = conclusion[:1500]
    if re.search(r"असत्य|मिथ्या", head):
        return "false"
    if re.search(r"साक्ष्य\s+की\s+कमी|सूत्रहीन", head):
        return "true-but-unproved"
    if re.search(r"सत्य", head):
        return "true"
    return "unclear"


def f1(pred, ref):
    if not pred and not ref:
        return 1.0
    hit = len(pred & ref)
    return 2 * hit / (len(pred) + len(ref)) if hit else 0.0


_FACT = re.compile(r"[A-Z]{2}\s?\d{1,2}\s?[A-Z]{0,3}\s?-?\s?\d{3,4}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{5,}")


def ungrounded_facts(prediction, source):
    """Vehicle numbers, dates and long numbers in the prediction that occur nowhere in the note: likely inventions."""
    squash = lambda s: re.sub(r"[\s\-./]", "", s)
    src = squash(source)
    facts = {squash(m) for m in _FACT.findall(prediction)}
    return sorted(f for f in facts if f not in src)


def score_case(case, prediction):
    source, reference = case["messages"][1]["content"], case["messages"][2]["content"]
    ref_parts, ref_dirs = split_answer(reference)
    hyp_parts, hyp_dirs = split_answer(prediction)
    ref_con, hyp_con = ref_parts.get("निष्कर्ष", ""), hyp_parts.get("निष्कर्ष", "")
    # each reference direction against its closest predicted one (coverage), and the reverse (relevance)
    cover = [max((chrf(h, r) for h in hyp_dirs), default=0.0) for r in ref_dirs]
    relevant = [max((chrf(h, r) for r in ref_dirs), default=0.0) for h in hyp_dirs]
    facts = {_f for _f in (re.sub(r"[\s\-./]", "", m) for m in _FACT.findall(prediction))}
    bad = ungrounded_facts(prediction, source)
    return {
        "well_formed": all(k in hyp_parts for k in ("निष्कर्ष", "त्रुटि", "निर्देश")) and bool(hyp_dirs),
        "chrf_overall": round(chrf(prediction, reference), 1),
        "chrf_conclusion": round(chrf(hyp_con, ref_con), 1),
        "directions_coverage": round(sum(cover) / len(cover), 1) if cover else 0.0,
        "directions_relevance": round(sum(relevant) / len(relevant), 1) if relevant else 0.0,
        "directions_covered_share": round(sum(c >= 50 for c in cover) / len(cover), 3) if cover else 0.0,
        "n_directions_ref": len(ref_dirs), "n_directions_pred": len(hyp_dirs),
        "law_sections_f1": round(f1(law_sections(hyp_con[:1500]), law_sections(ref_con[:1500])), 3),
        "verdict_ref": verdict(ref_con), "verdict_pred": verdict(hyp_con),
        "ungrounded_fact_share": round(len(bad) / len(facts), 3) if facts else 0.0,
        "ungrounded_facts": bad[:20],
        "length_ratio": round(len(prediction) / max(1, len(reference)), 2),
    }


def summarise(rows):
    mean = lambda key: round(sum(r["scores"][key] for r in rows) / len(rows), 3)
    return {
        "cases": len(rows),
        "well_formed_share": mean("well_formed"),
        "chrf_overall": mean("chrf_overall"),
        "chrf_conclusion": mean("chrf_conclusion"),
        "directions_coverage": mean("directions_coverage"),
        "directions_relevance": mean("directions_relevance"),
        "directions_covered_share": mean("directions_covered_share"),
        "avg_directions_ref": mean("n_directions_ref"), "avg_directions_pred": mean("n_directions_pred"),
        "law_sections_f1": mean("law_sections_f1"),
        "verdict_agreement": round(sum(r["scores"]["verdict_ref"] == r["scores"]["verdict_pred"] for r in rows) / len(rows), 3),
        "ungrounded_fact_share": mean("ungrounded_fact_share"),
        "length_ratio": mean("length_ratio"),
        "hit_token_limit_share": round(sum(bool(r.get("hit_token_limit")) for r in rows) / len(rows), 3),
        "stopped_on_repeat_share": round(sum(bool(r.get("stopped_on_repeat")) for r in rows) / len(rows), 3),
    }


# ---------------------------------------------------------------- report

def write_report(path, rows, metrics, title):
    esc = lambda s: html.escape(s).replace("\n", "<br>")
    out = ["<!doctype html><meta charset='utf-8'><title>" + html.escape(title) + "</title><style>"
           "body{font-family:'Nirmala UI','Noto Sans Devanagari',sans-serif;margin:24px;line-height:1.55;color:#222}"
           "table{border-collapse:collapse;margin:8px 0 20px}td,th{border:1px solid #ccc;padding:4px 10px;text-align:left}"
           ".case{border-top:3px solid #888;margin-top:28px;padding-top:8px}.cols{display:flex;gap:16px}"
           ".cols>div{flex:1;background:#f7f7f7;padding:10px;border-radius:6px}.bad{color:#b00020}"
           "summary{cursor:pointer;color:#555}small{color:#555}</style>",
           f"<h1>{html.escape(title)}</h1><table>"]
    out += [f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in metrics.items()]
    out.append("</table><p><small>Cases are ordered worst first (by overall chrF).</small></p>")
    for r in sorted(rows, key=lambda r: r["scores"]["chrf_overall"]):
        s = r["scores"]
        flags = []
        if s["verdict_ref"] != s["verdict_pred"]:
            flags.append(f"finding differs: reference “{s['verdict_ref']}”, model “{s['verdict_pred']}”")
        if s["ungrounded_facts"]:
            flags.append("not found in the note: " + ", ".join(s["ungrounded_facts"]))
        if r.get("hit_token_limit"):
            flags.append("output cut off at the token limit")
        out.append(
            f"<div class='case'><b>{html.escape(str(r.get('source_file')))}</b><br><small>chrF {s['chrf_overall']} · "
            f"conclusion {s['chrf_conclusion']} · directions covered {s['directions_covered_share']:.0%} "
            f"({s['n_directions_pred']} written / {s['n_directions_ref']} in reference) · law sections F1 {s['law_sections_f1']}"
            f"</small>" + "".join(f"<div class='bad'>⚠ {html.escape(x)}</div>" for x in flags) +
            f"<details><summary>Case facts given to the model</summary><p>{esc(r['input'])}</p></details>"
            f"<div class='cols'><div><b>Reference (officer)</b><p>{esc(r['reference'])}</p></div>"
            f"<div><b>Model</b><p>{esc(r['prediction'])}</p></div></div></div>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    pred_path = os.path.join(args.out, "predictions.jsonl")
    with open(os.path.join(args.data, args.split + ".jsonl"), encoding="utf-8") as f:
        cases = [json.loads(line) for line in f]
    if args.limit:
        cases = cases[:args.limit]
    if not args.score_only:
        generate(args, cases, pred_path)

    by_id = {c["id"]: c for c in cases}
    rows = []
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            case = by_id.get(p["id"])
            if case:
                rows.append({**p, "input": case["messages"][1]["content"], "reference": case["messages"][2]["content"],
                             "scores": score_case(case, p["prediction"])})
    if not rows:
        raise SystemExit("no predictions to score")
    with open(os.path.join(args.out, "scored.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "input"}, ensure_ascii=False) + "\n")
    metrics = summarise(rows)
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    write_report(os.path.join(args.out, "report.html"), rows, metrics,
                 f"{args.adapter or args.model} — {args.split} set")
    print(json.dumps(metrics, ensure_ascii=False, indent=1))
    print("report:", os.path.join(args.out, "report.html"))


if __name__ == "__main__":
    main()
