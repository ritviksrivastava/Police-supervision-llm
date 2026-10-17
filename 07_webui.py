"""Web UI: an operator types the sections of a supervision note; the fine-tuned model drafts निष्कर्ष, त्रुटि and निर्देश.

Run
  python 07_webui.py add-user <name>                      # create or reset a login (asks for the password)
  python 07_webui.py serve --adapter ../runs/qwen35-4b-sn/final            # local adapter folder
  python 07_webui.py serve --adapter <hf-user>/<repo>                       # or the private Hugging Face repo
  python 07_webui.py serve --adapter ../runs/qwen35-4b-sn/merged --vllm-url http://127.0.0.1:8000   # many users
Options: --host 127.0.0.1 --port 7860 --max-new-tokens 4096 --root-path /supervision (behind a reverse proxy)

Nothing typed into the form is written to disk or logged; only timings are printed.
"""
import argparse, getpass, hashlib, hmac, importlib.util, json, os, re, secrets, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
USERS = os.path.join(HERE, "webui_users.json")
sys.path.insert(0, HERE)
from krutidev import kruti_to_unicode, looks_like_kruti  # noqa: E402

_spec = importlib.util.spec_from_file_location("evaluate", os.path.join(HERE, "06_evaluate.py"))
evaluate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate)          # reuse the loop guard and the answer splitter

SYSTEM = ("आप झारखण्ड पुलिस के पर्यवेक्षण पदाधिकारी हैं। नीचे एक कांड की प्राथमिकी, घटनास्थल, साक्षियों के बयान एवं "
          "अनुसंधान की प्रगति दी गई है। इसके आधार पर पर्यवेक्षण टिप्पणी का निष्कर्ष, त्रुटि तथा अनुसंधानकर्ता को निर्देश लिखें।")
MAX_INPUT_TOKENS = 16384     # the model never saw longer inputs in training

# (key, label, lines, placeholder) – order is the order they appear in a note
CASE_FIELDS = [
    ("ps", "थाना", 1, "जैसे: बरवाअड्डा"),
    ("case_no", "कांड संख्या / वर्ष", 1, "जैसे: 147/2025"),
    ("fir_date", "प्राथमिकी दिनांक", 1, "जैसे: 15.06.2025"),
    ("sections", "धारा", 1, "जैसे: 115(2)/126(2)/352/351(2)/3(5) बी0एन0एस0"),
    ("supervisor", "पर्यवेक्षणकर्ता का नाम एवं पद", 1, ""),
]
HEADER_FIELDS = [
    ("place_short", "घटनास्थल (पता)", 1, ""),
    ("distance", "थाना से दूरी एवं दिशा", 1, "जैसे: थाना से करीब 03 कि0मी0 दक्षिण"),
    ("occurrence", "घटना की तिथि एवं समय", 1, ""),
    ("fir_time", "प्राथमिकी की तिथि एवं समय", 1, ""),
    ("supervision_date", "पर्यवेक्षण की तिथि", 1, ""),
    ("io", "अनुसंधानकर्ता का पद एवं नाम", 1, ""),
]
SECTION_FIELDS = [
    ("fir", "प्राथमिकी", 6, "वादी/वादिनी, नामजद अभियुक्त एवं आरोप का विवरण"),
    ("place", "घटनास्थल", 4, "घटनास्थल का विवरण एवं चौहद्दी"),
    ("pros_ev", "अभियोजन साक्ष्य", 6, "वादी/वादिनी का बयान"),
    ("witnesses", "साक्षी", 8, "हर साक्षी का बयान अलग अनुच्छेद में लिखें (बीच में एक खाली पंक्ति छोड़ें)"),
    ("progress", "अनुसंधान में प्रगति", 5, ""),
    ("arrest", "गिरफ्तारी", 2, "खाली छोड़ने पर 'शून्य।' माना जाएगा"),
    ("surrender", "समर्पण", 2, ""),
    ("seizure", "तलाशी सह जप्ती", 2, "खाली छोड़ने पर 'शून्य।' माना जाएगा"),
    ("defence", "प्रतिरक्षा साक्ष्य", 2, "खाली छोड़ने पर 'शून्य।' माना जाएगा"),
]
ALL_FIELDS = CASE_FIELDS + HEADER_FIELDS + SECTION_FIELDS
NIL_IF_EMPTY = {"arrest", "seizure", "defence"}


# ---------------------------------------------------------------- logins

def _hash(password, salt):
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2 ** 14, r=8, p=1).hex()


def add_user(name):
    users = json.load(open(USERS, encoding="utf-8")) if os.path.exists(USERS) else {}
    pw = getpass.getpass(f"password for {name}: ")
    if len(pw) < 10 or pw != getpass.getpass("again: "):
        sys.exit("passwords differ or are shorter than 10 characters; nothing changed")
    salt = secrets.token_bytes(16)
    users[name] = {"salt": salt.hex(), "hash": _hash(pw, salt)}
    with open(USERS, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=1)
    os.chmod(USERS, 0o600)
    print(f"saved login '{name}' in {USERS}")


def check_login(name, password):
    try:
        users = json.load(open(USERS, encoding="utf-8"))
        u = users[name]
        return hmac.compare_digest(_hash(password, bytes.fromhex(u["salt"])), u["hash"])
    except (OSError, KeyError, ValueError):
        return False


# ---------------------------------------------------------------- prompt

# Letter patterns that are common in Kruti Dev text and rare in English (kk = ा-ा, ks = ो, Z = reph, …)
_KRUTI_MARKS = re.compile(r"kk|ks|kS|Z|\bgS|esa|\bdk\b|\bds\b|\bdh\b|\bfd\b|/k|\bvk|\biz|%&|¼|½|'k|~[A-Za-z]|[a-zA-Z;]A(?:\s|$)")


def is_kruti(text):
    """Detection tuned for short form fields as well as long paragraphs."""
    return looks_like_kruti(text) or len(_KRUTI_MARKS.findall(text)) >= max(2, len(text) // 40)


def to_unicode(text, force):
    """Convert text typed/pasted in the Kruti Dev font; leave Unicode and English alone."""
    text = (text or "").strip()
    if text and (force or is_kruti(text)) and not re.search(r"[ऀ-ॿ]", text):
        return kruti_to_unicode(text).strip()
    return text


def build_case_facts(v):
    """Lay the fields out the way supervision notes are written (the format the model was trained on)."""
    lines = ["पर्यवेक्षण टिप्पणी"]
    case = f"{v['ps']} थाना कांड सं0-{v['case_no']}" if v["ps"] else f"कांड सं0-{v['case_no']}"
    if v["fir_date"]:
        case += f", दिनांक-{v['fir_date']}"
    if v["sections"]:
        case += f", धारा-{v['sections']}"
    if v["supervisor"]:
        case += f" में {v['supervisor']} का पर्यवेक्षण टिप्पणी"
    lines.append(case + " :-")

    header = []
    if v["place_short"]:
        header.append(f"घटनास्थल :- {v['place_short']}")
    if v["distance"]:
        header.append(f"दूरी एवं दिशा :- {v['distance']}")
    dates = [f"({k}) {label} :- {v[key]}" for k, label, key in
             (("क", "घटना", "occurrence"), ("ख", "प्राथमिकी", "fir_time"), ("ग", "पर्यवेक्षण", "supervision_date"))
             if v[key]]
    if dates:
        header.append("तिथि एवं समय :- " + " ".join(dates))
    if v["io"]:
        header.append(f"अनुसंधानकर्ता का पद एवं नाम :- {v['io']}")
    if v["supervisor"]:
        header.append(f"पर्यवेक्षणकर्ता का नाम एवं पद :- {v['supervisor']}")
    lines += [f"{i:02d}. {h}" for i, h in enumerate(header, 1)]
    if v["supervision_date"] and v["io"]:
        lines.append(f"मैंने इस कांड का पर्यवेक्षण दिनांक-{v['supervision_date']} को अनुसंधानकर्ता {v['io']} के समक्ष किया।")

    n = 0
    for key, label, _, _ in SECTION_FIELDS:
        text = v[key] or ("शून्य।" if key in NIL_IF_EMPTY else "")
        if not text:
            continue
        if key == "witnesses":
            for para in [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]:
                lines.append(f"साक्षी :- {para}")
            continue
        n += 1
        lines.append(f"({n}) {label} :- {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------- model

class Drafter:
    def __init__(self, base, adapter, max_new_tokens, device="auto"):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.max_new_tokens = torch, max_new_tokens
        cuda = torch.cuda.is_available() and device != "cpu"
        self.tok = AutoTokenizer.from_pretrained(adapter)
        kwargs = {"dtype": torch.bfloat16, "device_map": {"": 0} if cuda else {"": "cpu"}}
        try:
            model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
        except ValueError:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(base, **kwargs)
        self.model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
        self.lock = threading.Lock()      # one draft at a time on the GPU

    def prompt(self, facts):
        text = self.tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": facts}],
                                            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return text, len(self.tok(text, add_special_tokens=False)["input_ids"])

    def stream(self, prompt_text):
        """Yield the draft as it is written. Stops early if the model starts repeating itself."""
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
        torch = self.torch

        class LoopGuard(StoppingCriteria):
            def __init__(self, n_prompt):
                self.n_prompt, self.calls = n_prompt, 0

            def __call__(self, input_ids, scores, **kwargs):
                self.calls += 1
                if self.calls % 16:
                    return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
                text = tok.decode(input_ids[0, self.n_prompt:], skip_special_tokens=True).rsplit("\n", 1)[0]
                looping = evaluate.drop_repeated_lines(text)[1] >= evaluate.LOOP_REPEATS
                return torch.full((input_ids.shape[0],), looping, dtype=torch.bool, device=input_ids.device)

        tok = self.tok
        enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        args = dict(**enc, max_new_tokens=self.max_new_tokens, do_sample=False, streamer=streamer,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([LoopGuard(enc["input_ids"].shape[1])]))
        failure = []

        def run():
            try:
                self.model.generate(**args)
            except Exception as e:           # without this the streamer would wait for text forever
                failure.append(e)
                streamer.end()

        with self.lock:
            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            text = ""
            for piece in streamer:
                text += piece
                yield text
            worker.join()
        if failure:
            raise RuntimeError(f"generation failed: {failure[0]!r}")


class RemoteDrafter:
    """Sends drafts to a vLLM (OpenAI-compatible) server, which batches many requests on the GPU at once."""

    concurrency = 10

    def __init__(self, url, model_dir, max_new_tokens):
        import requests
        from transformers import AutoTokenizer

        self.requests, self.url, self.max_new_tokens = requests, url.rstrip("/"), max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_dir)     # same chat template the model was trained with
        self.model = requests.get(f"{self.url}/v1/models", timeout=10).json()["data"][0]["id"]

    prompt = Drafter.prompt

    def stream(self, prompt_text):
        """Yield the draft as it is written; closing the connection when the model loops makes vLLM drop the request."""
        body = {"model": self.model, "prompt": prompt_text, "max_tokens": self.max_new_tokens, "temperature": 0,
                "stream": True, "skip_special_tokens": True}
        text = ""
        try:
            with self.requests.post(f"{self.url}/v1/completions", json=body, stream=True, timeout=(10, 120)) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    if line == "data: [DONE]":
                        break
                    text += json.loads(line[6:])["choices"][0].get("text", "")
                    yield text
                    complete = text.rsplit("\n", 1)[0]
                    if "\n" in text and evaluate.drop_repeated_lines(complete)[1] >= evaluate.LOOP_REPEATS:
                        break                 # leaving the with-block closes the stream and cancels the request
        except self.requests.RequestException as e:
            raise RuntimeError(f"generation failed: {e!r}") from e


# ---------------------------------------------------------------- UI

def make_app(drafter):
    import gradio as gr

    def draft(kruti, *values):
        v = {key: to_unicode(val, kruti) for (key, *_), val in zip(ALL_FIELDS, values)}
        converted = [v[key] for key, *_ in ALL_FIELDS]
        keep = [gr.update() for _ in ALL_FIELDS]
        if not v["fir"] or not v["case_no"]:
            yield keep + ["", "", "", "⚠ कम से कम कांड संख्या और प्राथमिकी भरें।"]
            return
        prompt_text, n = drafter.prompt(build_case_facts(v))
        if n > MAX_INPUT_TOKENS:
            yield converted + ["", "", "", f"⚠ विवरण बहुत लंबा है ({n:,} टोकन; सीमा {MAX_INPUT_TOKENS:,})। कृपया संक्षिप्त करें।"]
            return
        start, text = time.time(), ""
        try:
            for text in drafter.stream(prompt_text):
                yield converted + [text, "", "", f"मसौदा लिखा जा रहा है… {time.time() - start:.0f} सेकंड"]
        except RuntimeError as e:
            print(e, flush=True)
            yield converted + [text, "", "", "⚠ मसौदा तैयार करते समय त्रुटि हुई। कृपया पुनः प्रयास करें या व्यवस्थापक को सूचित करें।"]
            return
        clean, dropped = evaluate.drop_repeated_lines(text)
        if dropped >= evaluate.LOOP_REPEATS:          # stopped mid-line by the loop guard: drop the unfinished line
            clean, dropped = evaluate.drop_repeated_lines(text.rsplit("\n", 1)[0])
            dropped = max(dropped, evaluate.LOOP_REPEATS)
        parts, directions = evaluate.split_answer(clean)
        dirs = "\n".join(f"{i}. {d}" for i, d in enumerate(directions, 1))
        note = f"मसौदा तैयार ({time.time() - start:.0f} सेकंड)। कृपया जाँच कर आवश्यक संशोधन करें।"
        if dropped >= evaluate.LOOP_REPEATS:
            note += " ⚠ मॉडल निर्देश दोहराने लगा था; दोहराए गए निर्देश हटा दिए गए हैं – निर्देश अधूरे हो सकते हैं।"
        if not parts:
            yield converted + [clean, "", "", "⚠ मॉडल का उत्तर अपेक्षित प्रारूप में नहीं है; पूरा उत्तर निष्कर्ष बॉक्स में है।"]
            return
        print(f"draft: {n} input tokens, {time.time() - start:.0f}s, loop guard {'on' if dropped >= 3 else 'off'}",
              flush=True)
        yield converted + [parts.get("निष्कर्ष", ""), parts.get("त्रुटि", ""), dirs, note]

    with gr.Blocks(title="पर्यवेक्षण टिप्पणी – मसौदा सहायक") as app:
        gr.Markdown("## पर्यवेक्षण टिप्पणी – मसौदा सहायक\n"
                    "कांड के विवरण भरें और **मसौदा तैयार करें** दबाएँ। मॉडल *निष्कर्ष*, *त्रुटि* एवं *निर्देश* का "
                    "मसौदा लिखेगा। **यह केवल मसौदा है** – पर्यवेक्षण पदाधिकारी द्वारा जाँच एवं संशोधन आवश्यक है।")
        boxes = {}
        with gr.Row():
            with gr.Column(scale=1):
                kruti = gr.Checkbox(label="Kruti Dev फ़ॉन्ट में टाइप/पेस्ट किया गया पाठ (यूनिकोड में बदलें)", value=False)
                with gr.Accordion("कांड का विवरण", open=True):
                    for key, label, lines, ph in CASE_FIELDS:
                        boxes[key] = gr.Textbox(label=label, lines=lines, placeholder=ph)
                with gr.Accordion("अन्य विवरण (वैकल्पिक)", open=False):
                    for key, label, lines, ph in HEADER_FIELDS:
                        boxes[key] = gr.Textbox(label=label, lines=lines, placeholder=ph)
                for key, label, lines, ph in SECTION_FIELDS:
                    boxes[key] = gr.Textbox(label=label, lines=lines, placeholder=ph)
                go = gr.Button("मसौदा तैयार करें", variant="primary")
            with gr.Column(scale=1):
                status = gr.Markdown("")
                conclusion = gr.Textbox(label="निष्कर्ष", lines=14, interactive=True)
                lapses = gr.Textbox(label="त्रुटि", lines=3, interactive=True)
                directions = gr.Textbox(label="निर्देश", lines=14, interactive=True)
        inputs = [kruti] + [boxes[key] for key, *_ in ALL_FIELDS]
        outputs = [boxes[key] for key, *_ in ALL_FIELDS] + [conclusion, lapses, directions, status]
        go.click(draft, inputs=inputs, outputs=outputs)
    return app


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add-user")
    a.add_argument("name")
    s = sub.add_parser("serve")
    s.add_argument("--base", default="Qwen/Qwen3.5-4B")
    s.add_argument("--adapter", required=True, help="adapter folder or private Hugging Face repo id")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=7860)
    s.add_argument("--root-path", default="")
    s.add_argument("--max-new-tokens", type=int, default=4096)
    s.add_argument("--no-login", action="store_true", help="only for testing on this machine")
    s.add_argument("--device", default="auto", choices=["auto", "cpu"], help="cpu: test without using the GPU")
    s.add_argument("--vllm-url", default="",
                   help="e.g. http://127.0.0.1:8000 - generate on a vLLM server (many users at once); "
                        "--adapter must then be the merged model folder, used for its tokenizer")
    args = p.parse_args()

    if args.cmd == "add-user":
        add_user(args.name)
        return
    if not args.no_login and not os.path.exists(USERS):
        sys.exit(f"no logins yet - create one with:  python {os.path.basename(__file__)} add-user <name>")
    if args.no_login and args.host not in ("127.0.0.1", "localhost"):
        sys.exit("--no-login is only allowed with --host 127.0.0.1")
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    t = time.time()
    if args.vllm_url:
        drafter = RemoteDrafter(args.vllm_url, args.adapter, args.max_new_tokens)
        limit = RemoteDrafter.concurrency
        print(f"using vLLM at {args.vllm_url} (model {drafter.model}); up to {limit} drafts at once", flush=True)
    else:
        drafter = Drafter(args.base, args.adapter, args.max_new_tokens, args.device)
        limit = 1
        print(f"model loaded in {time.time() - t:.0f}s", flush=True)
    app = make_app(drafter)
    app.queue(default_concurrency_limit=limit, max_size=30)
    app.launch(server_name=args.host, server_port=args.port, root_path=args.root_path or None,
               auth=None if args.no_login else check_login, auth_message="पर्यवेक्षण टिप्पणी – मसौदा सहायक",
               share=False, show_error=False)


if __name__ == "__main__":
    main()
