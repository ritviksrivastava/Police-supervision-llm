"""records.jsonl -> de-duplicated chat-format train/val/test JSONL for the supervision-note task.

Input  (user):      every paragraph of the note before the conclusion (header, FIR, scene, statements, progress, ...)
Output (assistant): निष्कर्ष + त्रुटि + निर्देश
"""
import collections, hashlib, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "dataset", "supervision")
OUTPUT_LABELS = ("conclusion", "lapses", "directions")
SYSTEM = ("आप झारखण्ड पुलिस के पर्यवेक्षण पदाधिकारी हैं। नीचे एक कांड की प्राथमिकी, घटनास्थल, साक्षियों के बयान एवं "
          "अनुसंधान की प्रगति दी गई है। इसके आधार पर पर्यवेक्षण टिप्पणी का निष्कर्ष, त्रुटि तथा अनुसंधानकर्ता को निर्देश लिखें।")
_ITEM_NUM = re.compile(r"^\s*(?:[(\[]?\s*(?:[0-9०-९]{1,2}|[क-हक्षत्रज्ञ]{1,2}|[ivxIVX]{1,4}|[A-Za-z])\s*\.?\s*[)\]][.\-–]?|[0-9०-९]{1,2}\s*[.\-–]|[•*\-–])\s*")
_LEADIN = re.compile(r"निर्दे[शष].{0,60}(?::|ः|कि|हैं?|[-–])\s*-?\s*$")   # "अनुसंधानकर्ता को निर्देश दिया जाता है कि :-"
NOTE_TYPES = ("supervision_note", "supervision_cum_final_report")
MIN_INPUT, MIN_CONCLUSION, MIN_DEVANAGARI = 1000, 200, 0.5


def build(rec):
    """Return (example, None) or (None, reason_dropped)."""
    if rec["doc_type"] not in NOTE_TYPES:
        return None, "not a supervision note (" + rec["doc_type"] + ")"
    if rec["devanagari_ratio"] < MIN_DEVANAGARI:
        return None, "text not Hindi after conversion"
    sec = rec["sections"]
    conclusion = "\n".join(sec.get("conclusion", []))
    directions = [_ITEM_NUM.sub("", d).strip() for d in sec.get("directions", [])]
    directions = [d for d in directions if d and not (len(d) < 220 and _LEADIN.search(d))]
    if len(conclusion) < MIN_CONCLUSION:
        return None, "no / too-short conclusion"
    if not directions:
        return None, "no directions"
    inp = "\n".join(text for label, text, _ in rec["paragraphs"] if label not in OUTPUT_LABELS + ("footer",))
    if len(inp) < MIN_INPUT:
        return None, "input too short"
    lapses = " ".join(sec.get("lapses", [])) or "शून्य।"
    out = ("निष्कर्ष :- " + conclusion + "\n\nत्रुटि :- " + lapses + "\n\nनिर्देश :-\n"
           + "\n".join(f"{i}. {d}" for i, d in enumerate(directions, 1)))
    c = rec["case"]
    case_key = (f"{re.sub(r'[^क-ह]', '', c['police_station'] or '')}|{c['case_no']}|{c['case_year']}"
                if c["case_no"] else "doc|" + rec["doc_id"])
    return {
        "id": rec["doc_id"], "note_kind": rec["doc_type"], "case_key": case_key, "source_file": rec["source_file"], "case": c,
        "n_input_chars": len(inp), "n_output_chars": len(out), "n_directions": len(directions),
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": inp},
                     {"role": "assistant", "content": out}],
    }, None


def fingerprint(ex):
    """Near-duplicate key: copies of a note share the opening of the conclusion even if the file was re-saved."""
    out = ex["messages"][2]["content"]
    return hashlib.sha1(re.sub(r"[^क-ह0-9]", "", out)[:500].encode("utf-8")).hexdigest()


def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))] if values else 0


def main():
    dropped, examples = collections.Counter(), {}
    n_records = 0
    with open(os.path.join(OUT, "records.jsonl"), encoding="utf-8") as f:
        for line in f:
            n_records += 1
            ex, reason = build(json.loads(line))
            if ex is None:
                dropped[reason] += 1
                continue
            fp = fingerprint(ex)
            if fp in examples:
                dropped["duplicate"] += 1
                if ex["n_input_chars"] <= examples[fp]["n_input_chars"]:
                    continue
            examples[fp] = ex

    splits = collections.defaultdict(list)
    for ex in examples.values():   # split by case so one case never straddles train and eval
        bucket = int(hashlib.sha1(ex["case_key"].encode("utf-8")).hexdigest(), 16) % 20
        splits["val" if bucket == 0 else "test" if bucket == 1 else "train"].append(ex)
    for name, rows in splits.items():
        with open(os.path.join(OUT, name + ".jsonl"), "w", encoding="utf-8") as f:
            for ex in sorted(rows, key=lambda e: e["id"]):
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    rows = list(examples.values())
    stats = {
        "records_read": n_records, "examples": len(rows),
        "splits": {k: len(v) for k, v in splits.items()},
        "dropped": dict(dropped.most_common()),
        "by_note_kind": dict(collections.Counter(e["note_kind"] for e in rows)),
        "distinct_cases": len({e["case_key"] for e in rows}),
        "input_chars": {p: pct([e["n_input_chars"] for e in rows], q) for p, q in (("p50", .5), ("p90", .9), ("p99", .99))},
        "output_chars": {p: pct([e["n_output_chars"] for e in rows], q) for p, q in (("p50", .5), ("p90", .9), ("p99", .99))},
        "directions_per_note": {p: pct([e["n_directions"] for e in rows], q) for p, q in (("p50", .5), ("p90", .9))},
    }
    json.dump(stats, open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
