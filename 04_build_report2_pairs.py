"""Pair each Report-2 (प्रतिवेदन-02) with the supervision note it was issued on, and write chat-format JSONL.

Input  (user):      the full supervision note
Output (assistant): the Report-2, from its case line onwards (memo-number header lines are dropped)

A Report-2 and a note are matched when they carry the same case number and year AND the texts corroborate each
other (police station, FIR date, and how much of the report's vocabulary appears in the note).
"""
import collections, csv, difflib, hashlib, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "dataset", "sn_to_report2")
NOTE_TYPES = ("supervision_note", "supervision_cum_final_report")
SYSTEM = ("आप पुलिस अधीक्षक कार्यालय, धनबाद के अपराध शाखा में कार्यरत हैं। नीचे एक कांड की पर्यवेक्षण टिप्पणी दी गई है। "
          "इसके आधार पर उस कांड का प्रतिवेदन-02 तैयार करें।")
MIN_OVERLAP = 0.55          # share of the report's distinct words that must occur in the note
MIN_OVERLAP_CERTAIN = 0.45
MIN_OVERLAP_UNCONFIRMED = 0.75   # needed when neither police station nor FIR date can confirm the match

_WORD = re.compile(r"[ऀ-ॿ]{4,}|[A-Za-z0-9]{4,}")
_REPORT_NO = re.compile(r"प्रतिवेदन\s*[-–:]*\s*0?([0-9०-९]{1,2})")
_HEADER_LINE = re.compile(r"ज्ञापांक|वि[0०.]\s*प्र[0०.]\s*सं?[0०.]")
_FILE_CASE = re.compile(r"(\d{1,4})\s*[-@#/_., ]+\s*(?:20)?(\d{2})(?!\d)")


def text_of(rec):
    return "\n".join(t for _, t, _ in rec["paragraphs"])


def words(text):
    return set(_WORD.findall(text))


def skeleton(ps):
    return re.sub(r"[^क-ह]", "", ps or "")


def case_id(rec):
    """(case_no, 2-digit year) from the case line, else from the filename."""
    c = rec["case"]
    if c["case_no"] and c["case_year"]:
        return c["case_no"], c["case_year"] % 100
    name = re.sub(r"\.docx?$", "", rec["source_file"].split("__")[-1].split("/")[-1], flags=re.I)
    m = _FILE_CASE.search(name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def load(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def report_no(rec):
    m = _REPORT_NO.search(" ".join(t for _, t, _ in rec["paragraphs"][:6]))
    return int(m.group(1).translate(str.maketrans("०१२३४५६७८९", "0123456789"))) if m else None


def report_body(rec):
    """Report text from the case line onwards; the memo/reference header above it is not derivable from the note."""
    paras = [t for _, t, _ in rec["paragraphs"]]
    start = 0
    for i, p in enumerate(paras[:6]):
        if _HEADER_LINE.search(p) or re.fullmatch(r"\s*प्रतिवेदन\s*[-–:]*\s*0?[2२]\s*", p):
            start = i + 1
    return "प्रतिवेदन-02\n" + "\n".join(paras[start:])


def main():
    os.makedirs(OUT, exist_ok=True)
    notes, reports = {}, {}
    for name in ("supervision", "report2"):
        for rec in load(os.path.join(ROOT, "dataset", name, "records.jsonl")):
            if rec["devanagari_ratio"] < 0.5:
                continue
            if rec["doc_type"] in NOTE_TYPES:
                notes.setdefault(rec["doc_id"], rec)
            elif rec["doc_type"] == "report_2" and report_no(rec) == 2:
                reports.setdefault(rec["doc_id"], rec)

    by_case = collections.defaultdict(list)
    for n in notes.values():
        cid = case_id(n)
        if cid:
            n["_words"] = words(text_of(n))
            by_case[cid].append(n)

    stats = collections.Counter(notes=len(notes), reports=len(reports))
    best = {}   # one pair per (note case key): the report with the highest overlap
    audit = []
    for r in reports.values():
        cid = case_id(r)
        if not cid:
            stats["report: case number not found"] += 1
            continue
        cands = by_case.get(cid)
        if not cands:
            stats["report: no note with that case number"] += 1
            continue
        rw = words(report_body(r))
        if len(rw) < 40:
            stats["report: too short"] += 1
            continue
        n = max(cands, key=lambda c: len(rw & c["_words"]))
        overlap = len(rw & n["_words"]) / len(rw)
        ps_sim = difflib.SequenceMatcher(None, skeleton(r["case"]["police_station"]),
                                         skeleton(n["case"]["police_station"])).ratio()
        same_date = bool(r["case"]["case_date"]) and r["case"]["case_date"] == n["case"]["case_date"]
        confirmed = ps_sim >= 0.7 or same_date
        certain = ps_sim >= 0.7 and same_date    # same station, number, year and FIR date: wording may still differ
        ok = overlap >= (MIN_OVERLAP_CERTAIN if certain else MIN_OVERLAP if confirmed else MIN_OVERLAP_UNCONFIRMED)
        audit.append([cid[0], cid[1], r["case"]["police_station"], n["case"]["police_station"], round(ps_sim, 2),
                      same_date, round(overlap, 2), "matched" if ok else "rejected", r["source_file"], n["source_file"]])
        if not ok:
            stats["report: candidate note rejected (different case / low overlap)"] += 1
            continue
        c = n["case"]
        key = f"{skeleton(c['police_station'])}|{c['case_no']}|{c['case_year']}" if c["case_no"] else "doc|" + n["doc_id"]
        if key not in best or overlap > best[key][0]:
            if key in best:
                stats["duplicate report for the same case"] += 1
            best[key] = (overlap, r, n)
        else:
            stats["duplicate report for the same case"] += 1

    splits = collections.defaultdict(list)
    for key, (overlap, r, n) in best.items():
        inp, out = text_of(n), report_body(r)
        ex = {"id": r["doc_id"], "case_key": key, "case": n["case"], "match_overlap": round(overlap, 3),
              "note_file": n["source_file"], "report_file": r["source_file"], "note_id": n["doc_id"],
              "n_input_chars": len(inp), "n_output_chars": len(out),
              "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": inp},
                           {"role": "assistant", "content": out}]}
        bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % 20   # same split rule as the note dataset
        splits["val" if bucket == 0 else "test" if bucket == 1 else "train"].append(ex)
    for name, rows in splits.items():
        with open(os.path.join(OUT, name + ".jsonl"), "w", encoding="utf-8") as f:
            for ex in sorted(rows, key=lambda e: e["id"]):
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "match_audit.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_no", "year", "report_ps", "note_ps", "ps_similarity", "same_fir_date", "overlap", "result",
                    "report_file", "note_file"])
        w.writerows(sorted(audit, key=lambda a: a[6]))

    rows = [e for v in splits.values() for e in v]
    pct = lambda vals, p: sorted(vals)[min(len(vals) - 1, int(p * len(vals)))] if vals else 0
    summary = {
        "pairs": len(rows), "splits": {k: len(v) for k, v in splits.items()}, **stats,
        "input_chars": {k: pct([e["n_input_chars"] for e in rows], q) for k, q in (("p50", .5), ("p90", .9), ("p99", .99))},
        "output_chars": {k: pct([e["n_output_chars"] for e in rows], q) for k, q in (("p50", .5), ("p90", .9), ("p99", .99))},
    }
    json.dump(summary, open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
