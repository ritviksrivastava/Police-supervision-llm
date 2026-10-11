"""Read every supervision .docx, convert legacy-font text to Unicode, split into sections, write records.jsonl.

Usage: python 02_extract.py                    (reads ../data/word + ../data/word_converted, writes ../dataset/supervision/records.jsonl)
       python 02_extract.py report-2 report2   (same for ../report-2, writes ../dataset/report2/records.jsonl)
"""
import hashlib, json, os, re, sys, zipfile
from concurrent.futures import ProcessPoolExecutor
from lxml import etree

from krutidev import is_legacy_font, kruti_to_unicode, looks_like_kruti

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# ---------------------------------------------------------------- docx -> Unicode paragraphs

def _fonts_of(rpr):
    f = rpr.find(W + "rFonts") if rpr is not None else None
    if f is None:
        return None
    return f.get(W + "ascii") or f.get(W + "hAnsi") or f.get(W + "cs")


def _load_styles(z):
    """Return (font per styleId resolved through basedOn, default paragraph style id, document default font)."""
    try:
        root = etree.fromstring(z.read("word/styles.xml"))
    except KeyError:
        return {}, None, None
    raw, default_style = {}, None
    for s in root.iter(W + "style"):
        sid = s.get(W + "styleId")
        based = s.find(W + "basedOn")
        raw[sid] = (_fonts_of(s.find(W + "rPr")), based.get(W + "val") if based is not None else None)
        if s.get(W + "type") == "paragraph" and s.get(W + "default") == "1":
            default_style = sid
    dd = root.find(f"{W}docDefaults/{W}rPrDefault/{W}rPr")
    doc_default = _fonts_of(dd)

    def resolve(sid, depth=0):
        if sid not in raw or depth > 10:
            return None
        font, based = raw[sid]
        return font or resolve(based, depth + 1)

    return {sid: resolve(sid) for sid in raw}, default_style, doc_default


def _paragraph_text(p, styles, default_style, doc_default):
    ppr = p.find(W + "pPr")
    pstyle = ppr.find(W + "pStyle") if ppr is not None else None
    para_font = styles.get(pstyle.get(W + "val") if pstyle is not None else default_style) or doc_default

    groups = []  # consecutive runs with the same legacy flag are converted together (glyphs span runs)
    for r in p.iter(W + "r"):
        rpr = r.find(W + "rPr")
        font = _fonts_of(rpr)
        if font is None and rpr is not None and rpr.find(W + "rStyle") is not None:
            font = styles.get(rpr.find(W + "rStyle").get(W + "val"))
        legacy = is_legacy_font(font or para_font)
        text = "".join(
            (c.text or "") if c.tag == W + "t" else " "
            for c in r if c.tag in (W + "t", W + "tab", W + "br", W + "cr"))
        if not text:
            continue
        if groups and groups[-1][0] == legacy:
            groups[-1][1] += text
        else:
            groups.append([legacy, text])

    out = []
    for legacy, text in groups:
        if re.search(r"[ऀ-ॿ]", text):      # already Unicode, whatever the font says
            out.append(text)
        elif legacy or looks_like_kruti(text):
            out.append(kruti_to_unicode(text))
        else:
            out.append(text)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def read_docx(path):
    """Return paragraphs in reading order; each table row becomes one 'cell | cell' paragraph."""
    z = zipfile.ZipFile(path)
    styles, default_style, doc_default = _load_styles(z)
    body = etree.fromstring(z.read("word/document.xml")).find(W + "body")
    paras = []
    for el in body:
        if el.tag == W + "p":
            paras.append(_paragraph_text(el, styles, default_style, doc_default))
        elif el.tag == W + "tbl":
            for tr in el.iter(W + "tr"):
                cells = []
                for tc in tr.findall(W + "tc"):
                    t = " ".join(filter(None, (_paragraph_text(p, styles, default_style, doc_default)
                                               for p in tc.iter(W + "p"))))
                    if t and (not cells or cells[-1] != t):
                        cells.append(t)
                paras.append(" | ".join(cells))
    # drop empty and purely decorative lines (e.g. "::::0000::::")
    return [p for p in paras if p and not re.fullmatch(r"[\W_0०ः]+", p)]


# ---------------------------------------------------------------- section splitting

_NUM = r"^\s*(?:[(\[]?\s*[0-9०-९]{1,2}\s*[)\].]{1,2}|\(\s*[क-हA-Za-z]\s*\))?[\s.\-–:|]*"
_SEP = r"[^:।]{0,30}?(?::|ः|[-–](?:\s|$)|\|)"
_HEADINGS = [
    ("fir", r"प्राथमिकी(?!\s*(?:अभि|के\s|की\s|में))"),
    ("place", r"घटना\s?स्थल"),
    ("prosecution_evidence", r"अभियोजन\s+साक्ष्य"),
    ("witnesses", r"(?:स्वतंत्र\s+|अन्य\s+)?साक्षी"),
    ("investigation_progress", r"अनुसंधान\s+में?ें?\s+प्रगति"),
    ("arrest", r"गिर[फ़्‍]+तारी"),
    ("search_seizure", r"(?:जप्ती|जब्ती|तलाशी)"),
    ("surrender", r"समर्पण"),
    ("defence_evidence", r"(?:प्रतिरक्षा|प्रतिवादी|बचाव)\s+साक्ष्य"),
    ("lapses", r"त्रु(?:टि|टी)"),
    ("conclusion", r"नि[षश]्कर्[षश]"),          # श/ष are often swapped by typists
    ("directions", r"(?:अनु(?:संधानकर्ता|संधानकर्त्ता|0\s*कर्ता|०\s*कर्ता)\s+को\s+)?निर्दे[शष]"),
    ("footer", r"अभियुक्त(?:ों|ो)?\s*(?:की\s+|का\s+|के\s+)?विवरण"),
]
_HEADINGS = [(k, re.compile(_NUM + "(" + v + _SEP + r")\s*-?\s*")) for k, v in _HEADINGS]
_DIRECTIONS_LEADIN = re.compile(r"निर्दे[शष].{0,60}(?::|ः|कि|हैं?|[-–])\s*-?\s*$")
_FOOTER_LINE = re.compile(
    r"^\s*(?:प्रा[0०.]?\s*अभि|प्राथमिकी\s+अभि[0०]|अप्रा|जेल|फिरार|का[0०.]\s*दै|का[ंण]्?ड\s+दैनिकी|दैनिकी\s+सं|"
    r"अनु[0०.]\s*कर्त|अनुसंधानकर्त्?ता\s*[:ः-]|प्रति\s*[:ः-]|प्रति[0०]|ज्ञापांक|"
    r"अभियुक्त(?:ों|ो)?\s*(?:की\s+|का\s+|के\s+)?विवरण)")
# Un-headed direction: "अतः अनुसंधानकर्ता को ... निर्देश/आदेश दिया जाता है ..."
_DIRECTION_SENTENCE = re.compile(
    r"^(?:अतः?|अत:|अतएव)?[\s,]*(?:इस\s+)?(?:का[ंण]्?ड\s+में\s+)?अनु(?:संधानकर्त्?ता|[0०.]\s*कर्त्?ता)\s+को\s")
# Un-headed conclusion: "अबतक के अनुसंधान, ... से यह कांड ... सत्य प्रतीत होता है"
_CONCLUSION_SENTENCE = re.compile(r"^(?:अतः?|अत:|इस\s+प्रकार)?[\s,]*अब\s?तक\s+के\s+अनुसंधान")
OUTPUT_LABELS = ("conclusion", "lapses", "directions")


def _label(paras, implicit_conclusion):
    out, cur, in_header = [], "header", True
    for p in paras:
        heading_len = 0
        for key, rx in _HEADINGS:
            m = rx.match(p[:120])
            if m:
                # The header block lists short fields ("घटनास्थल :- <village>") that reuse section names.
                if in_header and len(p) < 110 and key in ("place", "fir", "footer"):
                    break
                cur, heading_len, in_header = key, m.end(), False
                break
        else:
            if cur in ("conclusion", "lapses") and len(p) < 220 and _DIRECTIONS_LEADIN.search(p):
                cur, heading_len = "directions", len(p)
            elif cur in ("conclusion", "lapses") and _DIRECTION_SENTENCE.match(p) and re.search(r"निर्दे[शष]|आदेश", p):
                cur = "directions"
            elif cur in ("conclusion", "directions", "lapses") and _FOOTER_LINE.match(p):
                cur = "footer"
            elif implicit_conclusion and cur not in OUTPUT_LABELS + ("footer",) and _CONCLUSION_SENTENCE.match(p):
                cur, in_header = "conclusion", False
        out.append((cur, p, heading_len))
    return out


def label_paragraphs(paras):
    """Return [(label, paragraph, heading_len)]; heading_len is how many leading chars are the section heading."""
    out = _label(paras, implicit_conclusion=False)
    if not any(label == "conclusion" for label, _, _ in out):   # some offices never write the "निष्कर्ष" heading
        out = _label(paras, implicit_conclusion=True)
    return out


# ---------------------------------------------------------------- metadata

_CASE = re.compile(
    r"([ऀ-ॿ][ऀ-ॿ0०.()\s]{1,40}?)\s*(?:थाना|ओ[0०.]\s*पी[0०.]?)?[\s,]*का[ंण]?्?[डण]्?ड?\s*"
    r"(?:सं(?:ख्या|[0०.])?|सख्या|संo)\s*[-–:.]*\s*(\d{1,4})\s*[/@]\s*(\d{2,4})")
_DATE = re.compile(r"दि(?:नांक|[0०.])\s*[-–:.]*\s*(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})")
_SECTIONS = re.compile(r"धारा\s*[-–:.]*\s*(.+?)(?=,?\s+(?:में|मे|के\s+अ[ंन]्?तर्गत|अंतर्गत)(?:\s|$)|\s*।|$)")


def parse_case(paras):
    case = {"police_station": None, "case_no": None, "case_year": None, "case_date": None, "sections_of_law": None}
    head, m = "", None
    for i, p in enumerate(paras[:10]):
        if _CASE.search(p):
            # the case line sometimes wraps onto the next paragraph before reaching "धारा"
            head = p if "धारा" in p else " ".join(paras[i:i + 3])
            m = _CASE.search(head)
            break
    if m:
        ps = re.sub(r"^(?:.*कार्यालय|.*दिनांक\S*)\s*", "", m.group(1)).strip(" ।.,-")
        year = int(m.group(3))
        case.update(police_station=ps or None, case_no=int(m.group(2)), case_year=year + 2000 if year < 100 else year)
        tail = head[m.end():]
        d = _DATE.search(tail)
        if d:
            y = int(d.group(3))
            case["case_date"] = "%04d-%02d-%02d" % (y + 2000 if y < 100 else y, int(d.group(2)), int(d.group(1)))
        s = _SECTIONS.search(tail)
        if s:
            case["sections_of_law"] = s.group(1).strip(" ,।")[:300]
    return case


def doc_type(paras):
    # Judge by the title block only: body paragraphs mention "पर्यवेक्षण" in every kind of document.
    title = []
    for p in paras[:7]:
        if len(p) > 450:
            break
        title.append(p)
    head = " ".join(title)
    final = bool(re.search(r"अ[ंन]्?तिम", head))
    if re.search(r"प्रतिवेदन\s*[-–]?\s*0?[2२]", head):
        return "report_2"
    if re.search(r"पर्यवेक्षण\s+टिप्पण", head) or ("पर्यवेक्षण" in head and not final):
        return "supervision_cum_final_report" if final else "supervision_note"
    if final or re.search(r"प्रगति\s+प्रतिवेदन|समीक्षा", head):
        return "final_or_progress_report"
    return "other"


# ---------------------------------------------------------------- driver

def process(path):
    rel = os.path.relpath(path, ROOT).replace("\\", "/")
    try:
        paras = read_docx(path)
    except Exception as e:
        return {"source_file": rel, "error": f"{type(e).__name__}: {e}"}
    if not paras:
        return {"source_file": rel, "error": "empty"}
    labelled = label_paragraphs(paras)
    sections, order = {}, []
    for label, text, heading_len in labelled:
        if label not in sections:
            sections[label] = []
            order.append(label)
        body = text[heading_len:].strip()
        if body:
            sections[label].append(body)
    full = "\n".join(paras)
    return {
        "doc_id": hashlib.sha1(re.sub(r"\s+", "", full).encode("utf-8")).hexdigest()[:16],
        "source_file": rel,
        "doc_type": doc_type(paras),
        "case": parse_case(paras),
        "section_order": order,
        "sections": sections,
        "paragraphs": [list(t) for t in labelled],
        "n_chars": len(full),
        "devanagari_ratio": round(len(re.findall(r"[ऀ-ॿ]", full)) / max(1, len(re.findall(r"\S", full))), 3),
    }


def main():
    # Optional: python 02_extract.py <source-folder-under-project> <output-folder-under-dataset>
    src, out = (sys.argv[1], sys.argv[2]) if len(sys.argv) == 3 else ("data", "supervision")
    src_dirs = [os.path.join(ROOT, src, "word"), os.path.join(ROOT, src, "word_converted")]
    files = [os.path.join(d, f) for d in src_dirs if os.path.isdir(d)
             for f in sorted(os.listdir(d)) if re.search(r"\.doc[xm]$", f, re.I) and not f.startswith("~$")]
    out_dir = os.path.join(ROOT, "dataset", out)
    os.makedirs(out_dir, exist_ok=True)
    ok = err = 0
    with ProcessPoolExecutor() as ex, \
            open(os.path.join(out_dir, "records.jsonl"), "w", encoding="utf-8") as fo, \
            open(os.path.join(out_dir, "extract_errors.jsonl"), "w", encoding="utf-8") as fe:
        for rec in ex.map(process, files, chunksize=40):
            if "error" in rec:
                err += 1
                fe.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                ok += 1
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{ok} records, {err} unreadable, from {len(files)} files")


if __name__ == "__main__":
    main()
