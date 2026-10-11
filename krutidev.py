"""Kruti Dev 010 (legacy glyph font) -> Unicode Devanagari.

The table follows the widely used Kruti Dev converter, extended with glyph variants seen in this corpus
(curly quotes typed for श्/ष्, Ÿ, Ê, ´, bare ª/î).
"""
import re

_SRC = [
    "ñ", "Q+Z", "sas", "aa", ")Z", "ZZ",
    "å", "ƒ", "„", "…", "†", "‡", "ˆ", "‰", "Š", "‹",
    "¶+", "d+", "[+k", "[+", "x+", "T+", "t+", "M+", "<+", "Q+", ";+", "j+", "u+",
    "Ùk", "Ÿk", "Ù", "Ÿ", "ä", "–", "—", "é", "™", "=kk", "f=k",
    "à", "á", "â", "ã", "ºz", "º", "í", "{k", "{", "=", "«",
    "Nî", "Vî", "Bî", "Mî", "<î", "|", "K", "}",
    "J", "Vª", "Mª", "<ªª", "Nª", "Ø", "Ý", "nzZ", "æ", "ç", "Á", "xz", "#", ":",
    "v‚", "vks", "vkS", "vk", "v", "b±", "Ã", "bZ", "b", "m", "Å", ",s", ",", "_",
    "ô", "d", "Dk", "D", "[k", "[", "x", "Xk", "X", "Ä", "?k", "?", "³",
    "pkS", "p", "Pk", "P", "N", "t", "Tk", "T", ">", "÷", "¥", "´",
    "ê", "ë", "V", "B", "ì", "ï", "M", "<", ".k", ".",
    "r", "Rk", "R", "Fk", "F", ")", "n", "/k", "èk", "/", "Ë", "è", "u", "Uk", "U",
    "i", "Ik", "I", "Q", "¶", "c", "Ck", "C", "Hk", "H", "e", "Ek", "E",
    ";", "¸", "j", "y", "Yk", "Y", "G", "o", "Ok", "O",
    "'k", "‘k", "“k", "”k", "'", "‘", "“", "”", "\"k", "’k", "\"", "’", "l", "Lk", "L", "g",
    "È", "Ê", "z", "ª", "î",
    "Ì", "Í", "Î", "Ï", "Ñ", "Ò", "Ó", "Ô", "Ö", "Ük", "Ü",
    "‚", "ks", "kS", "k", "h", "q", "w", "`", "s", "S",
    "a", "¡", "%", "W", "•", "·", "~j", "~", "\\", "+",
    "^", "*", "Þ", "ß", "(", "¼", "½", "¿", "À", "¾", "A", "-", "&", "Œ", "]", "@",
]
_DST = [
    "॰", "QZ+", "sa", "a", "र्द्ध", "Z",
    "०", "१", "२", "३", "४", "५", "६", "७", "८", "९",
    "फ़्", "क़", "ख़", "ख़्", "ग़", "ज़्", "ज़", "ड़", "ढ़", "फ़", "य़", "ऱ", "ऩ",
    "त्त", "त्त", "त्त्", "त्त्", "क्त", "दृ", "कृ", "न्न", "न्न्", "=k", "f=",
    "ह्न", "ह्य", "हृ", "ह्म", "ह्र", "ह्", "द्द", "क्ष", "क्ष्", "त्र", "त्र्",
    "छ्य", "ट्य", "ठ्य", "ड्य", "ढ्य", "द्य", "ज्ञ", "द्व",
    "श्र", "ट्र", "ड्र", "ढ्र", "छ्र", "क्र", "फ्र", "र्द्र", "द्र", "प्र", "प्र", "ग्र", "रु", "रू",
    "ऑ", "ओ", "औ", "आ", "अ", "ईं", "ई", "ई", "इ", "उ", "ऊ", "ऐ", "ए", "ऋ",
    "क्क", "क", "क", "क्", "ख", "ख्", "ग", "ग", "ग्", "घ", "घ", "घ्", "ङ",
    "चै", "च", "च", "च्", "छ", "ज", "ज", "ज्", "झ", "झ्", "ञ", "ञ",
    "ट्ट", "ट्ठ", "ट", "ठ", "ड्ड", "ड्ढ", "ड", "ढ", "ण", "ण्",
    "त", "त", "त्", "थ", "थ्", "द्ध", "द", "ध", "ध", "ध्", "ध्", "ध्", "न", "न", "न्",
    "प", "प", "प्", "फ", "फ्", "ब", "ब", "ब्", "भ", "भ्", "म", "म", "म्",
    "य", "य्", "र", "ल", "ल", "ल्", "ळ", "व", "व", "व्",
    "श", "श", "श", "श", "श्", "श्", "श्", "श्", "ष", "ष", "ष्", "ष्", "स", "स", "स्", "ह",
    "ीं", "ीZ", "्र", "्र", "्य",
    "द्द", "ट्ट", "ट्ठ", "ड्ड", "कृ", "भ", "्य", "ड्ढ", "झ्", "श", "श्",
    "ॉ", "ो", "ौ", "ा", "ी", "ु", "ू", "ृ", "े", "ै",
    "ं", "ँ", "ः", "ॅ", "ऽ", "ऽ", "्र", "्", "?", "़",
    "‘", "’", "“", "”", ";", "(", ")", "{", "}", "=", "।", ".", "-", "॰", ",", "/",
]
assert len(_SRC) == len(_DST)

_C = r"[क-हक़-य़]़?"                      # one consonant (with optional nukta)
_CLUSTER = rf"{_C}(?:्{_C})*"              # conjunct cluster
_MATRA = r"[ािीुूृेैोौंँःॅॉ़]*"

_I = re.compile(rf"f({_CLUSTER})")                 # ि is typed before its consonant
_I_REPH = re.compile(rf"Æ({_CLUSTER})")            # ि + reph
_I_ANUSVARA = re.compile(rf"Ç({_CLUSTER})")        # ि + anusvara
_I_REPH_ANUSVARA = re.compile(rf"É({_CLUSTER})")
_REPH = re.compile(rf"({_CLUSTER})({_MATRA})Z")    # reph is typed after its syllable

_CLEANUP = [
    ("ाे", "ो"), ("ाै", "ौ"), ("अा", "आ"), ("आे", "ओ"), ("आै", "औ"), ("एे", "ऐ"), ("ाॅ", "ॉ"),
    ("ंु", "ुं"), ("ंू", "ूं"), ("ंा", "ां"), ("ंी", "ीं"), ("ंे", "ें"), ("ंै", "ैं"), ("ंो", "ों"), ("ंौ", "ौं"),
    ("ाें", "ों"), ("ाैं", "ौं"), ("्ा", ""), ("ेे", "े"), ("ाा", "ा"), ("ंं", "ं"),
]


def kruti_to_unicode(text: str) -> str:
    if not text.strip():
        return text
    text = text.replace("±", "Zं").replace(" ", " ")
    for src, dst in zip(_SRC, _DST):
        if src in text:
            text = text.replace(src, dst)
    text = _I_REPH_ANUSVARA.sub(r"र्\1िं", text)
    text = _I_REPH.sub(r"र्\1ि", text)
    text = _I_ANUSVARA.sub(r"\1िं", text)
    text = _I.sub(r"\1ि", text)
    text = _REPH.sub(r"र्\1\2", text)
    for src, dst in _CLEANUP:
        text = text.replace(src, dst)
    # '%' is visarga in Kruti Dev, but before '-' or after a space it is being used as a colon.
    text = re.sub(r"\s*ः\s*-", " :-", text)
    text = re.sub(r"(?<=\s)ः", ":", text)
    return text


_LEGACY_FONT = re.compile(r"kruti|dev\s*lys|krishna|chanakya|shusha|walkman", re.I)
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
# Very common Kruti Dev word forms; used to spot legacy text whose font can't be resolved.
_KRUTI_HINT = re.compile(r"\b(dk|ds|dh|esa|gS|gSa|fd|ls|dks|ij|,oa|Fkkuk|dkaM|dk.M|fnukad|/kkjk|ftyk)\b|%&|¼|½")


def is_legacy_font(name) -> bool:
    return bool(name and _LEGACY_FONT.search(name))


def looks_like_kruti(text: str) -> bool:
    if _DEVANAGARI.search(text):
        return False
    return len(_KRUTI_HINT.findall(text)) >= max(1, len(text) // 120)
