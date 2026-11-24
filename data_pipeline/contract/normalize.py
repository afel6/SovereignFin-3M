"""Reference text normalization N (docs/contracts.md, "Text normalization").

N(s) applies, in this order:
1. delete code points of category Cc or Cf that are not White_Space;
2. map every Pd character and U+2212 to "-", U+2018/U+2019 to "'", U+201C/U+201D to '"';
3. remove the tag pattern  < /? [ASCII letter or "."] [^<>]* >  until none is left;
4. collapse each run of the White_Space code points below to one space and strip both ends;
5. Unicode NFC (not NFKC).
"""

import re
import sys
import unicodedata

# The Unicode White_Space property, listed explicitly. str.isspace() and \s also match
# U+001C-U+001F, which are Cc and not White_Space, so step 1 deletes them instead.
WHITE_SPACE = frozenset(
    [*range(0x09, 0x0E), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B)]
    + [0x2028, 0x2029, 0x202F, 0x205F, 0x3000]
)
MINUS_SIGN = 0x2212
QUOTE_MAP = {0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"'}


def _build_tables() -> tuple[dict[int, None], dict[int, str]]:
    deletions: dict[int, None] = {}
    replacements: dict[int, str] = {MINUS_SIGN: "-", **QUOTE_MAP}
    for code_point in range(sys.maxunicode + 1):
        category = unicodedata.category(chr(code_point))
        if category in ("Cc", "Cf") and code_point not in WHITE_SPACE:
            deletions[code_point] = None
        elif category == "Pd":
            replacements[code_point] = "-"
    return deletions, replacements


_DELETE_TABLE, _REPLACE_TABLE = _build_tables()
TAG_PATTERN = re.compile(r"</?[A-Za-z.][^<>]*>")
_WHITE_SPACE_RUN = re.compile("[" + "".join(re.escape(chr(c)) for c in sorted(WHITE_SPACE)) + "]+")


def _remove_tags(text: str) -> str:
    while True:
        stripped = TAG_PATTERN.sub("", text)
        if stripped == text:
            return text
        text = stripped


def normalize_text(s: str) -> str:
    """Return N(s); see the module docstring for the five steps."""
    text = s.translate(_DELETE_TABLE)
    text = text.translate(_REPLACE_TABLE)
    text = _remove_tags(text)
    text = _WHITE_SPACE_RUN.sub(" ", text).strip(" ")
    return unicodedata.normalize("NFC", text)
