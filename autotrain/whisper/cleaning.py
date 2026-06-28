import re
import unicodedata

# --- unicode standardization tables -------------------------------------------------

# CJK / full-width punctuation that NFKC does NOT fold to ASCII (ideographic full stop,
# enumeration comma, full-width colon/semicolon, CJK brackets and dashes). We map them to
# a single ASCII punctuation set so the model sees consistent punctuation across en/ms/zh.
# (NFKC already handles the full-width comma '，', question '？' and exclamation '！'.)
_CJK_PUNCT = str.maketrans({
    '。': '.', '、': ',', '〜': '~', '～': '~',
    '；': ';', '：': ':', '·': ' ', '・': ' ',
    '「': '"', '」': '"', '『': '"', '』': '"',
    '《': '"', '》': '"', '〈': '"', '〉': '"',
    '【': '(', '】': ')', '〔': '(', '〕': ')',
})

# Curly quotes / dashes / ellipsis -> ASCII so they don't fragment the vocab.
_QUOTES_DASHES = str.maketrans({
    '‘': "'", '’': "'", '“': '"', '”': '"',
    '–': '-', '—': '-', '―': '-', '−': '-',
})

# Zero-width / BiDi / BOM control characters that carry no acoustic content.
_INVISIBLES = re.compile(r'[­​-‏‪-‮⁠﻿]')


def fix_spacing(text):
    quote_pattern = r'"([^"]*)"'
    def fix_quotes(match):
        content = match.group(1).strip()
        return f'"{content}"'

    text = re.sub(quote_pattern, fix_quotes, text)

    paren_pattern = r'\(([^)]*)\)'
    def fix_parens(match):
        content = match.group(1).strip()
        return f'({content})'

    text = re.sub(paren_pattern, fix_parens, text)
    text = re.sub(r'\s+([,\.!?])', r'\1', text)
    return text

def whisper_textcleaning(text):
    # --- unicode standardization (run first, before any regex) ---
    # NFKC folds full-width letters/digits + '，！？' to ASCII and the ideographic space to ' '.
    text = unicodedata.normalize('NFKC', text)
    text = _INVISIBLES.sub('', text)
    text = text.translate(_QUOTES_DASHES)
    text = text.translate(_CJK_PUNCT)
    text = text.replace('…', '...')

    text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
    text = re.sub(r'\b(?:ok|oke|okay|okey|okie)\b', 'OK', text, flags=re.IGNORECASE)
    # nasal hesitations (hmm/mm/erm/um/uh/uhm ...) -> single canonical token
    text = re.sub(r'\b(?:h+m+|u+h*m+|u+h+|erm+|mm+)\b', 'herm', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(a+h+|a+\s*a+)(?=[\s,\.!?]|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\.{2,}', ',', text)
    text = re.sub(r'(?<=\s)-(\w+)\b', r'\1', text)
    text = re.sub(r'\b(\w+)-(?=\s|$)', r'\1', text)
    text = re.sub(r'\b(um|uh|aa|erm|herm)(\s+\1)+', r'\1', text, flags=re.IGNORECASE)
    # collapse repeated end-punctuation ("!!" / "??" / ",,")
    text = re.sub(r'([,!?])\1+', r'\1', text)
    # space after , ! ? unless the next char is whitespace or a CJK character
    text = re.sub(r'([,!?])(?=[^\s一-鿿])', r'\1 ', text)
    # space after a sentence-ending period only before a capital letter (keep decimals,
    # domains and emails intact: 3.30 / i.unify.my / name@gmail.com)
    text = re.sub(r'(?<!\d)\.(?=[A-Z])', '. ', text)
    text = fix_spacing(text)
    # tidy punctuation stranded by removed interjections ("Sekejap. Ah, ..." / "Gun, ah?"):
    text = re.sub(r'\s*,\s*(?=[.!?])', '', text)           # drop a comma sitting before . ! ?
    text = re.sub(r'([.!?])\s*,', r'\1', text)              # drop a comma sitting after . ! ?
    text = re.sub(r'([,.!?])(?:\s*[,.!?])+', r'\1', text)   # collapse any remaining run -> first mark
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[\s,.?!:;]+', '', text)                # strip leading punctuation
    text = re.sub(r'\s*[,:;]+$', '', text)                  # strip a dangling trailing , : ;
    return text.strip()


# --- language detection + whisper formatting --------------------------------------

# CJK ideograph ranges (Unified, Extension-A, Compatibility). Used to decide Chinese
# by character ratio — the bahasa/en fastText model has no `zh` label, so Chinese
# transcripts would otherwise be misdetected as `en`. (Full-width CJK *punctuation*
# is already folded to ASCII by whisper_textcleaning, so only ideographs match here.)
_CJK = re.compile(r'[一-鿿㐀-䶿豈-﫿]')


def chinese_ratio(text):
    """Fraction (0..1) of non-whitespace characters that are CJK ideographs."""
    chars = re.sub(r'\s+', '', text)
    if not chars:
        return 0.0
    return len(_CJK.findall(chars)) / len(chars)


def detect_language(text, lang_model, chinese_threshold=0.5):
    """Whisper language code for `text`:

    - `zh` when CJK ideographs are at least `chinese_threshold` of the characters
      (default 50%), checked first since the fastText model can't see Chinese;
    - otherwise the fastText bahasa/en model
      (`mesolitica/fasttext-language-detection-bahasa-en`): `bahasa` -> `ms`,
      anything else (incl. `english` / `other`) -> `en`.

    `lang_model` is a loaded fastText model.
    """
    if chinese_ratio(text) >= chinese_threshold:
        return "zh"
    line = text.replace("\n", " ").replace("\r", " ").strip()
    if not line:
        return "en"
    labels, _ = lang_model.predict(line, k=10)
    clean = [l.replace("__label__", "") for l in labels]
    top = clean[0]
    if top == "other" and len(clean) > 1:
        top = clean[1]
    return "ms" if top == "bahasa" else "en"


def format_whisper(text, lang_model, task="transcribe"):
    """Standardize + clean `text` (via `whisper_textcleaning`), detect its language,
    and wrap it in Whisper's no-timestamp prompt:

        <|startoftranscript|><|LANG|><|TASK|><|notimestamps|> TEXT<|endoftext|>

    Returns None when the text is empty after cleaning, so the caller can emit the
    empty/silence target instead. Domain-specific marker removal (e.g. dropping
    call-centre `CALL ENDS` tags) should happen before calling this.
    """
    text = whisper_textcleaning(text)
    if not text:
        return None
    lang = detect_language(text, lang_model)
    return f'<|startoftranscript|><|{lang}|><|{task}|><|notimestamps|> {text}<|endoftext|>'


# --- examples / self-test ---------------------------------------------------------
# Verified before -> after pairs over real en / ms / zh call-centre transcripts.
# Run `python cleaning.py` to eyeball the standardization (and catch regressions).
EXAMPLES = [
    # full-width punctuation -> ASCII; Chinese stays space-tight, Latin gets a space
    ('啊， 不行。Incoming，嗯，对吗？', '啊, 不行. Incoming,嗯,对吗?'),
    # full-width digit '７' -> '7'
    ('我的所在地方是在７号。', '我的所在地方是在7号.'),
    # OK spelling variants (incl. Malay Okey / Oke) -> OK
    ('Okey boleh. Ok. Okay. okie.', 'OK boleh. OK. OK. OK.'),
    # hesitations (uhm / uh / mm / hmm / erm) -> canonical "herm"
    ('Uhm. Uh betul. Mm. hmm. erm.', 'herm. herm betul. herm. herm. herm.'),
    # interjection removed + stranded punctuation tidied
    ('Dah boleh on. Ah. OK. ', 'Dah boleh on. OK.'),
    # trailing interjection: keep the stronger '?', drop the dangling comma
    ('Miss Shin Jung Gun, ah?', 'Miss Shin Jung Gun?'),
    # "..." / leading space collapse + Okey -> OK
    ('Memang... reset port. Okey, boleh-boleh.', 'Memang, reset port. OK, boleh-boleh.'),
    # non-speech tag -> empty (caller emits the silence target)
    ('[silence]', ''),
    # PRESERVED: decimals, domains, emails, account numbers, spelled-out letters
    ('slot tomorrow 3.30 PM, refer i.unify.my/oss', 'slot tomorrow 3.30 PM, refer i.unify.my/oss'),
    ('email fulingchow@gmail.com, account 1052-442876', 'email fulingchow@gmail.com, account 1052-442876'),
    ('U-N-I-F-I, K-O-H L-E-E M-O-I', 'U-N-I-F-I, K-O-H L-E-E M-O-I'),
]


if __name__ == '__main__':
    ok = True
    for raw, expected in EXAMPLES:
        got = whisper_textcleaning(raw)
        passed = got == expected
        ok = ok and passed
        print(f'[{"ok" if passed else "XX"}] {raw!r}\n     -> {got!r}')
        if not passed:
            print(f'     != expected {expected!r}')
    print('\nALL EXAMPLES PASS' if ok else '\nSOME EXAMPLES FAILED')
