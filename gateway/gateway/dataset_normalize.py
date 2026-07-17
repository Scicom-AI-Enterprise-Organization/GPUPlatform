"""LLM transcription normalization (constrained respelling) for S3 audio datasets.

Ports the reference experiment `ucc_ai_research/speech/stt/llm_normalize_experiment.py`
into a reusable gateway helper. The goal: canonicalise transcription-CONVENTION
noise (particle/filler spellings la/lah, ya/ye, Malay affix spacing, zh spacing…)
WITHOUT changing what was said, so a downstream STT finetune trains on a consistent
convention instead of fighting spelling variance.

Two guards keep it fail-safe (a bad normalization is DROPPED, the original text
kept):
  1. a deterministic structural check (`validate_edits`) — the normalized text must
     decompose into identical words, whitelisted 1:1 respells, and letter-preserving
     affix joins; anything else (added/deleted/renumbered/romanized-zh word) is a
     violation. This is free (no LLM) and always runs.
  2. an optional LLM-as-judge pass (few-shot JSON verdict) for the subtler cases.

The prompt/few-shots/whitelist below are copied verbatim from the tuned reference —
edit them there and here together.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

import httpx

# ---------------------------------------------------------------- prompts ----

# Canonical convention (mirror of the reference's Phase-0 sketch). Respell-only.
CONVENTION = """\
Canonical spellings (respell these variants ONLY, wherever they appear as a whole word):
- lah / laa / laaa      -> la
- ye / yah / yer / yeah -> ya
- ah / aa / aaa / a (ONLY when it is a standalone hesitation, not the article 'a') -> ah
- emm / em / err / erm / hmm / umm / um / mmm / uh / uhh / uhm -> herm
- okay / oke / okey / okie -> ok
- k (a standalone letter 'k'/'K' used as shorthand for OK) -> ok
- takde / xde           -> tiada
- nak                   -> nak   (keep; do NOT expand to hendak)
- x (standalone, meaning 'tak') -> tak
Malay morphology — JOIN wrongly-spaced affixes (context-dependent, adjacent words only):
- verbal suffix -kan:  "peduli kan aku" -> "pedulikan aku"   (verb + object)
  BUT the question-tag/emphasis particle stays separate: "betul kan?" -> "betul kan?"
- possessive/definite -nya: "rumah nya" -> "rumahnya"
- passive prefix di-: "di beritahu" -> "diberitahu"
  BUT the locative preposition stays separate: "di rumah" -> "di rumah"
English orthography (respell/join only — NEVER fix tense, articles or word choice;
Manglish stays Manglish):
- orait / oraite / alrite -> alright
- mis-spaced compounds JOIN: "can not" -> "cannot", "every thing" -> "everything",
  "some one" -> "someone", "any way" (meaning anyway) -> "anyway"
Chinese orthography (verbatim otherwise — never convert traditional/simplified,
never fix grammar or particles):
- remove spaces BETWEEN Chinese characters: "我 要 报案" -> "我要报案"
  (keep the single space where Chinese meets a Latin word: "我帮你 check")
- full-width punctuation -> half-width: "，" -> ","
- ⚠ The respelling whitelist above applies to LATIN-script words ONLY. Chinese
  filler/particle characters (啊 嗯 哦 咯 啦 哈 呃 唉 嘛 咧) are NOT the Latin
  fillers — NEVER romanize, translate or respell them: 啊 stays 啊 (never "ah"),
  嗯嗯 stays 嗯嗯 (never "herm"), 啦 stays 啦 (never "la").
NUMBERS: NEVER touch numbers in any way — digits stay digits, spelled-out numbers
stay spelled-out, full-width digits stay full-width, Chinese numerals stay Chinese.
Rules (STRICT):
1. NEVER add, remove, reorder or translate words. Allowed changes are ONLY the
   whitelisted respellings (one word -> one word) and the affix JOINS above
   (two adjacent words -> one word, letters unchanged).
2. NEVER fix grammar beyond the affix joins, and never fix casing or punctuation.
   Verbatim transcript stays verbatim.
3. If a word is not covered above, copy it unchanged.
4. If unsure, keep the original word."""

NORM_FEWSHOT = [
    # (input, output) — few-shot pairs, Malaysian call-center register.
    ("OK Madam so you can receive the SMS lah, please share your rating ye",
     "OK Madam so you can receive the SMS la, please share your rating ya"),
    ("emm boleh saya tahu nombor akaun awak, okay saya check dulu yah",
     "herm boleh saya tahu nombor akaun awak, ok saya check dulu ya"),
    ("K terima kasih. Sama.",
     "ok terima kasih. Sama."),
    ("hold on aa, the line got problem laa, hmm I call you back",
     "hold on ah, the line got problem la, herm I call you back"),
    ("Ini orang tak nak peduli kan aku, betul kan? Nanti saya bagi tahu rumah nya kat mana lah",
     "Ini orang tak nak pedulikan aku, betul kan? Nanti saya bagi tahu rumahnya kat mana la"),
    ("Orait I can not see the reference number, every thing blank one, wait ah",
     "Alright I cannot see the reference number, everything blank one, wait ah"),
    ("好 的 先生 ，你 等 一下 ya, 我 帮 你 check １２３４",
     "好的先生, 你等一下 ya, 我帮你 check １２３４"),
    ("啊 好的 ，嗯嗯 ，就 这样 啦 。OK thank you lah",
     "啊好的, 嗯嗯, 就这样啦。OK thank you la"),
]

JUDGE_FEWSHOT = [
    ("saya nak tanya lah boleh tak", "saya nak tanya la boleh tak",
     {"ok": True, "violations": []}),
    ("dia tak peduli kan aku pun, betul kan", "dia tak pedulikan aku pun, betul kan",
     {"ok": True, "violations": []}),
    ("我 要 问 一下 this one can not claim meh", "我要问一下 this one cannot claim meh",
     {"ok": True, "violations": []}),
    ("嗯嗯 好 的 啦", "herm herm 好的 la",
     {"ok": False, "violations": ["Chinese fillers 嗯嗯/啦 were romanized — must stay 嗯嗯/啦"]}),
    ("please hold on ye I check the system", "please hold on I check the system",
     {"ok": False, "violations": ["word 'ye' was DELETED instead of respelled to 'ya'"]}),
    ("the bill is 25 ringgit lah", "the bill is twenty five ringgit la",
     {"ok": False, "violations": ["'25' was rewritten to 'twenty five' — not a whitelisted respelling"]}),
]


def _normalize_prompt(text: str) -> list:
    msgs = [{"role": "system", "content":
             "You normalize Malaysian call-center transcripts (Malay/English code-switched) "
             "to a canonical spelling convention. You ONLY respell whitelisted particle/filler "
             "variants. You never change anything else.\n\n" + CONVENTION +
             '\n\nReply with JSON only: {"text": "<normalized transcript>"}'}]
    for src, dst in NORM_FEWSHOT:
        msgs.append({"role": "user", "content": src})
        msgs.append({"role": "assistant", "content": json.dumps({"text": dst})})
    msgs.append({"role": "user", "content": text})
    return msgs


def _judge_prompt(original: str, normalized: str) -> list:
    msgs = [{"role": "system", "content":
             "You are a strict validator for a transcript-normalization step. The ONLY allowed "
             "changes are the whitelisted respellings (one word -> one word) and the Malay affix "
             "JOINS (two adjacent words -> one word, letters unchanged) below; every other word "
             "must be byte-identical.\n\n" + CONVENTION +
             "\n\nPure casing differences (e.g. 'OK' vs 'ok', 'Yeah' -> 'ya'/'Ya') are "
             "ACCEPTABLE — the downstream scorer is case-insensitive; do not flag them.\n"
             "\nThink through the comparison silently first. Then reply with JSON only: "
             '{"ok": true|false, "violations": [...]} — each violation a single factual '
             "clause of at most 15 words, NO reasoning, no second-guessing. If after "
             "checking you find no real violation, reply exactly "
             '{"ok": true, "violations": []}.'}]
    for o, n, verdict in JUDGE_FEWSHOT:
        msgs.append({"role": "user", "content": f"ORIGINAL: {o}\nNORMALIZED: {n}"})
        msgs.append({"role": "assistant", "content": json.dumps(verdict)})
    msgs.append({"role": "user", "content": f"ORIGINAL: {original}\nNORMALIZED: {normalized}"})
    return msgs


def extract_json(s: str) -> dict:
    """Parse the FIRST JSON object in a (possibly fenced/chatty) LLM reply."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", s):
        try:
            obj, _ = dec.raw_decode(s[m.start():])
            return obj
        except Exception:  # noqa: BLE001
            continue
    raise ValueError(f"no JSON in: {s[:200]!r}")

# ------------------------------------------------------- deterministic guard --

RESPELL: dict[str, str] = {}
for _variants, _canon in ((("lah", "laa", "laaa"), "la"), (("ye", "yah", "yer", "yeah"), "ya"),
                          (("aa", "aaa", "ah"), "ah"), (("okay", "oke", "okey", "okie", "ok", "k"), "ok"),
                          (("emm", "em", "err", "erm", "hmm", "umm", "um", "mmm",
                            "uh", "uhh", "uhm"), "herm"),
                          (("takde", "xde"), "tiada"), (("x",), "tak"),
                          (("orait", "oraite", "alrite"), "alright")):
    for _v in _variants:
        RESPELL[_v] = _canon


def _strip(w: str) -> str:
    return re.sub(r"^\W+|\W+$", "", unicodedata.normalize("NFKC", w)).lower()


_CJK = re.compile(r"([㐀-䶿一-鿿豈-﫿])")


def _tokens(text: str) -> list:
    """NFKC-fold, put every CJK char in its own token (so zh spacing changes are
    invisible), turn ALL punctuation (incl. 。，) into separators, lowercase."""
    text = _CJK.sub(r" \1 ", unicodedata.normalize("NFKC", text))
    text = re.sub(r"[^\w\s]", " ", text)
    return text.lower().split()


def validate_edits(orig: str, norm: str, max_join: int = 8) -> list:
    """Deterministic structural check: norm must be reproducible from orig using only
    (a) identical words (casing/punct/zh-spacing-insensitive), (b) whitelisted
    respells (1:1), (c) joins of <=max_join adjacent words whose letters
    concatenate unchanged. CJK chars are compared one-by-one, so romanizing a
    Chinese filler (嗯 -> herm) is a violation, while re-spacing zh text is not."""
    ow, nw = _tokens(orig), _tokens(norm)
    i = j = 0
    violations = []
    while j < len(nw):
        if i >= len(ow):
            violations.append(f"extra word(s) added near '{nw[j]}'")
            break
        o, n = _strip(ow[i]), _strip(nw[j])
        if o == n or RESPELL.get(o) == n:
            i += 1
            j += 1
            continue
        joined = False
        for k in range(2, max_join + 1):
            if "".join(_strip(w) for w in ow[i:i + k]) == n:
                i += k
                j += 1
                joined = True
                break
        if joined:
            continue
        violations.append(f"'{' '.join(ow[i:i+2])}' -> '{nw[j]}' is not a whitelisted edit")
        i += 1
        j += 1
    if not violations and i < len(ow):
        violations.append(f"word(s) deleted near '{' '.join(ow[i:i+3])}'")
    return violations

# ---------------------------------------------------------------- client -----


@dataclass
class NormResult:
    text: str            # the text to keep (normalized if accepted, else original)
    normalized: str      # the raw LLM normalization (for logging/inspection)
    ok: bool             # accepted (passed guards) — i.e. `text == normalized`
    changed: bool        # accepted AND differs from the original
    violations: list     # why it was rejected (deterministic + judge), if any


class Normalizer:
    """Constrained transcription normalizer backed by an OpenAI-compatible chat
    endpoint. Thread-safe: `normalize_one` is a pure function of its argument (the
    httpx client is created per call), so it can be driven from a ThreadPoolExecutor."""

    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None,
                 timeout: float = 120.0, retries: int = 3, judge: bool = True):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or ""
        self.timeout = timeout
        self.retries = retries
        self.judge = judge

    def _chat(self, messages: list, temperature: float = 0.0) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {"model": self.model, "messages": messages,
                "temperature": temperature, "max_tokens": 1024}
        url = f"{self.base_url}/chat/completions"
        last: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                r = httpx.post(url, json=body, headers=headers, timeout=self.timeout)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001
                last = e
        raise RuntimeError(f"LLM call failed after {self.retries} tries: {last}")

    def normalize_one(self, text: str) -> NormResult:
        """Normalize one transcript, fail-safe. An empty/whitespace transcript is
        returned unchanged (no LLM call). A normalization that fails the
        deterministic guard (always) or the LLM judge (when enabled) is REJECTED —
        the original text is kept and the reason recorded."""
        if not (text or "").strip():
            return NormResult(text=text, normalized=text, ok=True, changed=False, violations=[])
        norm = extract_json(self._chat(_normalize_prompt(text)))["text"]
        # A no-op normalization is trivially safe: skip both guards (an LLM judge
        # occasionally hallucinates a "violation" when asked to compare identical
        # strings, which would wrongly reject an already-clean transcript).
        if norm == text:
            return NormResult(text=text, normalized=norm, ok=True, changed=False, violations=[])
        violations = list(validate_edits(text, norm))
        ok = not violations
        if ok and self.judge:
            verdict = extract_json(self._chat(_judge_prompt(text, norm)))
            if not verdict.get("ok"):
                ok = False
                violations.extend(verdict.get("violations") or ["judge rejected the normalization"])
        kept = norm if ok else text
        return NormResult(text=kept, normalized=norm, ok=ok,
                          changed=ok and kept != text, violations=violations)
