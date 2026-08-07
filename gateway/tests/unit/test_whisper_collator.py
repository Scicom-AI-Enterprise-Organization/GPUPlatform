"""The Whisper label collator must strip the leading <|startoftranscript|>.

`WhisperForConditionalGeneration.shift_tokens_right` builds the decoder input by
prepending `decoder_start_token_id` (= `<|startoftranscript|>`, 50258). If the
labels ALSO start with that token, the model is trained to emit it as its first
generated token — but `generate()` has already consumed it and expects a language
token next. The model diverges from the generation contract the longer it trains.

This failed silently for weeks because the original check compared against
`tokenizer.bos_token_id`, which for Whisper is `<|endoftext|>` (50257), NOT
`<|startoftranscript|>` (50258) — so the strip never fired. Neither guard rail
catches it: `eval_loss` scores the self-consistent shifted sequence and looks
fine, and `compute_metrics` decodes with `skip_special_tokens=True`, which hides
the stray token from WER/CER. The only symptom is WER peaking at the FIRST eval
and then collapsing to 80-95 as training proceeds.

The trainer is a standalone script shipped to the box by SFTP, and the collator is
a closure inside `train()`, so it can't be imported and called — these pin the
source-level invariant instead, the same way the augment registry-drift test does.
"""
import pathlib
import re

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_TRAINER = _ROOT / "gateway" / "gateway" / "training" / "whisper_finetune.py"

# Real Whisper ids — the whole point is that these two differ.
SOT = 50258   # <|startoftranscript|>  == decoder_start_token_id
EOT = 50257   # <|endoftext|>          == bos_token_id == eos_token_id == pad_token_id


def _source() -> str:
    return _TRAINER.read_text(encoding="utf-8")


def test_label_strip_does_not_use_bos_token_id():
    """`bos_token_id` is <|endoftext|> for Whisper — comparing labels[:, 0] against
    it is always False, so the strip silently never runs."""
    src = _source()
    strip_lines = [
        ln for ln in src.splitlines()
        if "lab[:, 0]" in ln and "==" in ln
    ]
    assert strip_lines, "the label-strip guard disappeared from the collator"
    for ln in strip_lines:
        assert "bos_token_id" not in ln, (
            "collator compares the first label token against tokenizer.bos_token_id "
            f"(<|endoftext|>, {EOT}) — it must compare against decoder_start_token_id "
            f"(<|startoftranscript|>, {SOT}), else the strip never fires: {ln.strip()}"
        )


def test_label_strip_uses_decoder_start_id():
    src = _source()
    assert re.search(r"lab\[:, 0\] == _decoder_start_id", src), (
        "the collator must strip labels[:, 0] when it equals the decoder start token"
    )
    # …and that id must come from the model config, with the SOT token as fallback.
    assert re.search(r"_decoder_start_id\s*=\s*\(", src)
    assert "decoder_start_token_id" in src
    assert "<|startoftranscript|>" in src


def test_strip_rule_semantics():
    """The rule itself: a label row that starts with SOT loses exactly that token;
    one that already starts with a language token is left alone."""
    def apply(row):
        return row[1:] if row[0] == SOT else row

    preformatted = [SOT, 50259, 50360, 50364, 7751, EOT]   # <|sot|><|en|><|transcribe|><|notimestamps|> hi<|eot|>
    assert apply(preformatted) == [50259, 50360, 50364, 7751, EOT]

    already_stripped = [50259, 50360, 50364, 7751, EOT]
    assert apply(already_stripped) == already_stripped

    # The old, broken comparison — proof it is a no-op on real Whisper labels.
    assert preformatted[0] != EOT, "labels start with SOT, so an == bos_token_id test never fires"
