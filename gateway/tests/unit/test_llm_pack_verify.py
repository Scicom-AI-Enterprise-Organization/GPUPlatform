"""llm_pack post-hoc verifiers — the datasets 'Verify pack' button's pure core.

check_bin re-checks a written bin's invariants + assistant mask; the tool-call
scanners catch the gemma raw-JSON-args regression. No stack, no tokenizer."""
from gateway.llm_pack import (
    check_bin,
    count_toolcalls,
    find_raw_json_toolcalls,
)


def test_check_bin_sound_bin():
    # two docs of len 2; position_ids reset per doc; a strict subset trained.
    rec = {
        "input_ids": [10, 11, 12, 13],
        "labels": [-100, 11, -100, 13],
        "position_ids": [0, 1, 0, 1],
        "attention_mask": [2, 2],
    }
    c = check_bin(rec)
    assert c["lengths_ok"] and c["position_ids_ok"]
    assert c["assistant_masked"] and c["trained"] == 2
    assert c["docs"] == 2 and c["tokens"] == 4 and c["have_labels"]


def test_check_bin_flags_length_mismatch():
    rec = {"input_ids": [1, 2, 3], "labels": [1, 2], "position_ids": [0, 1, 2],
           "attention_mask": [3]}
    assert check_bin(rec)["lengths_ok"] is False


def test_check_bin_flags_bad_position_ids():
    # position_ids NOT reset per doc (0..3 across two docs) → not sound.
    rec = {"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4],
           "position_ids": [0, 1, 2, 3], "attention_mask": [2, 2]}
    assert check_bin(rec)["position_ids_ok"] is False


def test_check_bin_all_trained_is_not_masked():
    # labels == input_ids → the gemma no-`{% generation %}` regression: trains the
    # WHOLE sequence, so assistant_masked must be False.
    rec = {"input_ids": [1, 2, 3], "labels": [1, 2, 3], "position_ids": [0, 1, 2],
           "attention_mask": [3]}
    c = check_bin(rec)
    assert c["assistant_masked"] is False and c["trained"] == 3


def test_check_bin_no_labels_column():
    rec = {"input_ids": [1, 2], "attention_mask": [2]}
    c = check_bin(rec)
    assert c["have_labels"] is False and c["assistant_masked"] is False


def test_raw_json_toolcall_is_gemma_only_defect():
    broken = 'foo <|tool_call>call:query{"msisdn": "60123"}<tool_call|>'
    native = 'foo <|tool_call>call:query{msisdn:<|"|>60123<|"|>}<tool_call|>'
    assert find_raw_json_toolcalls(broken, "gemma") == 1
    assert find_raw_json_toolcalls(native, "gemma") == 0
    # other archs use JSON args natively → never a defect
    assert find_raw_json_toolcalls(broken, "qwen") == 0
    assert find_raw_json_toolcalls(broken, "") == 0


def test_count_toolcalls():
    assert count_toolcalls("call:a{} then call:b{} and call:c.d{}") == 3
    assert count_toolcalls("no calls here") == 0
    assert count_toolcalls("") == 0
