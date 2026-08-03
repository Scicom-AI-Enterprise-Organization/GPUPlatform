"""llm_pack — DPO preference-pair packing.

Each test here pins a way a DPO pack can go wrong SILENTLY: the pack succeeds,
the bins are well-formed, and the trained objective is not the one intended.
No tokenizer download — `_FakeTok` renders a message list deterministically and
reports a generation mask the way a `{% generation %}` template does.
"""
import json

import pytest

from gateway import llm_pack as LP

IGN = LP.IGNORE_INDEX


# --- a deterministic stand-in for a chat template -----------------------------
# One token per word, plus a role marker. `assistant_masks` marks ONLY assistant
# words — i.e. what gemma-4's injected {% generation %} blocks produce.
class _FakeTok:
    def __init__(self, with_mask: bool = True):
        self.with_mask = with_mask
        self.chat_template = "stub"
        self.seen_tools = []

    def apply_chat_template(self, messages, tokenize=True, return_dict=True,
                            return_assistant_tokens_mask=False, chat_template=None,
                            tools=None, add_generation_prompt=False, **kw):
        if return_assistant_tokens_mask and not self.with_mask:
            raise TypeError("return_assistant_tokens_mask unsupported")
        self.seen_tools.append(tools)
        ids, mask = [], []
        if tools:
            for t in tools:
                ids.append(hash(json.dumps(t, sort_keys=True)) % 1000 + 9000)
                mask.append(0)
        for m in messages:
            role = m.get("role")
            for w in (str(m.get("content") or "") or "_").split():
                ids.append(abs(hash(w)) % 5000 + 1)
                mask.append(1 if role in ("assistant", "model") else 0)
        if add_generation_prompt:
            ids.append(7777)
            mask.append(0)
        out = {"input_ids": ids}
        if return_assistant_tokens_mask:
            out["assistant_masks"] = mask
        return out


def _rowget(row):
    return lambda k: row.get(k)


# Vocabulary is disjoint PER ROLE (env* / asst* / ask*) so a token id alone says
# which span it came from — the tests assert on that rather than on offsets.
PROMPT = [{"role": "user", "content": "ask1 ask2 ask3"}]
# An agentic trajectory: assistant turn, ENVIRONMENT tool result, assistant turn.
CHOSEN = [
    {"role": "assistant", "content": "asstC1 asstC2"},
    {"role": "tool", "content": "envC1 envC2 envC3 envC4"},
    {"role": "assistant", "content": "asstC3 asstC4 asstC5"},
]
REJECTED = [
    {"role": "assistant", "content": "asstR1 asstR2"},
    {"role": "tool", "content": "envR1 envR2 envR3"},
    {"role": "assistant", "content": "asstR3 asstR4"},
    {"role": "tool", "content": "envR4 envR5"},
    {"role": "assistant", "content": "asstR5"},
]


# --- extract_pair_messages ----------------------------------------------------

def test_json_string_message_lists_are_parsed_not_wrapped_as_a_response():
    """⚠ The regression that motivated this file. A parquet stores chosen/rejected
    message lists as JSON *strings*; dispatching on `isinstance(cell, str)` sent them
    down the plain-response branch, which packed the raw JSON TEXT as one assistant
    turn — training the model to emit `[{"role": "assistant", ...}]` verbatim."""
    row = {"prompt": json.dumps(PROMPT), "chosen": json.dumps(CHOSEN),
           "rejected": json.dumps(REJECTED)}
    trip = LP.extract_pair_messages(_rowget(row), chosen_field="chosen",
                                    rejected_field="rejected", prompt_field="prompt",
                                    arch="generic", all_reasoning=True)
    assert trip is not None
    prompt, chosen, rejected = trip
    assert prompt == PROMPT
    assert chosen == PROMPT + CHOSEN
    assert rejected == PROMPT + REJECTED
    # nothing anywhere may carry a serialized message list as its content
    for m in chosen + rejected:
        assert not str(m.get("content") or "").lstrip().startswith('[{"role"')


def test_continuation_sides_may_differ_in_length():
    """An agentic pair's two sides are whole trajectories — different turn counts.
    The ultrafeedback-shape equal-length + identical-prefix check would drop them
    all, packing 0 bins."""
    row = {"prompt": PROMPT, "chosen": CHOSEN, "rejected": REJECTED}
    trip = LP.extract_pair_messages(_rowget(row), chosen_field="chosen",
                                    rejected_field="rejected", prompt_field="prompt",
                                    arch="generic", all_reasoning=True)
    assert trip is not None
    _, chosen, rejected = trip
    assert len(chosen) != len(rejected)


def test_a_side_that_repeats_the_prompt_is_not_doubled():
    row = {"prompt": PROMPT, "chosen": PROMPT + CHOSEN, "rejected": PROMPT + REJECTED}
    prompt, chosen, rejected = LP.extract_pair_messages(
        _rowget(row), chosen_field="chosen", rejected_field="rejected",
        prompt_field="prompt", arch="generic", all_reasoning=True)
    assert chosen == PROMPT + CHOSEN
    assert rejected == PROMPT + REJECTED


def test_ultrafeedback_shape_still_works_without_a_prompt_column():
    shared = [{"role": "user", "content": "hi"}]
    row = {"chosen": shared + [{"role": "assistant", "content": "good answer"}],
           "rejected": shared + [{"role": "assistant", "content": "bad answer"}]}
    prompt, chosen, rejected = LP.extract_pair_messages(
        _rowget(row), chosen_field="chosen", rejected_field="rejected",
        prompt_field=None, arch="generic", all_reasoning=True)
    assert prompt == shared and len(chosen) == len(rejected) == 2


def test_plain_response_strings_still_work():
    row = {"prompt": "why is the sky blue", "chosen": "rayleigh scattering",
           "rejected": "because of the ocean"}
    prompt, chosen, rejected = LP.extract_pair_messages(
        _rowget(row), chosen_field="chosen", rejected_field="rejected",
        prompt_field="prompt", arch="generic", all_reasoning=True)
    assert prompt == [{"role": "user", "content": "why is the sky blue"}]
    assert chosen[-1] == {"role": "assistant", "content": "rayleigh scattering"}


def test_mismatched_prompts_are_dropped_without_a_prompt_column():
    row = {"chosen": [{"role": "user", "content": "a"},
                      {"role": "assistant", "content": "x"}],
           "rejected": [{"role": "user", "content": "DIFFERENT"},
                        {"role": "assistant", "content": "y"}]}
    assert LP.extract_pair_messages(_rowget(row), chosen_field="chosen",
                                    rejected_field="rejected", prompt_field=None,
                                    arch="generic", all_reasoning=True) is None


# --- tokenize_pair ------------------------------------------------------------

def _scored(ids, targets):
    """Decode-free view of what the DPO loss actually sums over."""
    return [ids[j + 1] for j, t in enumerate(targets) if t != IGN]


def test_environment_tokens_are_excluded_from_the_scored_region():
    """The DPO log-ratio must cover only tokens the POLICY emits. A `role: tool`
    result is environment output: scoring it rewards the model for the text that
    came back on the chosen side and penalizes the text on the rejected side."""
    tok = _FakeTok()
    c_ids, c_t, r_ids, r_t = LP.tokenize_pair(
        tok, PROMPT, PROMPT + CHOSEN, PROMPT + REJECTED, mask_env=True)
    env = _FakeTok().apply_chat_template(
        [CHOSEN[1]], return_assistant_tokens_mask=False)["input_ids"]
    assert env, "fixture sanity: the tool turn must produce tokens"
    assert not (set(env) & set(_scored(c_ids, c_t))), "tool-result tokens are scored"
    # ...while the assistant turns ARE scored
    asst = _FakeTok().apply_chat_template(
        [CHOSEN[2]], return_assistant_tokens_mask=False)["input_ids"]
    assert set(asst) & set(_scored(c_ids, c_t))


def test_mask_env_false_scores_the_whole_completion():
    tok = _FakeTok()
    c_ids, c_t, _, _ = LP.tokenize_pair(
        tok, PROMPT, PROMPT + CHOSEN, PROMPT + REJECTED, mask_env=False)
    env = _FakeTok().apply_chat_template(
        [CHOSEN[1]], return_assistant_tokens_mask=False)["input_ids"]
    assert set(env) & set(_scored(c_ids, c_t))


def test_a_template_without_a_generation_mask_degrades_to_full_completion():
    """Only gemma-4 carries an injected {% generation %} block today. Every other
    arch must keep the previous behaviour rather than score nothing."""
    tok = _FakeTok(with_mask=False)
    c_ids, c_t, r_ids, r_t = LP.tokenize_pair(
        tok, PROMPT, PROMPT + CHOSEN, PROMPT + REJECTED, mask_env=True)
    assert len(_scored(c_ids, c_t)) > 0 and len(_scored(r_ids, r_t)) > 0
    env = _FakeTok().apply_chat_template(
        [CHOSEN[1]], return_assistant_tokens_mask=False)["input_ids"]
    assert set(env) & set(_scored(c_ids, c_t))


def test_prompt_tokens_are_never_scored():
    tok = _FakeTok()
    c_ids, c_t, r_ids, r_t = LP.tokenize_pair(
        tok, PROMPT, PROMPT + CHOSEN, PROMPT + REJECTED, mask_env=True)
    p_len = len(_FakeTok().apply_chat_template(PROMPT)["input_ids"])
    for ids, targets in ((c_ids, c_t), (r_ids, r_t)):
        first = next(j for j, t in enumerate(targets) if t != IGN)
        assert first >= p_len - 1
        assert targets[-1] == IGN, "the final position has no next token"


def test_targets_are_pre_aligned_for_the_fused_dpo_kernel():
    tok = _FakeTok()
    c_ids, c_t, _, _ = LP.tokenize_pair(tok, PROMPT, PROMPT + CHOSEN, PROMPT + REJECTED)
    for j, t in enumerate(c_t):
        if t != IGN:
            assert t == c_ids[j + 1]


def test_tools_are_rendered_into_both_sides():
    """A preference over tool CALLS against a prompt that never declared the tools
    trains against a prompt the model is never served."""
    tok = _FakeTok()
    tools = [{"type": "function", "function": {"name": "kb_search", "parameters": {}}}]
    LP.tokenize_pair(tok, PROMPT, PROMPT + CHOSEN, PROMPT + REJECTED, tools=tools)
    assert tok.seen_tools and all(t == tools for t in tok.seen_tools), tok.seen_tools


# --- collate / bin invariants -------------------------------------------------

def test_dpo_bin_layout_is_first_k_chosen_then_k_rejected():
    """triton_dpo.fused_dpo_loss pairs doc k with doc K+k. A bin laid out any other
    way computes a log-ratio between the wrong two documents."""
    pairs = [([1, 2], [2, IGN], [3, 4, 5], [4, 5, IGN]),
             ([6], [IGN], [7, 8], [8, IGN])]
    s = LP.collate_dpo_bin(pairs)
    LP.assert_invariants(s)
    lens = list(s["attention_mask"])
    assert len(lens) == 4 and lens == [2, 1, 3, 2]  # chosen 2,1 then rejected 3,2
    assert sum(lens) == len(s["input_ids"]) == len(s["labels"]) == len(s["position_ids"])
