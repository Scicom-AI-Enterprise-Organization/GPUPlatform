"""`train_splits` — subset ablation over one merged ASR corpus.

A merged audio dataset labels every source by origin (`<config>/<split>` for a HF
subset, a bare `train` for a Label project), so training on some of them used to mean
re-merging and re-materialising the clips. These pin the matching rules that make the
config-flag version safe to trust: it only ever narrows the TRAINING half, and it says
so loudly when it selected less than was asked for.
"""
import pytest

from gateway.training.whisper_finetune import filter_train_splits

ROWS = [
    {"split": "train", "text": "callcentre-1"},
    {"split": "train", "text": "callcentre-2"},
    {"split": "dialogue_calls/train", "text": "dc"},
    {"split": "synthetic_podcast/train", "text": "sp"},
    {"split": "hard_entities_v2_texts/train", "text": "hev2"},
]


def labels(rows):
    return [r["split"] for r in rows]


def test_absent_or_empty_is_a_no_op():
    """The default must be byte-identical to the pre-existing behaviour."""
    assert filter_train_splits(ROWS, {}) is ROWS
    assert filter_train_splits(ROWS, {"train_splits": []}) is ROWS
    assert filter_train_splits(ROWS, {"train_splits": ["  "]}) is ROWS


def test_bare_train_does_not_select_every_subsets_train_half():
    """The whole point of the ablation: `train` is the call-centre rows ALONE.

    Matching on the split suffix instead would quietly select all five rows and every
    ablation would train on the identical corpus while reporting different names.
    """
    assert labels(filter_train_splits(ROWS, {"train_splits": ["train"]})) == ["train", "train"]


def test_config_prefix_and_full_label_both_match():
    by_prefix = filter_train_splits(ROWS, {"train_splits": ["synthetic_podcast"]})
    by_label = filter_train_splits(ROWS, {"train_splits": ["synthetic_podcast/train"]})
    assert labels(by_prefix) == labels(by_label) == ["synthetic_podcast/train"]


def test_combination_preserves_row_order():
    kept = filter_train_splits(ROWS, {"train_splits": ["train", "hard_entities_v2_texts"]})
    assert labels(kept) == ["train", "train", "hard_entities_v2_texts/train"]


def test_matching_nothing_raises_rather_than_training_on_zero_rows():
    with pytest.raises(RuntimeError) as e:
        filter_train_splits(ROWS, {"train_splits": ["hard_entities_v3"]})
    # The error has to name what WAS available — a bare "no rows" sends the next hour
    # at the dataset instead of at the typo.
    assert "hard_entities_v2_texts/train" in str(e.value)


def test_an_entry_that_matches_nothing_is_reported(capsys):
    """A typo'd subset must not read as 'that subset doesn't help'.

    This is the silently-dropped-augment-name failure in a place where it would corrupt
    a comparison rather than just weaken a run, so the partial match warns and continues.
    """
    kept = filter_train_splits(ROWS, {"train_splits": ["train", "sinthetic_podcast"]})
    assert labels(kept) == ["train", "train"]
    # The trainer's log() writes to stdout — that stream is what the run's log tail shows.
    assert "sinthetic_podcast" in capsys.readouterr().out


def test_unlabelled_rows_count_as_train():
    rows = [{"text": "no split column"}, {"split": "synthetic/train"}]
    assert filter_train_splits(rows, {"train_splits": ["train"]}) == [rows[0]]
