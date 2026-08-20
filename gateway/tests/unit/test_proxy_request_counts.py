"""Histogram interval selection for the Queue tab's bar chart.

The DB-backed halves of `/request-counts` are exercised against the live stack; what is
pinned here is `_auto_bucket`, because every way it can be wrong is a silently unusable
chart rather than an error: too fine and the browser renders thousands of 1px bars, too
coarse and a week collapses into one.
"""
import pytest

from gateway.proxy_api import (_AUTO_TARGET, _BUCKETS, _MAX_BUCKETS, _auto_bucket,
                               densify_buckets)

HOUR = 3600
DAY = 24 * HOUR


def test_ladder_is_ordered_finest_first():
    """`_auto_bucket` walks the dict in insertion order and takes the first rung that
    fits, so an out-of-order ladder would return a needlessly fine interval."""
    secs = list(_BUCKETS.values())
    assert secs == sorted(secs)


@pytest.mark.parametrize("span,expect", [
    (10 * 60, "30s"),         # 10 min -> 20 bars
    (30 * 60, "1m"),          # 30 min -> 30 bars
    (6 * HOUR, "10m"),        # 36 bars — matches the live check
    (24 * HOUR, "30m"),       # 48 bars
    (7 * DAY, "6h"),          # 28 bars
    (30 * DAY, "1d"),         # 30 bars
    (365 * DAY, "30d"),       # 12 bars
])
def test_auto_bucket_keeps_bar_count_under_target(span, expect):
    got = _auto_bucket(span)
    assert span / _BUCKETS[got] <= _AUTO_TARGET, f"{got} gives too many bars for {span}s"
    assert got == expect


def test_every_span_stays_under_the_bar_target():
    """Sweep rather than spot-check: the invariant is what matters, not the exact rung."""
    for span in (1, 59, 60, 3599, HOUR, 5 * HOUR, DAY, 3 * DAY, 14 * DAY,
                 90 * DAY, 2 * 365 * DAY):
        got = _auto_bucket(span)
        assert span / _BUCKETS[got] <= _AUTO_TARGET, (span, got)


def test_auto_bucket_picks_the_coarsest_that_fits():
    """Not merely 'a' rung that fits — the coarsest, or charts get needlessly dense."""
    for span in (HOUR, DAY, 7 * DAY, 30 * DAY):
        got = _auto_bucket(span)
        keys = list(_BUCKETS)
        i = keys.index(got)
        if i > 0:                      # the next finer rung must NOT have fitted
            assert span / _BUCKETS[keys[i - 1]] > _AUTO_TARGET


def test_absurd_and_degenerate_spans_do_not_raise():
    """A chart that renders something beats a 500. Zero/negative spans happen when an
    endpoint has one row (min == max) or none at all."""
    assert _auto_bucket(0) == "1m"
    assert _auto_bucket(-5) == "1m"
    assert _auto_bucket(10 ** 12) == next(reversed(_BUCKETS))


# ------------------------------------------------------------------ zero-filling
# `GROUP BY` emits only periods that HAVE rows. Two bars a fortnight apart then render
# side by side, implying consecutive periods — so the gaps must be materialised.

def _bucket(epoch, total=1, status="completed"):
    from datetime import datetime, timezone
    return {"_epoch": float(epoch),
            "ts": datetime.fromtimestamp(epoch, timezone.utc).isoformat(),
            "total": total, "by_status": {status: total}}


def test_gaps_become_zero_bars():
    # rows at t=0 and t=4h only; a 1h axis over 0..4h must yield 5 bars
    got, note = densify_buckets([_bucket(0), _bucket(4 * HOUR)], HOUR, 0, 4 * HOUR)
    assert note is None
    assert [b["total"] for b in got] == [1, 0, 0, 0, 1]
    assert all(b["by_status"] == {} for b in got[1:4])


def test_axis_spans_the_REQUESTED_window_not_just_the_data():
    """The reference behaviour: 'last 30 days' with traffic on one day still draws 30
    days of axis. Bounds come from the caller's window, not from min/max(created_at)."""
    got, _ = densify_buckets([_bucket(10 * DAY)], DAY, 0, 30 * DAY)
    assert len(got) == 31
    assert sum(b["total"] for b in got) == 1
    assert got[10]["total"] == 1


def test_existing_buckets_survive_intact():
    got, _ = densify_buckets([_bucket(0, total=7, status="blocked")], HOUR, 0, 2 * HOUR)
    assert got[0]["total"] == 7 and got[0]["by_status"] == {"blocked": 7}
    # the internal epoch key must not leak into the response
    assert all("_epoch" not in b for b in got)


def test_start_is_floored_to_the_interval():
    """An unaligned window start must snap to a bucket boundary, or every bar is offset
    from the buckets the SQL produced and nothing ever matches."""
    # window starts at 01:15 ; the only row is in the 01:00 bucket
    got, _ = densify_buckets([_bucket(3600)], HOUR, 3600 + 900, 3600 + 900)
    assert len(got) == 1
    assert got[0]["total"] == 1          # the 3600 bucket, not an empty 4500 one


def test_capped_path_still_strips_the_internal_epoch_key():
    """The cap returns the input list — which carries `_epoch`. That is an internal
    join key and must not reach the API response on any path."""
    got, note = densify_buckets([_bucket(0)], 60, 0, 30 * DAY)
    assert note
    assert all("_epoch" not in b for b in got)


def test_too_many_intervals_is_capped_and_reported():
    """30 days at 1m is 43,200 bars — one DOM element each. Refuse and say why."""
    got, note = densify_buckets([_bucket(0)], 60, 0, 30 * DAY)
    assert [b["total"] for b in got] == [1]   # left sparse, not truncated silently
    assert note and str(_MAX_BUCKETS) in note


def test_missing_bounds_or_interval_is_a_noop():
    """No axis to fill against — pass the buckets through untouched (bar the key)."""
    for args in ((HOUR, None, None), (0, 0, HOUR)):
        got, note = densify_buckets([_bucket(0)], *args)
        assert [b["total"] for b in got] == [1] and note is None


def test_empty_range_with_no_rows_still_draws_the_axis():
    """No traffic at all in the window is a legitimate answer — and it should look like
    an empty chart with an axis, not like a missing chart."""
    got, note = densify_buckets([], HOUR, 0, 3 * HOUR)
    assert note is None
    assert [b["total"] for b in got] == [0, 0, 0, 0]
