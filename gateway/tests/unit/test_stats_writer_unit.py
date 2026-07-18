"""stats_writer — pure pieces: intent coalescing and the queue-depth probe.
(The DB flush path needs Postgres and stays integration territory.)"""
from gateway import stats_writer


def test_merge_last_non_null_wins():
    dst = {"kind": "serverless", "id": "r1", "ttft_ms": 100, "pt": None, "ct": 5}
    stats_writer._merge(dst, {"kind": "serverless", "id": "r1",
                              "ttft_ms": None, "pt": 7, "ct": 9})
    # None never clobbers a value; later non-null values do.
    assert dst["ttft_ms"] == 100
    assert dst["pt"] == 7
    assert dst["ct"] == 9
    # kind/id are identity, never merged payload.
    assert dst["id"] == "r1"


def test_queue_depth_zero_when_not_started():
    assert stats_writer.queue_depth() == 0


def test_enqueue_noop_before_start():
    # Handlers may fire before lifespan starts the writer — must not raise.
    stats_writer.record_stream_completion("req-x", ttft_ms=1, pt=2, ct=3)
