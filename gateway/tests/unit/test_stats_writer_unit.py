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


# ---------- output-throughput sampling (the P10-TPS-alert post-mortem) --------
# Regression pins for the bug that fired ServerlessGPUProxyP10TPSLow against a
# healthy backend: `decode_tps` used to fall back to the FULL end-to-end latency
# whenever TTFT was unknown, so every non-streaming request reported
# tokens/(prefill+decode) as if it were the GPU's decode rate.

def test_decode_tps_streaming_excludes_ttft():
    # 100 tokens, 2.5s total, 0.5s of it prefill → 100/2.0s = 50 tok/s.
    assert stats_writer.decode_tps(100, 2500, 500) == 50.0


def test_decode_tps_none_without_ttft_never_uses_total_latency():
    # THE bug. A 20-token answer behind a 1s prefill in a 1.32s request must not
    # report 20/1.32 = 15 tok/s against a backend decoding at ~62 tok/s.
    assert stats_writer.decode_tps(20, 1320, None) is None


def test_decode_tps_rejects_zero_length_decode_window():
    # ttft ≈ latency = a buffered flush relayed through the streaming path.
    # Unguarded this is 50/0.010s = 5000 tok/s, which lands in the +Inf bucket
    # and poisons the histogram's _sum (and every mean derived from it).
    assert stats_writer.decode_tps(50, 2010, 2000) is None


def test_decode_tps_rejects_too_few_tokens():
    assert stats_writer.decode_tps(3, 2000, 500) is None
    assert stats_writer.decode_tps(None, 2000, 500) is None


def test_e2e_tps_is_defined_where_decode_tps_is_not():
    # The non-streaming request above still gets an end-to-end number — it just
    # lives in its own series, because it answers a different question.
    assert stats_writer.decode_tps(20, 1000, None) is None
    assert stats_writer.e2e_tps(20, 1000) == 20.0
    assert stats_writer.e2e_tps(0, 1000) is None
    assert stats_writer.e2e_tps(20, 0) is None
