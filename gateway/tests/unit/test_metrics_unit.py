"""metrics — HTTP instrumentation, job outcomes, per-app exposition filtering,
and the runtime-health additions (loop heartbeats, stats-writer gauges)."""
import time

from prometheus_client import generate_latest

from gateway import metrics


def _registry_text() -> str:
    return generate_latest(metrics._registry).decode()


def test_observe_http_counts_and_times():
    metrics.observe_http("GET", "/{app_id}/v1/chat/completions", 200, "tm-fleet", 0.123)
    text = _registry_text()
    assert 'serverless_http_requests_total{app_id="tm-fleet"' in text
    assert 'route="/{app_id}/v1/chat/completions"' in text
    # Histogram sum/count present for latency math.
    assert "serverless_http_request_duration_seconds_count" in text


def test_observe_job_outcome_skips_pending():
    metrics.observe_job_outcome("appx", "pending")
    assert 'serverless_jobs_total{app_id="appx",status="pending"}' not in _registry_text()
    metrics.observe_job_outcome("appx", "failed")
    assert 'serverless_jobs_total{app_id="appx",status="failed"}' in _registry_text()


def test_render_app_filters_to_one_app():
    metrics.observe_http("GET", "/{app_id}/v1/models", 200, "app-a", 0.01)
    metrics.observe_http("GET", "/{app_id}/v1/models", 200, "app-b", 0.01)
    out = metrics.render_app("app-a").decode()
    assert 'app_id="app-a"' in out
    assert 'app_id="app-b"' not in out


def test_ignore_paths_include_probes():
    # Probes + scrapes must never drown the HTTP metrics.
    for p in ("/health", "/ready", "/metrics", "/"):
        assert p in metrics.IGNORE_PATHS


def test_loop_heartbeat_sets_recent_timestamp():
    before = time.time()
    metrics.loop_heartbeat("unit-test-loop")
    text = _registry_text()
    for line in text.splitlines():
        if line.startswith("gateway_loop_last_tick_timestamp_seconds") and "unit-test-loop" in line:
            ts = float(line.rsplit(" ", 1)[1])
            assert before - 1 <= ts <= time.time() + 1
            break
    else:
        raise AssertionError("heartbeat series not found")


def test_unhandled_exception_counter_labelled_by_route():
    metrics.UNHANDLED_EXCEPTIONS.labels(route="/v1/boom").inc()
    assert 'gateway_unhandled_exceptions_total{route="/v1/boom"}' in _registry_text()


def test_runtime_health_sampler_never_raises_without_db():
    # Neither the DB engine nor the stats writer is initialised in unit tests —
    # the sampler must degrade silently (a scrape can't fail on self-observation).
    metrics._sample_runtime_health()


def test_observe_serverless_stream_partial_inputs():
    # ttft-only and tps-only observations must both be accepted.
    metrics.observe_serverless_stream("app-s", "m1", ttft_s=0.2)
    metrics.observe_serverless_stream("app-s", "m1", tps=42.0)
    text = _registry_text()
    assert "serverless_ttft_seconds_count" in text
    assert "serverless_tokens_per_second_count" in text
