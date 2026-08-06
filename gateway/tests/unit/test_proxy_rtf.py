"""proxy_api — audio duration measurement + real-time factor (RTF).

The audio proxies report RTF (processing wall seconds per audio second) on the response
(`X-RTF` / `X-RTFX` / `X-Audio-Seconds`) and to Prometheus. Duration is measured from the
container HEADER only — no decoder on the request path — so the interesting cases are the
formats we can measure (WAV at any rate/width/channels, headerless engine PCM) versus the
ones we must decline to guess at (mp3/opus/flac).
"""
from prometheus_client import generate_latest

from gateway import metrics, proxy_api as p


def _pcm(seconds: float, sr: int = 24000, ch: int = 1, width: int = 2) -> bytes:
    return b"\x00\x01" * int(seconds * sr * ch * width / 2)


def test_wav_duration_reads_the_fmt_chunk():
    # Any rate / channel count / sample width, not just the engine's 24k mono.
    assert p._audio_duration_s(p._pcm_to_wav(_pcm(3.0), 24000, 1, 16), "audio/wav") == 3.0
    assert p._audio_duration_s(p._pcm_to_wav(_pcm(2.0, 16000), 16000, 1, 16), "audio/wav") == 2.0
    stereo = p._pcm_to_wav(_pcm(1.5, 48000, 2), 48000, 2, 16)
    assert p._audio_duration_s(stereo, "audio/wave") == 1.5


def test_wav_duration_ignores_the_engines_bogus_data_size():
    # The TTS engine streams WAV declaring data size 0x7FFFFFFF; _parse_wav takes the
    # chunk to EOF, so the duration comes from the bytes actually present.
    wav = bytearray(p._pcm_to_wav(_pcm(1.0), 24000, 1, 16))
    wav[4:8] = (0x7FFFFFFF).to_bytes(4, "little")     # RIFF size
    wav[40:44] = (0x7FFFFFFF).to_bytes(4, "little")   # data size
    assert p._audio_duration_s(bytes(wav), "audio/wav") == 1.0


def test_headerless_pcm_measured_only_when_declared():
    raw = _pcm(0.5)
    assert p._audio_duration_s(raw, "audio/pcm") == 0.5
    assert p._audio_duration_s(raw, "application/octet-stream") == 0.5
    assert p._audio_duration_s(raw, "audio/pcm; rate=24000") == 0.5
    # A compressed body of the same length is NOT 0.5s of audio — decline, don't guess.
    assert p._audio_duration_s(raw, "audio/mpeg") is None
    assert p._audio_duration_s(raw, "audio/ogg") is None
    # …nor when only the filename says compressed (an upload with a vague content-type).
    assert p._audio_duration_s(raw, "application/octet-stream", "clip.m4a") is None
    assert p._audio_duration_s(b"", "audio/pcm") is None


def test_compressed_bytes_beat_a_vague_content_type():
    # An upload with content-type application/octet-stream and a bare filename — the
    # magic bytes are the only thing left to say "this is not raw PCM".
    tail = _pcm(1.0)
    for magic in (b"ID3\x04\x00\x00\x00\x00\x00\x00\x00\x00", b"\xff\xfb\x90\x00" + b"\x00" * 8,
                  b"OggS\x00\x02\x00\x00\x00\x00\x00\x00", b"fLaC\x00\x00\x00\x22\x00\x00\x00",
                  b"\x00\x00\x00\x20ftypM4A \x00\x00\x00\x00", b"\x1a\x45\xdf\xa3\x01\x00\x00\x00\x00\x00\x00\x1f"):
        assert p._audio_duration_s(magic + tail, "application/octet-stream", "audio") is None, magic[:4]
    # …while genuine PCM (which opens near silence) still measures.
    assert p._audio_duration_s(b"\x00\x00" + tail, "application/octet-stream", "audio") is not None


def test_stream_byte_count_duration_discounts_the_wav_header():
    per_s = p._TTS_PCM_SR * 2
    assert p._pcm_bytes_duration_s(per_s, "audio/pcm") == 1.0
    assert p._pcm_bytes_duration_s(per_s + 44, "audio/wav") == 1.0   # 44-byte header
    assert p._pcm_bytes_duration_s(0, "audio/pcm") is None
    assert p._pcm_bytes_duration_s(20, "audio/wav") is None          # header only, no audio
    assert p._pcm_bytes_duration_s(per_s, "audio/mpeg") is None


def test_body_duration_prefers_a_usable_upstream_value():
    assert p._body_duration_s({"duration": 12.48, "text": "hi"}) == 12.48
    assert p._body_duration_s({"duration": "3.5"}) == 3.5   # whisper has emitted strings
    assert p._body_duration_s({"duration": 0}) is None
    assert p._body_duration_s({"duration": "nope"}) is None
    assert p._body_duration_s({"text": "hi"}) is None
    assert p._body_duration_s("not a body") is None


def test_record_rtf_fields_and_headers():
    res = p._record_rtf("proxy-rtf-a", "whisper", "stt", 10.0, 0.5)
    assert res["audio_seconds"] == 10.0
    assert res["rtf"] == 0.05           # 0.5s of work per 10s of audio
    assert res["rtfx"] == 20.0          # …i.e. 20× real time
    assert p._rtf_headers(res) == {"X-Audio-Seconds": "10.000", "X-RTF": "0.0500", "X-RTFX": "20.00"}


def test_unmeasurable_duration_reports_no_rtf_at_all():
    # No duration (compressed audio) or no elapsed time → no fields, so no headers and
    # no metric — a missing RTF is always better than a fabricated one.
    assert p._record_rtf("proxy-rtf-b", "tts", "tts", None, 0.4) == {}
    assert p._record_rtf("proxy-rtf-b", "tts", "tts", 0.0, 0.4) == {}
    assert p._record_rtf("proxy-rtf-b", "tts", "tts", 3.0, 0.0) == {}
    assert p._rtf_headers({}) == {}
    assert p._rtf_headers({"status_code": 200}) == {}   # a non-audio forward result
    assert 'proxy="proxy-rtf-b"' not in generate_latest(metrics._registry).decode()


def test_metric_carries_kind_and_aggregate_counters():
    metrics.observe_audio_rtf("proxy-rtf-c", "whisper", "stt", 20.0, 1.0)
    metrics.observe_audio_rtf("proxy-rtf-c", "tts-model", "tts", 4.0, 2.0)
    out = metrics.render_proxy("proxy-rtf-c").decode()
    # The aggregate RTF is process_seconds/audio_seconds, so both totals must be exported.
    assert 'proxy_audio_seconds_total{kind="stt",model="whisper",proxy="proxy-rtf-c"} 20.0' in out
    assert 'proxy_audio_process_seconds_total{kind="stt",model="whisper",proxy="proxy-rtf-c"} 1.0' in out
    assert 'kind="tts"' in out and 'proxy_audio_rtf_count' in out
    # render_proxy stays scoped to one endpoint.
    assert "proxy-rtf-a" not in out


def test_zero_or_negative_inputs_never_reach_the_histogram():
    metrics.observe_audio_rtf("proxy-rtf-d", "m", "stt", 0.0, 1.0)
    metrics.observe_audio_rtf("proxy-rtf-d", "m", "stt", 5.0, 0.0)
    metrics.observe_audio_rtf("proxy-rtf-d", "m", "stt", -5.0, 1.0)
    assert metrics.render_proxy("proxy-rtf-d").decode() == ""
