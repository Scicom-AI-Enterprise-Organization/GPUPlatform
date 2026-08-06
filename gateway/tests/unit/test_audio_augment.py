"""LiveKit/WebRTC training-audio augmentation (whisper_finetune).

The trainer is a standalone script with no gateway imports, shipped to the box by
SFTP, so it is loaded here by path. Most techniques need only numpy; the ones that
need scipy/soxr/PyAV self-skip, and the real libopus round-trip is exercised by
`test_aug.py` against a speech clip rather than here.

The registry-drift test is the important one: `augment_techniques` is validated
against `training_api._AUG_TECHNIQUES`, so a technique added to the trainer but
not to that set is silently dropped from every run's config.
"""
import ast
import importlib.util
import pathlib
import re

import pytest

np = pytest.importorskip("numpy")

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_TRAINER = _ROOT / "gateway" / "gateway" / "training" / "whisper_finetune.py"
SR = 16000


@pytest.fixture(scope="module")
def wf():
    spec = importlib.util.spec_from_file_location("_wf_aug", _TRAINER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def speech():
    """A crude voiced signal with two real pauses and room tone at both ends."""
    rng = np.random.default_rng(0)
    t = np.arange(int(5.0 * SR)) / SR
    ph = 2 * np.pi * np.cumsum(120 + 25 * np.sin(2 * np.pi * 0.7 * t)) / SR
    x = sum(np.sin(k * ph) / k for k in range(1, 20))
    x *= 0.5 + 0.5 * np.sin(2 * np.pi * 4.5 * t)
    x[int(1.2 * SR):int(1.9 * SR)] = 0.0
    x[int(3.4 * SR):int(4.1 * SR)] = 0.0
    x += rng.normal(0, 1e-4, len(x))
    x[: int(0.4 * SR)] *= 0.02
    x[-int(0.4 * SR):] *= 0.02
    return (x / np.max(np.abs(x)) * 0.35).astype(np.float64)


# --- the drift guard -------------------------------------------------------
def _registry_from_source() -> set[str]:
    tree = ast.parse(_TRAINER.read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "_AUG_FUNCS" for t in n.targets):
            return {k.value for k in n.value.keys}
    raise AssertionError("_AUG_FUNCS not found in whisper_finetune.py")


def test_gateway_validation_set_matches_the_trainer_registry():
    from gateway.training_api import _AUG_TECHNIQUES
    assert _AUG_TECHNIQUES == _registry_from_source()


def test_web_form_offers_every_technique():
    src = (_ROOT / "web" / "src" / "app" / "(app)" / "autotrain" / "new" / "training-form.tsx").read_text()
    ids = set()
    for block in re.findall(
            r"const (?:AUG_OPTIONS|LIVEKIT_AUG_OPTIONS|STREAM_AUG_OPTIONS)[^=]*= \[(.*?)\n\];",
            src, re.S):
        ids |= set(re.findall(r'\{ id: "([a-z0-9_]+)"', block))
    assert ids == _registry_from_source()


def test_livekit_family_is_registered(wf):
    for name in ("livekit", "livekit_sip", "opus", "g711", "packet_loss", "dtx",
                 "agc", "webrtc_ns", "aec", "vad_clip", "resample_chain"):
        assert name in wf._AUG_FUNCS, name


def test_streaming_family_is_registered(wf):
    for name in ("livekit_stream", "hesitation", "vad_pad", "rampup", "denoiser",
                 "room_tone", "jitter"):
        assert name in wf._AUG_FUNCS, name


def test_automation_alias_lists_match_the_registry():
    """`livekit_all` keeps its old 11-technique meaning and `stream_all` carries the
    new ones — one technique is picked per sample, so quietly folding the streaming
    stages into `livekit_all` would halve how often an existing YAML's transport
    stages fire."""
    pytest.importorskip("httpx")     # run_pipeline sys.exits without its deps
    pytest.importorskip("yaml")
    spec = importlib.util.spec_from_file_location(
        "_automation_pipeline", _ROOT / "automation" / "run_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    families = set(mod.ALL_AUGMENTATIONS) | set(mod.LIVEKIT_AUGMENTATIONS) | set(mod.STREAM_AUGMENTATIONS)
    assert families == _registry_from_source()
    assert len(mod.LIVEKIT_AUGMENTATIONS) == 11
    assert set(mod.resolve_augment("voice_all")) == (
        set(mod.LIVEKIT_AUGMENTATIONS) | set(mod.STREAM_AUGMENTATIONS))


# --- numpy-only techniques -------------------------------------------------
def test_agc_gain_is_slew_limited(wf, speech):
    """AGC2 moves at ~3 dB/s, so a quiet clip arrives under-levelled and swells —
    the artefact that makes the first words of an utterance hard to recognise."""
    quiet = speech * 10 ** (-25 / 20)
    y = wf._agc(quiet, SR, initial_gain_db=0.0, rate_db_per_s=3.0)
    head = 20 * np.log10(np.sqrt(np.mean(y[: 2 * SR] ** 2)) + 1e-12)
    tail = 20 * np.log10(np.sqrt(np.mean(y[-2 * SR:] ** 2)) + 1e-12)
    assert tail > head + 3
    assert np.max(np.abs(y)) <= 1.0          # the limiter holds
    assert len(y) == len(quiet)


def test_packet_loss_conceals_rather_than_zeroing(wf, speech):
    """Opus PLC extrapolates the last pitch period; it does not punch holes. The
    older `dropout` technique zeroes chunks, which is a different artefact."""
    np.random.seed(3)
    plc = wf._packet_loss(speech, SR, loss=0.5, mean_burst=2.0, fec_recovery=0.0)
    assert len(plc) == len(speech)
    assert not np.allclose(plc, speech)
    # concealed regions carry signal, so exact zeros stay as rare as in the source
    assert np.mean(np.abs(plc) < 1e-12) <= np.mean(np.abs(speech) < 1e-12) + 0.02


def test_packet_loss_recovers_everything_when_red_always_wins(wf, speech):
    np.random.seed(3)
    assert np.array_equal(wf._packet_loss(speech, SR, loss=0.5, fec_recovery=1.0), speech)


def test_dtx_replaces_pause_content(wf, speech):
    y = wf._dtx(speech, SR, min_gap_ms=200.0)
    assert len(y) == len(speech)
    pause = slice(int(1.35 * SR), int(1.8 * SR))
    assert not np.allclose(y[pause], speech[pause])
    voiced = slice(int(2.3 * SR), int(2.8 * SR))
    assert np.allclose(y[voiced], speech[voiced])      # speech is left alone


def test_vad_clip_trims_silence_but_never_speech(wf, speech):
    """Cropping real speech against an unchanged transcript is how a finetune is
    taught to hallucinate, so this must only ever remove silence."""
    bounds = wf._speech_bounds(speech, SR)
    assert bounds is not None
    y = wf._vad_tighten(speech, SR, max_pad_ms=0.0)
    assert len(y) >= bounds[1] - bounds[0]
    assert len(y) < len(speech)


def test_g711_is_band_limited_to_the_phone_band(wf, speech):
    pytest.importorskip("scipy")
    y = wf._aug_g711(speech, SR)
    assert len(y) == len(speech)
    mag = np.abs(np.fft.rfft(y)) ** 2
    freq = np.fft.rfftfreq(len(y), 1 / SR)
    assert float(np.sum(mag[freq > 4200]) / np.sum(mag)) < 0.01


def test_mulaw_and_alaw_round_trip_preserves_shape_and_range(wf, speech):
    for law in ("u", "a"):
        y = wf._g711(speech, law)
        assert y.shape == speech.shape
        assert np.all(np.isfinite(y))
        assert np.max(np.abs(y)) <= 1.0
        assert float(np.corrcoef(speech, y)[0, 1]) > 0.99   # 8-bit, but not destructive


# --- the streaming regime --------------------------------------------------
def test_hesitation_inserts_quiet_time_and_leaves_speech_intact(wf, speech):
    """The pauses land at the quietest inter-word points (the same rule the
    benchmark's --pause-count uses), so the words themselves survive."""
    y = wf._insert_pauses(speech, SR, count=2, duration=0.7, fill="silence")
    assert len(y) == pytest.approx(len(speech) + 2 * 0.7 * SR, abs=SR // 50)
    # every sample of the original is still present, in order
    assert np.max(np.abs(np.sort(y)[-1000:] - np.sort(speech)[-1000:])) < 1e-9


def test_hesitation_never_pushes_speech_past_whispers_30s_window(wf):
    """Lengthening a clip past 30 s truncates it in the feature extractor while the
    label keeps every word — the hallucination lesson, from the other direction."""
    long = np.tile(np.sin(2 * np.pi * 140 * np.arange(SR) / SR) * 0.3, 29)
    y = wf._insert_pauses(long, SR, count=3, duration=1.2)
    assert len(y) <= 30.0 * SR
    assert len(y) > len(long)                       # it still inserted what fitted
    assert len(wf._insert_pauses(np.tile(long, 2), SR, count=3, duration=1.2)) == 2 * len(long)


def test_vad_pad_frames_the_segment_in_non_speech(wf, speech):
    """A turn silero keeps whole arrives with prefix padding and the min-silence
    hangover around it — the opposite of vad_clip, and the common case."""
    s, e = wf._speech_bounds(speech, SR)
    tight = speech[s:e]
    y = wf._segment_frame(tight, SR, head_s=0.8, tail_s=0.9)
    assert len(y) == pytest.approx(len(tight) + 1.7 * SR, abs=SR // 50)
    head_rms = np.sqrt(np.mean(y[: int(0.5 * SR)] ** 2))
    speech_rms = np.sqrt(np.mean(tight ** 2))
    assert head_rms < speech_rms * 0.05             # room tone, not speech
    assert head_rms > 0                             # but not digital silence either


def test_rampup_attenuates_the_onset_without_deleting_it(wf, speech):
    y = wf._receive_ramp(speech, SR, ramp_s=0.2, depth_db=18.0, hard_onset=True)
    s, _ = wf._speech_bounds(speech, SR)
    assert len(y) <= len(speech) - s + int(0.04 * SR)    # leading silence dropped
    onset = y[: int(0.05 * SR)]
    assert np.max(np.abs(onset)) > 0                     # never a hole
    late = y[int(0.5 * SR):int(0.8 * SR)]
    assert np.sqrt(np.mean(onset ** 2)) < np.sqrt(np.mean(late ** 2))


def test_denoiser_kills_the_noise_floor_but_holds_the_speech_peak(wf, speech):
    """Mirrors what the real GTCRN plugin measured: p25 frame level -28.9 -> -70.9 dB
    with speech peaks within 0.2 dB. That combination is the artefact whisper
    answers with invented content, so the model needs to have seen it."""
    pytest.importorskip("scipy")
    np.random.seed(5)
    noisy = wf._background(speech, SR, 10.0)
    y = wf._deep_denoise(noisy, SR, over_sub=3.5, floor_db=-55.0, gate_dbfs=-70.0)
    assert len(y) == len(noisy)

    def p25_db(sig):
        nf = len(sig) // (SR // 50)
        fr = sig[: nf * (SR // 50)].reshape(nf, SR // 50)
        return 20 * np.log10(np.percentile(np.sqrt(np.mean(fr ** 2, axis=1)), 25) + 1e-12)

    # gate_dbfs is an absolute target, so the floor lands NEAR it rather than
    # stacking with the spectral floor and disappearing 20 dB below anything real
    assert p25_db(noisy) - 25 > p25_db(y) > -85.0
    assert abs(20 * np.log10(np.max(np.abs(y)) / np.max(np.abs(noisy)))) < 0.2


def test_jitter_warping_stays_inside_its_budget(wf, speech):
    for seed in range(5):
        import random as _r
        _r.seed(seed)
        np.random.seed(seed)
        y = wf._neteq_warp(speech, SR, n_ops=4, max_op_ms=60.0)
        assert abs(len(y) - len(speech)) <= 0.02 * len(speech) + 2
        assert np.all(np.isfinite(y))


def test_livekit_stream_chain_is_stable_and_bounded(wf, speech):
    pytest.importorskip("scipy")
    import random as _r
    for seed in range(5):
        _r.seed(seed)
        np.random.seed(seed)
        y = np.asarray(wf._aug_livekit_stream(speech.copy(), SR), dtype=np.float64)
        assert np.all(np.isfinite(y)), seed
        assert y.size > SR, seed
        assert np.max(np.abs(y)) < 2.0, seed
        assert len(y) <= 30.0 * SR, seed


# --- the whole chain -------------------------------------------------------
def test_every_technique_produces_usable_audio(wf, speech):
    pytest.importorskip("scipy")
    import random
    for name, fn in wf._AUG_FUNCS.items():
        if name in ("pitch", "speed"):
            pytest.importorskip("librosa")
        random.seed(1)
        np.random.seed(1)
        y = np.asarray(fn(speech.copy(), SR), dtype=np.float64)
        assert y.size > 0, name
        assert np.all(np.isfinite(y)), name
        assert np.max(np.abs(y)) < 2.0, name
        # duration changes only where a stage inserts or removes time
        if name not in ("speed", "vad_clip", "livekit", "livekit_sip",
                        "livekit_stream", "hesitation", "vad_pad", "rampup"):
            assert 0.98 < len(y) / len(speech) < 1.02, name
        assert len(y) <= 30.0 * SR, name        # whisper's window is never exceeded


def test_livekit_chain_is_stable_across_random_draws(wf, speech):
    pytest.importorskip("scipy")
    import random
    for chain in ("livekit", "livekit_sip"):
        for seed in range(5):
            random.seed(seed)
            np.random.seed(seed)
            y = np.asarray(wf._AUG_FUNCS[chain](speech.copy(), SR), dtype=np.float64)
            assert np.all(np.isfinite(y)), (chain, seed)
            assert y.size > SR, (chain, seed)
            assert np.max(np.abs(y)) < 2.0, (chain, seed)


def test_augment_audio_returns_float32_and_survives_a_broken_technique(wf, speech):
    pytest.importorskip("scipy")
    out = wf._augment_audio(speech, SR, ["livekit"])
    assert out.dtype == np.float32

    # an unknown name is filtered out, not raised
    assert wf._augment_audio(speech, SR, ["nope"]).dtype == np.float32

    # a technique that raises falls back to untouched audio rather than killing the run
    wf._AUG_FUNCS["_boom"] = lambda x, sr: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        assert np.allclose(wf._augment_audio(speech, SR, ["_boom"]), speech.astype(np.float32))
    finally:
        del wf._AUG_FUNCS["_boom"]
