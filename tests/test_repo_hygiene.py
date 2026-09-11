"""Guards against committing a credential.

A real key once reached .env.example -- the tracked template, one letter away
from the ignored .env -- and was caught only by GitHub's push protection.
These run in the normal suite so the mistake is caught locally instead.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# KEY/TOKEN/SECRET/PASSWORD settings must be present but empty in the template.
_CREDENTIAL_LINE = re.compile(
    r"^([A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD))=(.*)$", re.MULTILINE
)


def test_env_example_has_no_assigned_credentials():
    assigned = [
        name
        for name, value in _CREDENTIAL_LINE.findall(
            (REPO_ROOT / ".env.example").read_text()
        )
        if value.strip()
    ]

    assert not assigned, (
        f"{assigned} has a value in .env.example. Real keys belong in .env, "
        f"which is gitignored; the template must ship empty."
    )


def test_env_is_ignored_by_git():
    ignored = subprocess.run(
        ["git", "check-ignore", ".env"], cwd=REPO_ROOT, capture_output=True, text=True
    )

    assert ignored.returncode == 0, ".env is not gitignored"


def test_no_tracked_file_contains_a_google_api_key():
    """Catches a key pasted into any tracked file, not just the template."""
    tracked = subprocess.run(
        ["git", "grep", "-lE", r"AIza[A-Za-z0-9_-]{30,}|AQ\.[A-Za-z0-9_-]{30,}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    if tracked.returncode == 0:
        pytest.fail(f"Possible API key in tracked files:\n{tracked.stdout}")
