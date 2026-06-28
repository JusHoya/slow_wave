"""Git provenance helpers for the Slow Wave run manifest (Phase 0).

Every run manifest records the git commit hash, dirty/clean working-tree
state, and branch so a result can be traced back to the exact code that
produced it (FR6.1 / ``docs/PHASE0_CONTRACT.md``).

All functions here are defensive: if ``git`` is not installed, the project is
not a git checkout, or a command times out, they return ``None`` (or an
all-``None`` dict) rather than raising. A manifest must always be writable,
even outside a git repository.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# Repo root = the directory two levels up from this file:
#   <repo>/slow_wave/repro/gitinfo.py  -> parents[2] == <repo>
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_git(*args: str) -> str | None:
    """Run ``git <args>`` at the repo root and return stripped stdout.

    Returns ``None`` if git is unavailable, the command fails (non-zero exit),
    or it times out. Never raises.

    Args:
        *args: Arguments to pass to the ``git`` executable.

    Returns:
        The stripped stdout string on success, otherwise ``None``.
    """

    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        # git missing (FileNotFoundError), timeout, or any other subprocess error.
        return None

    if result.returncode != 0:
        return None

    return result.stdout.strip()


def git_commit_hash(short: bool = False) -> str | None:
    """Return the current commit hash, or ``None`` if unavailable.

    Args:
        short: If ``True``, return the abbreviated hash (``git rev-parse
            --short HEAD``); otherwise the full 40-char hash.

    Returns:
        The commit hash string, or ``None`` if git is missing / this is not a
        git repository / the repo has no commits yet.
    """

    args = ["rev-parse"]
    if short:
        args.append("--short")
    args.append("HEAD")
    out = _run_git(*args)
    return out or None


def git_is_dirty() -> bool | None:
    """Return whether the working tree has uncommitted changes.

    Returns:
        ``True`` if ``git status --porcelain`` reports any changes, ``False``
        if the working tree is clean, and ``None`` if git is unavailable / this
        is not a git repository.
    """

    out = _run_git("status", "--porcelain")
    if out is None:
        return None
    return out != ""


def git_branch() -> str | None:
    """Return the current branch name, or ``None`` if unavailable.

    Uses ``git rev-parse --abbrev-ref HEAD``; on a detached HEAD this returns
    the string ``"HEAD"``.

    Returns:
        The branch name, or ``None`` if git is missing / not a repository.
    """

    out = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    return out or None


def git_info() -> dict:
    """Return a dict of git provenance for the manifest.

    Returns:
        A dict with keys ``"commit"`` (str | None), ``"dirty"`` (bool | None),
        and ``"branch"`` (str | None). All values are ``None`` outside a git
        checkout.
    """

    return {
        "commit": git_commit_hash(),
        "dirty": git_is_dirty(),
        "branch": git_branch(),
    }
