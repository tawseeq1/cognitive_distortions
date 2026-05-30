"""Pytest configuration shared by all tests.

WHY THIS FILE EXISTS
--------------------
Ensures the repository root is importable (so ``from src...`` works) regardless of
where pytest is launched from, and exposes a couple of lightweight helpers used
across tests. Tests deliberately avoid heavyweight fixtures so they can also be
executed by the minimal runner in ``scripts``/the validation report when pytest
is unavailable.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.data.make_sample import sample_community  # noqa: E402


def make_cfg():
    """Return the merged project config (sample mode is the default)."""
    return load_config()


def make_sample(community: str = "sgexams"):
    """Return a small synthetic ``{"posts", "comments"}`` dataset for tests."""
    return sample_community(make_cfg(), community)
