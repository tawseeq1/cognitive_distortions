"""Configuration loading and path resolution.

WHY THIS FILE EXISTS
--------------------
Every original notebook hard-coded absolute Google-Drive paths such as
``/content/drive/MyDrive/Cognitive_Distortions_DrFarhanAli_Tawseeq/...`` and
literal COVID timestamps. That made the code impossible to run anywhere else.

This module loads the YAML files in ``configs/`` into a single ``Config`` object,
resolves every path relative to the repository root (so the repo is portable),
and merges an optional ``config.local.yaml`` for machine-specific overrides that
should never be committed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Repository root = two levels up from this file (src/utils/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class Config:
    """In-memory view of the merged YAML configuration.

    Access nested values with dotted keys via :meth:`get`, and resolve any path
    string (absolute or repo-relative) to an absolute :class:`~pathlib.Path` via
    :meth:`path`.
    """

    data: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ access
    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Return a nested value addressed by a dotted key (e.g. ``"topic.k_max"``)."""
        node: Any = self.data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def path(self, dotted_key: str, default: Optional[str] = None) -> Path:
        """Resolve a configured path string to an absolute path under the repo root."""
        raw = self.get(dotted_key, default)
        if raw is None:
            raise KeyError(f"No path configured for '{dotted_key}'")
        return self.resolve_path(str(raw))

    @staticmethod
    def resolve_path(raw: str) -> Path:
        """Resolve ``raw`` to an absolute path; relative paths are anchored at repo root."""
        p = Path(os.path.expanduser(raw))
        return p if p.is_absolute() else (REPO_ROOT / p)


_DEFAULT_FILES = [
    "paths.yaml",
    "data.yaml",
    "temporal.yaml",
    "topic_modeling.yaml",
    "sentiment.yaml",
    "experiment.yaml",
]


def load_config(
    config_dir: Optional[Path] = None,
    extra_files: Optional[list[str]] = None,
) -> Config:
    """Load and merge all YAML config files into a single :class:`Config`.

    Files are merged in order; a ``config.local.yaml`` (git-ignored) is merged
    last so individual machines can override paths without editing tracked files.

    Parameters
    ----------
    config_dir:
        Directory containing the YAML files. Defaults to ``<repo>/configs``.
    extra_files:
        Additional YAML filenames to merge after the defaults.
    """
    config_dir = config_dir or CONFIGS_DIR
    merged: Dict[str, Any] = {}
    files = list(_DEFAULT_FILES) + list(extra_files or []) + ["config.local.yaml"]
    for name in files:
        fpath = config_dir / name
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        merged = _deep_merge(merged, loaded)
    return Config(data=merged)
