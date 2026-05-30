"""Tests for configuration loading and path resolution."""
from pathlib import Path

from conftest import make_cfg
from src.utils.config import Config


def test_config_loads_expected_keys():
    cfg = make_cfg()
    assert cfg.get("temporal.covid_start") == 1579478400
    assert cfg.get("temporal.seconds_per_week") == 604800
    assert cfg.get("data.use_sample") is True
    assert "sgexams" in cfg.get("experiment.communities")


def test_dotted_get_with_default():
    cfg = make_cfg()
    assert cfg.get("nope.not.here", "fallback") == "fallback"


def test_relative_paths_resolve_under_repo_root():
    cfg = make_cfg()
    figpath = cfg.path("paths.results_figures")
    assert figpath.is_absolute()
    assert figpath.name == "figures"


def test_resolve_path_absolute_passthrough():
    assert Config.resolve_path("/tmp/x") == Path("/tmp/x")
