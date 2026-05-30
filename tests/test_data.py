"""Tests for dataset loading and the synthetic-sample generator."""
from conftest import make_cfg
from src.data.datasets import add_text_column, load_community
from src.data.make_sample import sample_community


def test_sample_has_expected_schema():
    data = sample_community(make_cfg(), "sgexams")
    assert {"title", "body", "created_utc", "text"}.issubset(data["posts"].columns)
    assert {"body", "created_utc"}.issubset(data["comments"].columns)


def test_sample_is_deterministic():
    a = sample_community(make_cfg(), "teenagers")["comments"]
    b = sample_community(make_cfg(), "teenagers")["comments"]
    assert a.equals(b)


def test_teenagers_sample_larger_than_sgexams():
    cfg = make_cfg()
    teen = sample_community(cfg, "teenagers")["comments"]
    sg = sample_community(cfg, "sgexams")["comments"]
    assert len(teen) > len(sg)


def test_add_text_column_is_idempotent():
    import pandas as pd

    df = pd.DataFrame({"title": ["a"], "body": ["b"]})
    once = add_text_column(df)
    twice = add_text_column(once)
    assert "text" in twice.columns
    assert twice["text"].iloc[0] == "a.b"


def test_load_community_falls_back_to_sample():
    # Real CSVs are absent and use_sample is true -> loader returns synthetic data.
    data = load_community(make_cfg(), "sgexams")
    assert len(data["comments"]) > 0
    assert len(data["posts"]) > 0
