"""Tests for dictionary-based distortion detection."""
import pandas as pd

from conftest import make_sample
from src.cognitive_distortions.detect import (
    build_distortion_sets,
    count_distortions_in_text,
    detect_dataframe,
    distortion_totals,
)
from src.cognitive_distortions.target_words import category_names


def test_dictionary_has_twelve_categories():
    assert len(category_names()) == 12


def test_count_detects_known_phrase():
    dsets = build_distortion_sets()
    # "will fail" is a Catastrophizing n-gram; "always" is Dichotomous Reasoning.
    counts = count_distortions_in_text("i think it will fail and it always does", dsets)
    assert counts["Catastrophizing"] >= 1
    assert counts["Dichotomous Reasoning"] >= 1


def test_count_returns_all_categories_with_zeros():
    dsets = build_distortion_sets()
    counts = count_distortions_in_text("a perfectly neutral sentence about lunch", dsets)
    assert set(counts.keys()) == set(category_names())


def test_detect_dataframe_shape_and_totals():
    comments = make_sample("sgexams")["comments"]
    det = detect_dataframe(comments, ["body"])
    assert len(det) == len(comments)
    assert list(det.columns) == category_names()
    totals = distortion_totals(det)
    assert totals.sum() > 0  # the synthetic sample embeds real n-grams
