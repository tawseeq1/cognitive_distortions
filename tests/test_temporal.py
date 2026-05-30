"""Tests for time-series construction and temporal-analysis operations."""
import numpy as np

from conftest import make_cfg, make_sample
from src.cognitive_distortions.timeseries import weekly_distortion_counts
from src.temporal_analysis.timeseries_ops import (
    VOLUME_COL,
    covid_period,
    filter_and_identify_spikes,
    moving_average,
    normalize_by_volume,
    split_by_covid,
    zscore,
    zscore_frame,
)


def test_weekly_counts_has_volume_column():
    ts = weekly_distortion_counts(make_sample("sgexams")["comments"], ["body"])
    assert VOLUME_COL in ts.columns
    assert ts[VOLUME_COL].sum() > 0


def test_zscore_is_zero_mean_unit_sd():
    z = zscore([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(float(np.mean(z))) < 1e-9
    assert abs(float(np.std(z)) - 1.0) < 1e-9


def test_zscore_handles_constant_series():
    assert np.allclose(zscore([7, 7, 7]), 0.0)


def test_zscore_frame_columns_centered():
    ts = weekly_distortion_counts(make_sample("sgexams")["comments"], ["body"])
    z = zscore_frame(normalize_by_volume(ts, scale=100))
    assert abs(float(z.to_numpy().mean())) < 1e-6


def test_moving_average_length():
    assert moving_average([1, 2, 3, 4, 5], 2) == [1.5, 2.5, 3.5, 4.5]


def test_spike_detection_flags_a_jump():
    data = [1, 1, 1, 1, 10, 1, 1, 1]
    _, spikes = filter_and_identify_spikes(data, window_size=4)
    assert 4 in spikes


def test_covid_period_classification():
    cfg = make_cfg()
    cs, ce = cfg.get("temporal.covid_start"), cfg.get("temporal.covid_end")
    assert covid_period(cs - 1, cs, ce) == "before"
    assert covid_period(cs + 1, cs, ce) == "during"
    assert covid_period(ce + 1, cs, ce) == "after"


def test_split_by_covid_partitions_rows():
    cfg = make_cfg()
    comments = make_sample("sgexams")["comments"]
    parts = split_by_covid(comments, cfg.get("temporal.covid_start"),
                           cfg.get("temporal.covid_end"), cfg.get("temporal.end_date"))
    total = sum(len(v) for v in parts.values())
    assert total <= len(comments)  # end_date may exclude a tail
    assert set(parts.keys()) == {"before", "during", "after"}
