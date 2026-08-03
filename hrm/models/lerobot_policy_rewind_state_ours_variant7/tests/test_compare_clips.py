from __future__ import annotations

import numpy as np

from lerobot_policy_rewind_state_ours_variant7.common.compare_rewind_vs_variant7_clips import (
    _clip_scalars,
    aggregate_stats,
)


def test_clip_scalars_mean_and_final():
    curve = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    s = _clip_scalars(curve)
    assert s["clip_mean"] == 0.5
    assert s["final"] == 1.0


def _rec(clip_class, base_mean, base_final, v3_mean, v3_final):
    return {
        "clip_class": clip_class,
        "base": {"clip_mean": base_mean, "final": base_final},
        "variant7": {"clip_mean": v3_mean, "final": v3_final},
    }


def test_separation_sign_and_value():
    # success scalars deliberately above failure scalars.
    records = [
        _rec("success", 0.7, 0.8, 0.7, 0.9),
        _rec("success", 0.9, 0.9, 0.9, 0.95),
        _rec("failure", 0.3, 0.2, 0.1, 0.05),
        _rec("failure", 0.4, 0.3, 0.2, 0.15),
    ]
    stats = aggregate_stats(records)
    # base: success final mean 0.85, failure final mean 0.25 -> sep 0.60
    assert abs(stats["base"]["separation_final"] - 0.6) < 1e-9
    # base: success clip_mean 0.80, failure clip_mean 0.35 -> sep 0.45
    assert abs(stats["base"]["separation_clip_mean"] - 0.45) < 1e-9
    assert stats["base"]["separation_final"] > 0
    assert stats["variant7"]["separation_final"] > 0


def test_per_class_mean_std():
    records = [
        _rec("success", 0.6, 0.8, 0.0, 0.0),
        _rec("success", 1.0, 1.0, 0.0, 0.0),
        _rec("failure", 0.2, 0.1, 0.0, 0.0),
        _rec("failure", 0.2, 0.3, 0.0, 0.0),
    ]
    stats = aggregate_stats(records)
    base_succ = stats["base"]["success"]
    assert base_succ["clip_mean"]["mean"] == 0.8
    assert abs(base_succ["clip_mean"]["std"] - 0.2) < 1e-9  # std of [0.6, 1.0]
    assert base_succ["final"]["mean"] == 0.9
    base_fail = stats["base"]["failure"]
    assert abs(base_fail["final"]["mean"] - 0.2) < 1e-9
    assert abs(base_fail["final"]["std"] - 0.1) < 1e-9
    assert abs(base_fail["clip_mean"]["std"] - 0.0) < 1e-9


def test_models_are_independent():
    # base and variant7 carry different scalars; each block reflects only its own.
    records = [
        _rec("success", 0.5, 0.5, 0.9, 0.9),
        _rec("failure", 0.5, 0.5, 0.1, 0.1),
    ]
    stats = aggregate_stats(records)
    assert stats["base"]["separation_final"] == 0.0
    assert abs(stats["variant7"]["separation_final"] - 0.8) < 1e-9


def test_clip_scalars_rejects_empty_curve():
    import pytest
    with pytest.raises(ValueError):
        _clip_scalars(np.array([], dtype=np.float64))
