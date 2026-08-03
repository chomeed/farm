from __future__ import annotations

import numpy as np

from lerobot_policy_rewind_state_ours.common._match_helpers import (
    centered_window,
    critical_timestep,
    sample_random_clip_starts,
    smooth,
    strided_indices,
)


def test_smooth_window_one_is_identity():
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_array_equal(smooth(x, 1), x)


def test_smooth_window_three_is_rolling_mean():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    out = smooth(x, 3)
    # Edge-padded rolling mean preserves length and dampens noise.
    assert out.shape == x.shape
    assert out.dtype == np.float32
    # Middle window: (1+2+3)/3 == 2.0
    assert abs(out[2] - 2.0) < 1e-6


def test_critical_timestep_picks_smoothed_argmax():
    x = np.array([0.0, 0.0, 0.5, 1.0, 0.4, 0.4, 0.4], dtype=np.float32)
    t = critical_timestep(x, smooth_window=1)
    assert t == 3


def test_centered_window_clamps_to_episode():
    # Mid: clean centering.
    assert centered_window(t=50, span=16, ep_len=200) == (42, 58)
    # Near start: shifts forward.
    assert centered_window(t=2, span=16, ep_len=200) == (0, 16)
    # Near end: shifts backward.
    assert centered_window(t=199, span=16, ep_len=200) == (184, 200)
    # Episode shorter than span: clamps to (0, ep_len).
    assert centered_window(t=4, span=16, ep_len=10) == (0, 10)


def test_sample_random_clip_starts_no_replacement_when_room():
    rng = np.random.default_rng(0)
    starts = sample_random_clip_starts(ep_len=100, span=16, n=5, rng=rng)
    assert len(starts) == 5
    assert len(set(starts)) == 5
    assert all(0 <= s <= 100 - 16 for s in starts)
    assert starts == sorted(starts)


def test_sample_random_clip_starts_returns_empty_when_too_short():
    rng = np.random.default_rng(0)
    assert sample_random_clip_starts(ep_len=8, span=16, n=3, rng=rng) == []


def test_strided_indices():
    assert strided_indices(start=10, clip_len=4, frame_stride=2) == [10, 12, 14, 16]
    assert strided_indices(start=0, clip_len=3, frame_stride=1) == [0, 1, 2]
