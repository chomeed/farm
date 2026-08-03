from __future__ import annotations

import numpy as np

from lerobot_policy_rewind_state_ours_variant7.common._match_helpers import (
    banded_normalized_dtw,
    centered_window,
    critical_timestep,
    failure_clip_bounds,
    linear_time_penalty,
    sample_random_clip_starts,
    sample_random_variable_clip_bounds,
    smooth,
    strided_indices,
    uniform_subsample_indices,
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


# ── failure_clip_bounds ──────────────────────────────────────────────────────


def test_failure_clip_bounds_midepisode_anchors_at_critical_with_lead_in():
    # 100-frame episode, critical_t=40, 3 lead-in frames, 60 frames post.
    # Should give [37, 101)-clamped-to-[37, 100) -> (37, 100).
    start, end = failure_clip_bounds(
        crit_t=40, ep_len=100, pre_critical_frames=3, max_post_critical_frames=60,
    )
    assert start == 37
    assert end == 100  # crit_t + 1 + 60 = 101, clamped to ep_len=100


def test_failure_clip_bounds_short_post_window_respected():
    # max_post_critical_frames=10 from crit_t=40 -> end at crit_t+1+10 = 51.
    start, end = failure_clip_bounds(
        crit_t=40, ep_len=100, pre_critical_frames=3, max_post_critical_frames=10,
    )
    assert start == 37
    assert end == 51


def test_failure_clip_bounds_clips_lead_in_at_episode_start():
    # crit_t=2 with pre=5 would underflow; start clamps to 0.
    start, end = failure_clip_bounds(
        crit_t=2, ep_len=100, pre_critical_frames=5, max_post_critical_frames=20,
    )
    assert start == 0
    assert end == 23  # crit_t + 1 + 20


def test_failure_clip_bounds_critical_at_episode_end():
    # crit_t at last frame: clip is essentially the lead-in window.
    start, end = failure_clip_bounds(
        crit_t=99, ep_len=100, pre_critical_frames=3, max_post_critical_frames=20,
    )
    assert start == 96
    assert end == 100


def test_failure_clip_bounds_zero_pre_critical():
    start, end = failure_clip_bounds(
        crit_t=40, ep_len=100, pre_critical_frames=0, max_post_critical_frames=20,
    )
    assert start == 40
    assert end == 61


def test_failure_clip_bounds_approaching_is_mirror_of_leaving():
    # Same args, different direction: approaching swaps the roles of pre and
    # max_post so the critical point sits near the END of the clip instead of
    # the START. Concretely, with pre=3, max_post=60, crit_t=80, ep_len=100:
    #   leaving     = [80-3, 80+1+60) clipped to [77, 100)
    #   approaching = [80-60, 80+1+3)              = [20, 84)
    leaving = failure_clip_bounds(
        crit_t=80, ep_len=100, pre_critical_frames=3, max_post_critical_frames=60,
        direction="leaving",
    )
    approaching = failure_clip_bounds(
        crit_t=80, ep_len=100, pre_critical_frames=3, max_post_critical_frames=60,
        direction="approaching",
    )
    assert leaving == (77, 100)
    assert approaching == (20, 84)


def test_failure_clip_bounds_approaching_clips_long_pre_at_zero():
    # Approaching at early critical_t: long pre underflows -> clamp to 0.
    start, end = failure_clip_bounds(
        crit_t=10, ep_len=100, pre_critical_frames=5, max_post_critical_frames=60,
        direction="approaching",
    )
    assert start == 0       # max(0, 10 - 60)
    assert end == 16        # 10 + 1 + 5


def test_failure_clip_bounds_invalid_direction_raises():
    import pytest
    with pytest.raises(ValueError, match="direction"):
        failure_clip_bounds(
            crit_t=40, ep_len=100, pre_critical_frames=3,
            max_post_critical_frames=60, direction="sideways",
        )


def test_failure_clip_bounds_default_direction_is_leaving():
    # Backward compatibility: the original (4-arg) call still produces the
    # 'leaving' shape -- no callers need to change to keep old behavior.
    explicit_leaving = failure_clip_bounds(
        crit_t=40, ep_len=100, pre_critical_frames=3, max_post_critical_frames=20,
        direction="leaving",
    )
    default = failure_clip_bounds(
        crit_t=40, ep_len=100, pre_critical_frames=3, max_post_critical_frames=20,
    )
    assert default == explicit_leaving


# ── sample_random_variable_clip_bounds ───────────────────────────────────────


def test_sample_random_variable_clip_bounds_returns_n_sorted_clips():
    rng = np.random.default_rng(0)
    out = sample_random_variable_clip_bounds(
        ep_len=200, min_len=20, max_len=60, n=5, rng=rng,
    )
    assert len(out) == 5
    starts = [s for s, _ in out]
    assert starts == sorted(starts)
    for s, e in out:
        assert 0 <= s < e <= 200
        assert 20 <= (e - s) <= 60


def test_sample_random_variable_clip_bounds_too_short_returns_empty():
    rng = np.random.default_rng(0)
    out = sample_random_variable_clip_bounds(
        ep_len=10, min_len=20, max_len=60, n=5, rng=rng,
    )
    assert out == []


def test_sample_random_variable_clip_bounds_ep_just_long_enough():
    # ep_len == min_len: only valid clip is (0, min_len).
    rng = np.random.default_rng(0)
    out = sample_random_variable_clip_bounds(
        ep_len=20, min_len=20, max_len=60, n=5, rng=rng,
    )
    for s, e in out:
        assert s == 0
        assert e == 20


# ── uniform_subsample_indices ────────────────────────────────────────────────


def test_uniform_subsample_indices_basic():
    # Clip [10, 26) has 16 frames -> target 16 is identity.
    idx = uniform_subsample_indices(start=10, end=26, target_len=16)
    assert idx == list(range(10, 26))


def test_uniform_subsample_indices_downsample_includes_endpoints():
    # 100-frame raw clip [0, 100), target 16. First idx is 0, last is 99.
    idx = uniform_subsample_indices(start=0, end=100, target_len=16)
    assert len(idx) == 16
    assert idx[0] == 0
    assert idx[-1] == 99
    # Strictly non-decreasing.
    assert all(idx[i] <= idx[i + 1] for i in range(len(idx) - 1))


def test_uniform_subsample_indices_clip_shorter_than_target_repeats():
    # 8-frame raw clip, target 16: indices must stay inside [start, end) and
    # have length 16 (repeats allowed for short clips).
    idx = uniform_subsample_indices(start=5, end=13, target_len=16)
    assert len(idx) == 16
    for i in idx:
        assert 5 <= i < 13
    assert idx[0] == 5
    assert idx[-1] == 12


# ── banded_normalized_dtw ────────────────────────────────────────────────────


def test_banded_normalized_dtw_identical_sequences_is_zero():
    rng = np.random.default_rng(0)
    seq = rng.standard_normal((16, 4)).astype(np.float64)
    assert banded_normalized_dtw(seq, seq, band=3) == 0.0


def test_banded_normalized_dtw_offset_shorter_than_band_finite():
    # Same sequence with a tiny per-frame perturbation; small offset stays
    # well inside band=3, so cost is finite (not inf).
    rng = np.random.default_rng(0)
    a = rng.standard_normal((16, 4)).astype(np.float64)
    b = a + 0.01
    cost = banded_normalized_dtw(a, b, band=3)
    assert np.isfinite(cost)
    assert cost > 0.0


def test_banded_normalized_dtw_is_path_length_normalized():
    # Path-length normalized DTW for two identical-length sequences with
    # one constant additive offset per frame should equal that offset
    # (because the diagonal path has length == T and each step costs
    # exactly the offset).
    T, D = 16, 3
    a = np.zeros((T, D), dtype=np.float64)
    b = np.ones((T, D), dtype=np.float64) * 0.5  # per-frame distance = 0.5*sqrt(D)
    expected = 0.5 * np.sqrt(D)
    cost = banded_normalized_dtw(a, b, band=3)
    assert abs(cost - expected) < 1e-6


def test_banded_normalized_dtw_band_prevents_pathological_collapse():
    # Construct a "stationary" failure clip (constant) and a success clip
    # that moves -- unconstrained DTW would let the stationary clip warp
    # onto a single low-cost success frame. The band caps how far off the
    # diagonal we can stray, so the cost is bounded below.
    T = 16
    a = np.zeros((T, 1), dtype=np.float64)
    b = np.linspace(-10.0, 10.0, T).reshape(T, 1)
    # With band=0 (diagonal only) cost equals mean |b[i] - 0| = mean |b[i]|.
    diag_cost = banded_normalized_dtw(a, b, band=0)
    expected_diag = float(np.mean(np.abs(b[:, 0])))
    assert abs(diag_cost - expected_diag) < 1e-6
    # With a wider band the optimal cost can only go down (more warping
    # freedom), but never become negative.
    wider = banded_normalized_dtw(a, b, band=5)
    assert wider <= diag_cost + 1e-9
    assert wider >= 0.0


def test_banded_normalized_dtw_supports_different_lengths():
    # 16 vs 12: should still produce a finite path-length normalized score.
    rng = np.random.default_rng(0)
    a = rng.standard_normal((16, 2)).astype(np.float64)
    b = rng.standard_normal((12, 2)).astype(np.float64)
    cost = banded_normalized_dtw(a, b, band=5)
    assert np.isfinite(cost)
    assert cost >= 0.0


# ── linear_time_penalty ──────────────────────────────────────────────────────


def test_linear_time_penalty_endpoints():
    w = linear_time_penalty(T=16, start=1.0, end=0.0)
    assert w.shape == (16,)
    assert w[0] == 1.0
    assert w[-1] == 0.0
    # Monotone non-increasing.
    assert np.all(np.diff(w) <= 0.0)


def test_linear_time_penalty_constant_zero_is_zeros():
    w = linear_time_penalty(T=8, start=0.0, end=0.0)
    np.testing.assert_array_equal(w, np.zeros(8, dtype=np.float64))


def test_linear_time_penalty_single_frame_is_start_value():
    # T=1 degenerate: only one position; use the start value.
    w = linear_time_penalty(T=1, start=0.7, end=0.0)
    assert w.shape == (1,)
    assert w[0] == 0.7


# ── time-weighted banded DTW ────────────────────────────────────────────────


def test_banded_dtw_time_penalty_all_zero_matches_no_penalty():
    # All-zero penalty schedule must be equivalent to no penalty at all.
    rng = np.random.default_rng(0)
    a = rng.standard_normal((16, 3)).astype(np.float64)
    b = rng.standard_normal((16, 3)).astype(np.float64)
    base = banded_normalized_dtw(a, b, band=3)
    weighted = banded_normalized_dtw(
        a, b, band=3, time_penalty_weights=np.zeros(16),
    )
    assert abs(base - weighted) < 1e-12


def test_banded_dtw_time_penalty_punishes_early_warps():
    # Construct (a, b) where the optimal unpenalized DTW path warps early
    # to align two offset spikes. With a heavy early-warp penalty, the
    # same warp becomes expensive enough that the diagonal (zero-warp)
    # path wins, raising the returned cost.
    T, D = 8, 1
    a = np.zeros((T, D), dtype=np.float64)
    b = np.zeros((T, D), dtype=np.float64)
    a[4, 0] = 1.0  # spike on a at frame 4
    b[2, 0] = 1.0  # spike on b at frame 2 -> needs warping to align
    base = banded_normalized_dtw(a, b, band=3)
    heavy_early = linear_time_penalty(T=T, start=10.0, end=0.0)
    penalized = banded_normalized_dtw(
        a, b, band=3, time_penalty_weights=heavy_early,
    )
    assert penalized > base


def test_banded_dtw_time_penalty_indifferent_when_no_warp_possible():
    # Two identical sequences: optimal path is the diagonal, no warping
    # happens regardless of penalty. Time penalty contributes 0 per cell
    # (|i - j| = 0 on the diagonal), so cost stays 0.
    seq = np.ones((10, 2), dtype=np.float64)
    weighted = banded_normalized_dtw(
        seq, seq, band=3,
        time_penalty_weights=linear_time_penalty(10, 5.0, 0.0),
    )
    assert weighted == 0.0


def test_banded_dtw_time_penalty_wrong_length_raises():
    a = np.zeros((10, 2), dtype=np.float64)
    b = np.zeros((10, 2), dtype=np.float64)
    import pytest as _pytest
    with _pytest.raises(ValueError, match="time_penalty_weights"):
        banded_normalized_dtw(
            a, b, band=3, time_penalty_weights=np.zeros(5),
        )
