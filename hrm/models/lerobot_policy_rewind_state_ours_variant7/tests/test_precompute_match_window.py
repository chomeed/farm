from __future__ import annotations

import sys

import pytest
import torch


def _tagged_h(B: int, n_streams: int, clip_len: int, D: int) -> torch.Tensor:
    """Stream-major transformer output where token (stream s, frame t) holds
    the constant value s*100 + t across all D channels."""
    h = torch.zeros(B, n_streams * clip_len, D)
    for s in range(n_streams):
        for t in range(clip_len):
            h[:, s * clip_len + t, :] = s * 100 + t
    return h


def test_pool_last_k_frames_picks_last_k_per_stream():
    from lerobot_policy_rewind_state_ours_variant7.common import (
        precompute_preference_pairs as m,
    )
    B, n_streams, clip_len, D, k = 1, 3, 16, 4, 5
    h = _tagged_h(B, n_streams, clip_len, D)
    out = m._pool_last_k_frames(h, n_streams, clip_len, k)

    assert out.shape == (B, n_streams * D)
    # Last 5 frames of a 16-frame clip => t in {11,..,15}, mean = 13.0.
    expected_mean_t = sum(range(clip_len - k, clip_len)) / k
    for s in range(n_streams):
        block = out[0, s * D:(s + 1) * D]
        assert torch.allclose(block, torch.full((D,), s * 100 + expected_mean_t))


def test_pool_last_k_frames_clamps_k_to_clip_len():
    from lerobot_policy_rewind_state_ours_variant7.common import (
        precompute_preference_pairs as m,
    )
    B, n_streams, clip_len, D = 1, 3, 8, 2
    h = _tagged_h(B, n_streams, clip_len, D)
    # k larger than clip_len => use the whole clip.
    out = m._pool_last_k_frames(h, n_streams, clip_len, k=999)
    whole_mean_t = sum(range(clip_len)) / clip_len
    for s in range(n_streams):
        block = out[0, s * D:(s + 1) * D]
        assert torch.allclose(block, torch.full((D,), s * 100 + whole_mean_t))


def test_pool_last_k_frames_rejects_wrong_token_count():
    from lerobot_policy_rewind_state_ours_variant7.common import (
        precompute_preference_pairs as m,
    )
    h = torch.zeros(1, 17, 4)  # 17 is not n_streams*clip_len for any sane pair
    with pytest.raises(ValueError, match="expected n_streams"):
        m._pool_last_k_frames(h, n_streams=3, clip_len=16, k=5)


def test_match_window_argparser_default(monkeypatch):
    from lerobot_policy_rewind_state_ours_variant7.common import (
        precompute_preference_pairs as m,
    )
    monkeypatch.setattr(
        sys, "argv",
        ["precompute", "--base_ckpt", "x", "--success_repo_id", "a",
         "--failure_repo_id", "b", "--output_manifest", "/tmp/o.json"],
    )
    args = m.parse_args()
    assert args.match_window == 5
    assert args.smooth_window == 7  # distinct knob, unchanged
    assert args.match_state_weight == 0.0  # default: no state weighting


def test_weighted_match_embedding_shape():
    from lerobot_policy_rewind_state_ours_variant7.common import (
        precompute_preference_pairs as m,
    )
    B, n_cam, clip_len, D, state_dim, k = 1, 2, 8, 4, 3, 3
    h = torch.randn(B, (n_cam + 1) * clip_len, D)
    state = torch.randn(B, clip_len, state_dim)
    out = m._weighted_match_embedding(h, state, n_cam, clip_len, k,
                                      state_weight=0.4)
    # [state block: k*state_dim] + [n_cam camera blocks: D each]
    assert out.shape == (B, k * state_dim + n_cam * D)


def test_weighted_match_embedding_is_weighted_sum_of_block_cosines():
    """cos(emb_f, emb_s) must equal w_state*cos(state) + w_cam*sum cos(cam)."""
    from lerobot_policy_rewind_state_ours_variant7.common import (
        precompute_preference_pairs as m,
    )
    torch.manual_seed(0)
    n_cam, clip_len, D, state_dim, k, w = 2, 8, 4, 3, 3, 0.4
    cam_w = (1.0 - w) / n_cam

    def blocks(h, state):
        hs = h.view(1, n_cam + 1, clip_len, D)
        cams = [hs[0, c, -k:, :].mean(0) for c in range(n_cam)]
        st = state[0, -k:, :].reshape(-1)
        return st, cams

    def unit(x):
        return x / (x.norm() + 1e-8)

    h_f = torch.randn(1, (n_cam + 1) * clip_len, D)
    h_s = torch.randn(1, (n_cam + 1) * clip_len, D)
    st_f = torch.randn(1, clip_len, state_dim)
    st_s = torch.randn(1, clip_len, state_dim)

    emb_f = m._weighted_match_embedding(h_f, st_f, n_cam, clip_len, k, w)[0]
    emb_s = m._weighted_match_embedding(h_s, st_s, n_cam, clip_len, k, w)[0]
    got = torch.dot(emb_f, emb_s) / (emb_f.norm() * emb_s.norm())

    sf, cf = blocks(h_f, st_f)
    ss, cs = blocks(h_s, st_s)
    expected = w * torch.dot(unit(sf), unit(ss))
    for c in range(n_cam):
        expected = expected + cam_w * torch.dot(unit(cf[c]), unit(cs[c]))

    assert torch.allclose(got, expected, atol=1e-5)


def test_weighted_match_embedding_rejects_bad_weight():
    from lerobot_policy_rewind_state_ours_variant7.common import (
        precompute_preference_pairs as m,
    )
    h = torch.randn(1, 3 * 8, 4)
    state = torch.randn(1, 8, 3)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="state_weight"):
            m._weighted_match_embedding(h, state, 2, 8, 3, state_weight=bad)


import numpy as np


def test_dtw_distance_identical_is_zero():
    from lerobot_policy_rewind_state_ours_variant7.common import (
        precompute_preference_pairs as m,
    )
    seq = np.array([[0.0], [1.0], [2.0], [3.0]])
    assert m._dtw_distance(seq, seq) == 0.0


def test_dtw_distance_ignores_pure_time_stretch():
    """A trajectory and a time-stretched copy of it have DTW distance 0."""
    from lerobot_policy_rewind_state_ours_variant7.common import (
        precompute_preference_pairs as m,
    )
    a = np.array([[0.0], [1.0], [2.0]])
    b = np.array([[0.0], [0.0], [1.0], [2.0], [2.0]])  # same path, slower
    assert m._dtw_distance(a, b) == 0.0


def test_dtw_distance_known_value():
    from lerobot_policy_rewind_state_ours_variant7.common import (
        precompute_preference_pairs as m,
    )
    a = np.array([[0.0], [1.0]])
    b = np.array([[0.0], [2.0]])
    # acc[2,2] = |1-2| + min(path) = 1.0
    assert m._dtw_distance(a, b) == pytest.approx(1.0)


def test_blend_match_scores_w0_is_cosine_ranking():
    from lerobot_policy_rewind_state_ours_variant7.common import (
        precompute_preference_pairs as m,
    )
    cos = np.array([[0.1, 0.9, 0.5]])
    dtw = np.array([[5.0, 1.0, 9.0]])
    score = m._blend_match_scores(cos, dtw, dtw_weight=0.0)
    # w=0 => score is the row min-max of cosine; argmax matches cosine argmax.
    assert int(score.argmax(axis=1)[0]) == int(cos.argmax(axis=1)[0])


def test_blend_match_scores_w1_follows_dtw():
    from lerobot_policy_rewind_state_ours_variant7.common import (
        precompute_preference_pairs as m,
    )
    cos = np.array([[0.1, 0.9, 0.5]])
    dtw = np.array([[5.0, 1.0, 9.0]])  # smallest distance at index 1
    score = m._blend_match_scores(cos, dtw, dtw_weight=1.0)
    # w=1 => best score is the smallest DTW distance.
    assert int(score.argmax(axis=1)[0]) == int(dtw.argmin(axis=1)[0])


def test_blend_match_scores_dtw_can_flip_the_winner():
    from lerobot_policy_rewind_state_ours_variant7.common import (
        precompute_preference_pairs as m,
    )
    # Cosine prefers index 0; DTW strongly prefers index 1.
    cos = np.array([[0.95, 0.90]])
    dtw = np.array([[100.0, 0.0]])
    score = m._blend_match_scores(cos, dtw, dtw_weight=0.6)
    assert int(score.argmax(axis=1)[0]) == 1


def test_match_dtw_weight_argparser_default(monkeypatch):
    from lerobot_policy_rewind_state_ours_variant7.common import (
        precompute_preference_pairs as m,
    )
    monkeypatch.setattr(
        sys, "argv",
        ["precompute", "--base_ckpt", "x", "--success_repo_id", "a",
         "--failure_repo_id", "b", "--output_manifest", "/tmp/o.json"],
    )
    args = m.parse_args()
    assert args.match_dtw_weight == 0.0  # default: pure cosine
