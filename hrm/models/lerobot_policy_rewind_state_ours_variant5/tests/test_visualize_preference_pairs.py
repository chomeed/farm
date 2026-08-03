from __future__ import annotations

import sys

import pytest
import torch


def test_module_imports_and_has_main():
    from lerobot_policy_rewind_state_ours_variant5.common import (
        visualize_preference_pairs as m,
    )
    assert hasattr(m, "main")
    assert hasattr(m, "parse_args")


def test_argparser_defaults(monkeypatch):
    from lerobot_policy_rewind_state_ours_variant5.common import (
        visualize_preference_pairs as m,
    )
    monkeypatch.setattr(
        sys, "argv",
        ["viz", "--output_dir", "/tmp/viz_out"],
    )
    args = m.parse_args()
    assert args.output_dir == "/tmp/viz_out"
    assert args.manifest == (
        "/scratch2/chomeed/manifests/rewind_state_ours_variant5/"
        "preference_pairs.json"
    )
    assert args.n_pairs == 8
    assert args.seed == 0
    assert args.fps == 10


def test_render_row_writes_nonempty_mp4(tmp_path):
    from lerobot_policy_rewind_state_ours_variant5.common import (
        visualize_preference_pairs as m,
    )
    T, C, H, W = 4, 3, 32, 32
    # Two panels, each with both camera keys.
    panels = []
    for _ in range(2):
        panel = {
            cam: torch.rand(T, C, H, W) for cam in m.CAMERA_KEYS
        }
        panels.append(panel)
    labels = ["FAIL ep=0", "GOOD#1 ep=1"]
    out = tmp_path / "row.mp4"
    m._render_row(panels, labels, out, fps=10)
    assert out.exists()
    assert out.stat().st_size > 0
    # The MP4 should decode to T frames.
    import imageio.v2 as imageio
    reader = imageio.get_reader(str(out))
    n = reader.count_frames()
    reader.close()
    assert n == T


def test_render_row_label_count_mismatch_raises(tmp_path):
    from lerobot_policy_rewind_state_ours_variant5.common import (
        visualize_preference_pairs as m,
    )
    T, C, H, W = 2, 3, 16, 16
    panels = [{cam: torch.rand(T, C, H, W) for cam in m.CAMERA_KEYS}]
    with pytest.raises(ValueError, match="labels"):
        m._render_row(panels, ["a", "b"], tmp_path / "x.mp4", fps=10)


def test_sample_pairs_count_and_clamp():
    from lerobot_policy_rewind_state_ours_variant5.common import (
        visualize_preference_pairs as m,
    )
    pairs = [{"failure": {"episode_index": i}} for i in range(10)]
    got = m._sample_pairs(pairs, n_pairs=4, seed=0)
    assert len(got) == 4
    # Clamp: asking for more than available returns all.
    got_all = m._sample_pairs(pairs, n_pairs=99, seed=0)
    assert len(got_all) == 10


def test_sample_pairs_seed_deterministic():
    from lerobot_policy_rewind_state_ours_variant5.common import (
        visualize_preference_pairs as m,
    )
    pairs = [{"failure": {"episode_index": i}} for i in range(20)]
    a = m._sample_pairs(pairs, n_pairs=5, seed=42)
    b = m._sample_pairs(pairs, n_pairs=5, seed=42)
    assert [p["failure"]["episode_index"] for p in a] == \
           [p["failure"]["episode_index"] for p in b]


def test_sample_pairs_empty_raises():
    from lerobot_policy_rewind_state_ours_variant5.common import (
        visualize_preference_pairs as m,
    )
    with pytest.raises(ValueError, match="empty"):
        m._sample_pairs([], n_pairs=3, seed=0)
