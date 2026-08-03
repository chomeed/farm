from __future__ import annotations

import json
from pathlib import Path

import pytest

from lerobot_policy_rewind_state_ours_variant3.common.manifest import (
    PreferenceManifest,
    load_manifest,
    write_manifest,
)


def _example_manifest_dict() -> dict:
    return {
        "base_ckpt": "/abs/path/to/010000/pretrained_model",
        "success_repo_id": "chomeed/robomimic_square_resized_peg_success",
        "failure_repo_id": "chomeed/robomimic_square_resized_peg_failure",
        "clip_len": 16,
        "frame_stride": 1,
        "smooth_window": 7,
        "top_k": 3,
        "pairs": [
            {
                "failure": {"episode_index": 7, "start": 102, "end": 118, "critical_t": 110},
                "success_candidates": [
                    {"episode_index": 3,  "start": 80, "end": 96,  "cosine": 0.842, "reward_delta": 0.21},
                    {"episode_index": 12, "start": 96, "end": 112, "cosine": 0.811, "reward_delta": 0.18},
                ],
            }
        ],
    }


def test_write_then_load_round_trip(tmp_path: Path):
    p = tmp_path / "pairs.json"
    write_manifest(p, _example_manifest_dict())
    m = load_manifest(p)
    assert isinstance(m, PreferenceManifest)
    assert m.base_ckpt.endswith("010000/pretrained_model")
    assert m.clip_len == 16
    assert m.frame_stride == 1
    assert len(m.pairs) == 1
    pair = m.pairs[0]
    assert pair["failure"]["critical_t"] == 110
    assert pair["success_candidates"][0]["cosine"] == pytest.approx(0.842)


def test_load_manifest_rejects_missing_field(tmp_path: Path):
    p = tmp_path / "pairs.json"
    bad = _example_manifest_dict()
    del bad["clip_len"]
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="clip_len"):
        load_manifest(p)


def test_assert_compatible_passes_on_match(tmp_path: Path):
    p = tmp_path / "pairs.json"
    write_manifest(p, _example_manifest_dict())
    m = load_manifest(p)
    # Should not raise.
    m.assert_compatible(
        base_ckpt="/abs/path/to/010000/pretrained_model",
        clip_len=16,
        frame_stride=1,
    )


def test_assert_compatible_raises_on_ckpt_mismatch(tmp_path: Path):
    p = tmp_path / "pairs.json"
    write_manifest(p, _example_manifest_dict())
    m = load_manifest(p)
    with pytest.raises(ValueError, match="base_ckpt"):
        m.assert_compatible(
            base_ckpt="/somewhere/else/pretrained_model",
            clip_len=16,
            frame_stride=1,
        )


def test_assert_compatible_raises_on_clip_len_mismatch(tmp_path: Path):
    p = tmp_path / "pairs.json"
    write_manifest(p, _example_manifest_dict())
    m = load_manifest(p)
    with pytest.raises(ValueError, match="clip_len"):
        m.assert_compatible(
            base_ckpt="/abs/path/to/010000/pretrained_model",
            clip_len=32,
            frame_stride=1,
        )
