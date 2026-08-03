from __future__ import annotations

from types import SimpleNamespace

import pytest


def _fake_cfg(repo_id: str, ftune: bool, manifest_path: str) -> SimpleNamespace:
    """Mimic the TrainPipelineConfig surface that the factory reads."""
    dataset = SimpleNamespace(repo_id=repo_id, root=None)
    policy = SimpleNamespace(
        finetune_encoder=ftune,
        encoder_camera_keys=[
            "observation.images.agentview",
            "observation.images.eye_in_hand",
        ],
        preference_pairs_path=manifest_path,
        preference_clip_len=16,
        preference_frame_stride=1,
        preference_topk=3,
        preference_seed=0,
        pretrained_path=None,
    )
    return SimpleNamespace(dataset=dataset, policy=policy)


def test_factory_robomimic_make_dataset_returns_wrapper(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    # Skip if local cache absent: instantiation hits LeRobotDataset.
    from pathlib import Path
    cache = Path("/scratch2/chomeed/hf_cache/lerobot/chomeed/robomimic_square_resized_peg_success")
    if not cache.exists():
        pytest.skip("local lerobot cache absent")

    import json
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({
        "base_ckpt": "/abs/path/to/010000/pretrained_model",
        "success_repo_id": "chomeed/robomimic_square_resized_peg_success",
        "failure_repo_id": "chomeed/robomimic_square_resized_peg_failure",
        "clip_len": 16, "frame_stride": 1, "smooth_window": 7, "top_k": 1,
        "pairs": [{
            "failure": {"episode_index": 0, "start": 0, "end": 16, "critical_t": 8},
            "success_candidates": [
                {"episode_index": 0, "start": 0, "end": 16, "cosine": 1.0, "reward_delta": 0.0}
            ],
        }],
    }))

    from lerobot_policy_rewind_state_ours_variant2.common.factory_robomimic import make_dataset
    from lerobot_policy_rewind_state_ours_variant2.common.preference_datasets import (
        PreferenceTaggedRewindDataset,
    )
    cfg = _fake_cfg("chomeed/robomimic_square_resized_peg_success", True, str(manifest))
    ds = make_dataset(cfg)
    assert isinstance(ds, PreferenceTaggedRewindDataset)
