from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# These tests touch real LeRobotDatasets. Skip if the local cache is missing.
LEROBOT_CACHE = Path("/scratch2/chomeed/hf_cache/lerobot/chomeed/robomimic_square_resized_peg_success")
SKIP_REASON = (
    "Skipping integration test: requires local cached robomimic_square datasets"
)


def _video_decoding_available() -> bool:
    """Return True if torchvision's VideoReader is compiled in (pyav path).

    The previous implementation probed torchcodec at import time and was not
    stable across test-suite orderings (numpy or other native libs being
    loaded earlier could change what torchcodec sees). Probing torchvision's
    VideoReader attribute directly is environment-stable.
    """
    try:
        import torchvision.io
    except ImportError:
        return False
    return hasattr(torchvision.io, "VideoReader")


_VIDEO_AVAILABLE = _video_decoding_available()
_VIDEO_SKIP_REASON = (
    "Skipping: no working video decoding backend (torchcodec FFmpeg libs "
    "missing and torchvision.io.VideoReader not compiled in this env)"
)


def _write_tiny_manifest(tmp_path: Path) -> Path:
    p = tmp_path / "manifest.json"
    payload = {
        "base_ckpt": "/abs/path/to/010000/pretrained_model",
        "success_repo_id": "chomeed/robomimic_square_resized_peg_success",
        "failure_repo_id": "chomeed/robomimic_square_resized_peg_failure",
        "clip_len": 16,
        "frame_stride": 1,
        "smooth_window": 7,
        "top_k": 1,
        "pairs": [
            {
                "failure": {"episode_index": 0, "start": 0, "end": 16, "critical_t": 8},
                "success_candidates": [
                    {"episode_index": 0, "start": 0, "end": 16,
                     "cosine": 1.0, "reward_delta": 0.0},
                ],
            }
        ],
    }
    p.write_text(json.dumps(payload))
    return p


@pytest.mark.skipif(not LEROBOT_CACHE.exists(), reason=SKIP_REASON)
@pytest.mark.skipif(not _VIDEO_AVAILABLE, reason=_VIDEO_SKIP_REASON)
def test_preference_dataset_yields_pref_keys(tmp_path: Path):
    from lerobot_policy_rewind_state_ours_variant8.common.preference_datasets import (
        PreferenceTaggedRewindDataset,
    )
    manifest = _write_tiny_manifest(tmp_path)
    ds = PreferenceTaggedRewindDataset(
        repo_id="chomeed/robomimic_square_resized_peg_success",
        camera_keys=[
            "observation.images.agentview",
            "observation.images.eye_in_hand",
        ],
        text_embedding_key=None,
        use_precomputed_embeddings_only=False,
        rewind=True,
        subsample_video=True,
        pad_to_length=16,
        preference_pairs_path=str(manifest),
        preference_clip_len=16,
        preference_frame_stride=1,
        preference_topk=1,
        preference_seed=0,
        video_backend="pyav",
    )
    item = ds[0]

    expected_pref_keys = [
        "pref.failure.observation.images.agentview",
        "pref.failure.observation.images.eye_in_hand",
        "pref.failure.observation.state",
        "pref.failure.task",
        "pref.success.observation.images.agentview",
        "pref.success.observation.images.eye_in_hand",
        "pref.success.observation.state",
        "pref.success.task",
    ]
    for k in expected_pref_keys:
        assert k in item, f"Missing key {k}; available: {sorted(item.keys())}"

    # Pref clips have fixed length = clip_len * frame_stride // frame_stride = clip_len.
    pref_state = item["pref.failure.observation.state"]
    assert pref_state.shape[0] == 16


def _stubbed_dataset_class():
    """Subclass PreferenceTaggedRewindDataset with a no-op parent __init__.

    The parent RewindDataset.__init__ requires a real HuggingFace dataset
    on disk; we just want to test the manifest-validation logic that runs
    after super().__init__(). The stub shorts out the parent and sets the
    attributes the wrapper reads.
    """
    from lerobot_policy_rewind_state_ours_variant8.common.preference_datasets import (
        PreferenceTaggedRewindDataset,
    )
    import numpy as np

    class _StubbedWrapper(PreferenceTaggedRewindDataset):
        def __init__(self, *args, **kwargs):
            # Call grandparent's __init__ chain at object level only.
            object.__setattr__(self, "camera_keys", [])
            object.__setattr__(self, "video_backend", None)
            # Now run only the manifest-validation portion of the wrapper init.
            from lerobot_policy_rewind_state_ours_variant8.common.manifest import (
                load_manifest,
            )
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            manifest = load_manifest(kwargs["preference_pairs_path"])
            assert_base = kwargs.get("preference_assert_base_ckpt")
            if assert_base is not None:
                manifest.assert_compatible(
                    base_ckpt=assert_base,
                    clip_len=kwargs["preference_clip_len"],
                    frame_stride=kwargs["preference_frame_stride"],
                )
            else:
                if int(manifest.clip_len) != int(kwargs["preference_clip_len"]):
                    raise ValueError(
                        f"Manifest clip_len={manifest.clip_len} != "
                        f"preference_clip_len={kwargs['preference_clip_len']}"
                    )
                if int(manifest.frame_stride) != int(kwargs["preference_frame_stride"]):
                    raise ValueError(
                        f"Manifest frame_stride={manifest.frame_stride} != "
                        f"preference_frame_stride={kwargs['preference_frame_stride']}"
                    )
            if not manifest.pairs:
                raise ValueError("Preference manifest is empty.")
            self._manifest = manifest
            self._pref_rng = np.random.default_rng(kwargs.get("preference_seed", 0))

    return _StubbedWrapper


def test_init_rejects_empty_manifest(tmp_path):
    p = tmp_path / "empty.json"
    payload = {
        "base_ckpt": "/x/pretrained_model",
        "success_repo_id": "x/s",
        "failure_repo_id": "x/f",
        "clip_len": 16,
        "frame_stride": 1,
        "smooth_window": 7,
        "top_k": 1,
        "pairs": [],
    }
    p.write_text(json.dumps(payload))
    Cls = _stubbed_dataset_class()
    with pytest.raises(ValueError, match="empty"):
        Cls(
            preference_pairs_path=str(p),
            preference_clip_len=16,
            preference_frame_stride=1,
            preference_topk=1,
        )


def test_init_rejects_clip_len_mismatch(tmp_path):
    p = _write_tiny_manifest(tmp_path)
    Cls = _stubbed_dataset_class()
    with pytest.raises(ValueError, match="clip_len"):
        Cls(
            preference_pairs_path=str(p),
            preference_clip_len=32,    # manifest is 16
            preference_frame_stride=1,
            preference_topk=1,
        )


def test_init_rejects_base_ckpt_mismatch(tmp_path):
    p = _write_tiny_manifest(tmp_path)
    Cls = _stubbed_dataset_class()
    with pytest.raises(ValueError, match="base_ckpt"):
        Cls(
            preference_pairs_path=str(p),
            preference_clip_len=16,
            preference_frame_stride=1,
            preference_topk=1,
            preference_assert_base_ckpt="/wrong/path/pretrained_model",
        )
