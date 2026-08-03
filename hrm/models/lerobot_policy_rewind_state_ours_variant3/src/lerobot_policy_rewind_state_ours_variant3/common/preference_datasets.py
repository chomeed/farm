"""
PreferenceTaggedRewindDataset.

Wraps the base `RewindDataset` so each `__getitem__` returns the standard
rewind sample *plus* one (failure-critical, success-anchor) preference pair
drawn from the manifest produced by `precompute_preference_pairs.py`.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_policy_rewind_state.common.rewind_datasets import RewindDataset

from .manifest import load_manifest


def _strided(start: int, clip_len: int, stride: int) -> list[int]:
    return [start + i * stride for i in range(clip_len)]


def _query_video(
    dataset: LeRobotDataset, ep_idx: int, abs_indices: list[int], camera_key: str
) -> torch.Tensor:
    """Mirror RewindDataset._load_single_camera_frames for a foreign dataset.

    `abs_indices` are absolute hf-dataset row indices. The caller in
    `_load_pref_clip` is responsible for translating within-episode offsets
    via `ep_start`.
    """
    if camera_key in dataset.meta.video_keys:
        timestamps = dataset.hf_dataset[abs_indices]["timestamp"]
        timestamps = [
            ts.item() if isinstance(ts, torch.Tensor) else float(ts)
            for ts in timestamps
        ]
        return dataset._query_videos({camera_key: timestamps}, ep_idx)[camera_key]
    items = dataset.hf_dataset[abs_indices][camera_key]
    if isinstance(items, torch.Tensor):
        return items
    return torch.stack(items)


def _query_feature(
    dataset: LeRobotDataset, ep_idx: int, abs_indices: list[int], feature_key: str
) -> torch.Tensor:
    items = dataset.hf_dataset[abs_indices][feature_key]
    if isinstance(items, torch.Tensor):
        return items
    if len(items) == 0:
        raise ValueError(f"No values for {feature_key!r}")
    if isinstance(items[0], torch.Tensor):
        return torch.stack(items)
    return torch.as_tensor(np.asarray(items))


def _episode_task(dataset: LeRobotDataset, ep_idx: int) -> str:
    ep = dataset.meta.episodes[ep_idx]
    ep_start = int(ep["dataset_from_index"])
    item = dataset.hf_dataset[ep_start]
    task_idx = (
        item["task_index"].item()
        if isinstance(item["task_index"], torch.Tensor)
        else int(item["task_index"])
    )
    return dataset.meta.tasks.iloc[task_idx].name


class PreferenceTaggedRewindDataset(RewindDataset):
    """RewindDataset + one preference pair per __getitem__.

    Reads a JSON manifest produced by `precompute_preference_pairs.py`.
    The pair is loaded from the dataset handles named in the manifest, NOT
    from `self`. This means the wrapper can be instantiated on any base
    dataset; the manifest fully controls where pref clips come from.
    """

    def __init__(
        self,
        *args,
        preference_pairs_path: str,
        preference_clip_len: int,
        preference_frame_stride: int,
        preference_topk: int,
        preference_seed: int = 0,
        # Optional override for asserting against the base ckpt at __init__.
        preference_assert_base_ckpt: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        manifest = load_manifest(preference_pairs_path)
        if preference_assert_base_ckpt is not None:
            manifest.assert_compatible(
                base_ckpt=preference_assert_base_ckpt,
                clip_len=preference_clip_len,
                frame_stride=preference_frame_stride,
            )
        else:
            # Skip ckpt check (caller has not supplied it), but still enforce
            # clip-len/stride consistency.
            if int(manifest.clip_len) != int(preference_clip_len):
                raise ValueError(
                    f"Manifest clip_len={manifest.clip_len} != "
                    f"preference_clip_len={preference_clip_len}"
                )
            if int(manifest.frame_stride) != int(preference_frame_stride):
                raise ValueError(
                    f"Manifest frame_stride={manifest.frame_stride} != "
                    f"preference_frame_stride={preference_frame_stride}"
                )

        if not manifest.pairs:
            raise ValueError("Preference manifest is empty.")

        self._manifest = manifest
        self._pref_clip_len = int(preference_clip_len)
        self._pref_frame_stride = int(preference_frame_stride)
        self._pref_topk = max(1, int(preference_topk))
        self._pref_rng = np.random.default_rng(preference_seed)

        # Inherit video_backend from the base dataset so pref handles use the
        # same decoder (important on nodes where torchcodec is unavailable).
        _video_backend = getattr(self, "video_backend", None)
        self._success_handle = LeRobotDataset(
            repo_id=manifest.success_repo_id,
            video_backend=_video_backend,
        )
        self._failure_handle = LeRobotDataset(
            repo_id=manifest.failure_repo_id,
            video_backend=_video_backend,
        )

        # Camera keys to load for pref clips: the same encoder camera keys the
        # parent `RewindDataset` already uses, which is itself derived from
        # the policy config when the wrapper is built by the factory.
        self._pref_camera_keys = list(self.camera_keys) if self.camera_keys else []

    def _load_pref_clip(
        self,
        handle: LeRobotDataset,
        ep_idx: int,
        start: int,
        subsampled_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        n_eps = handle.meta.total_episodes
        if ep_idx < 0 or ep_idx >= n_eps:
            raise ValueError(
                f"Manifest references episode_index={ep_idx} but the dataset "
                f"handle has only {n_eps} episodes. The manifest is likely "
                f"stale; regenerate it with precompute_preference_pairs.py."
            )
        # New variable-length manifests carry the exact frames used at match
        # time as `subsampled_indices`; fall back to strided sampling from
        # `start` for older fixed-length manifests.
        if subsampled_indices is not None:
            abs_indices = [int(i) for i in subsampled_indices]
            if len(abs_indices) != self._pref_clip_len:
                raise ValueError(
                    f"subsampled_indices length ({len(abs_indices)}) does not "
                    f"match preference_clip_len ({self._pref_clip_len}). "
                    f"Regenerate the manifest."
                )
        else:
            abs_indices = _strided(
                start, self._pref_clip_len, self._pref_frame_stride
            )
        ep = handle.meta.episodes[ep_idx]
        ep_start = int(ep["dataset_from_index"])
        abs_global = [ep_start + i for i in abs_indices]

        # Manifest invariant: the precompute script clamps clips into
        # [0, ep_len). If a stale manifest violates this we'd silently cross
        # episode boundaries (absolute indexing is flat across all episodes).
        ep_end = int(ep["dataset_to_index"])
        if max(abs_global) >= ep_end:
            raise ValueError(
                f"Manifest pref clip overshoots episode bounds: ep_idx={ep_idx} "
                f"start={start} max_rel={max(abs_indices)} -> hf row "
                f"{max(abs_global)} but episode ends at {ep_end - 1}. "
                f"Regenerate the manifest."
            )

        out: dict[str, Any] = {}
        for cam in self._pref_camera_keys:
            out[cam] = _query_video(handle, ep_idx, abs_global, cam).float()
        out["observation.state"] = _query_feature(
            handle, ep_idx, abs_global, "observation.state"
        ).float()
        out["task"] = _episode_task(handle, ep_idx)
        return out

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)

        n_pairs = len(self._manifest.pairs)
        pi = int(self._pref_rng.integers(0, n_pairs))
        pair = self._manifest.pairs[pi]

        n_cands = min(self._pref_topk, len(pair["success_candidates"]))
        ci = int(self._pref_rng.integers(0, n_cands))
        succ = pair["success_candidates"][ci]
        fail = pair["failure"]

        fail_clip = self._load_pref_clip(
            self._failure_handle,
            ep_idx=int(fail["episode_index"]),
            start=int(fail["start"]),
            subsampled_indices=fail.get("subsampled_indices"),
        )
        succ_clip = self._load_pref_clip(
            self._success_handle,
            ep_idx=int(succ["episode_index"]),
            start=int(succ["start"]),
            subsampled_indices=succ.get("subsampled_indices"),
        )

        for k, v in fail_clip.items():
            item[f"pref.failure.{k}"] = v
        for k, v in succ_clip.items():
            item[f"pref.success.{k}"] = v
        return item
