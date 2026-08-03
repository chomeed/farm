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

from ._match_helpers import uniform_subsample_indices
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
        # Failure-clip sampling strategy.
        # "manifest": use the manifest's stored subsampled_indices.
        # "hybrid": ignore stored indices, recompute as 8 uniform from
        # [0, raw_end - n_recent) + 8 contiguous at [raw_end - n_recent, raw_end).
        failure_sampling_mode: str = "manifest",
        failure_hybrid_n_recent: int = 8,
        success_sampling_mode: str = "hybrid",
        success_hybrid_n_recent: int = 8,
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
        if failure_sampling_mode not in ("manifest", "hybrid"):
            raise ValueError(
                f"failure_sampling_mode must be 'manifest' or 'hybrid', got {failure_sampling_mode!r}"
            )
        self._failure_sampling_mode = failure_sampling_mode
        self._failure_hybrid_n_recent = int(failure_hybrid_n_recent)
        if self._failure_sampling_mode == "hybrid":
            if self._failure_hybrid_n_recent <= 0 or self._failure_hybrid_n_recent >= self._pref_clip_len:
                raise ValueError(
                    f"failure_hybrid_n_recent ({self._failure_hybrid_n_recent}) must be in (0, "
                    f"preference_clip_len={self._pref_clip_len})"
                )
        if success_sampling_mode not in ("manifest", "hybrid"):
            raise ValueError(
                f"success_sampling_mode must be 'manifest' or 'hybrid', got {success_sampling_mode!r}"
            )
        self._success_sampling_mode = success_sampling_mode
        self._success_hybrid_n_recent = int(success_hybrid_n_recent)
        if self._success_sampling_mode == "hybrid":
            if self._success_hybrid_n_recent <= 0 or self._success_hybrid_n_recent >= self._pref_clip_len:
                raise ValueError(
                    f"success_hybrid_n_recent ({self._success_hybrid_n_recent}) must be in (0, "
                    f"preference_clip_len={self._pref_clip_len})"
                )

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

    def _compute_failure_indices(self, fail: dict) -> list[int]:
        """Return the per-frame absolute indices to load for one failure clip.

        - "manifest" mode: pass through `fail["subsampled_indices"]` (the
          16 indices the precompute script chose around critical_t).
        - "hybrid" mode: pick a random "anchor" from the manifest's
          `subsampled_indices` (so the clip's right endpoint e-1 lands on
          one of the 16 precomputed positions), then recompute as
          n_uniform frames evenly spaced in [0, anchor + 1 - n_recent)
          PLUS n_recent contiguous frames at [anchor + 1 - n_recent,
          anchor + 1). Different anchor per __getitem__ call → augmented
          coverage of the failure tail.
        """
        if self._failure_sampling_mode == "manifest":
            return [int(i) for i in fail["subsampled_indices"]]

        # In "hybrid" mode this single-clip path is no longer used by
        # __getitem__ (we emit a stack of clips, one per manifest anchor).
        # Keep the random-anchor fallback in case external callers use it.
        n_recent = self._failure_hybrid_n_recent
        n_uniform = self._pref_clip_len - n_recent
        candidates = [int(i) for i in fail["subsampled_indices"] if int(i) >= n_recent]
        if not candidates:
            raise ValueError(
                f"Failure pair has no manifest index >= n_recent={n_recent}; "
                f"shrink failure_hybrid_n_recent or rebuild the manifest."
            )
        anchor = int(self._pref_rng.choice(candidates))
        raw_end = anchor + 1
        uniform_idx = uniform_subsample_indices(0, raw_end - n_recent, n_uniform)
        recent_idx = list(range(raw_end - n_recent, raw_end))
        return [int(i) for i in uniform_idx] + recent_idx

    def _compute_failure_anchor_stack(self, fail: dict) -> list[list[int]]:
        """For hybrid mode: return ONE 16-frame hybrid clip per manifest anchor.

        Output: list of `n_anchors` lists, each of length `preference_clip_len`.
        Each inner list is `n_uniform` uniformly-spaced frames in
        [0, anchor + 1 - n_recent) followed by `n_recent` contiguous frames
        ending at `anchor`. The k-th sublist's LAST index is the k-th
        manifest anchor — so a forward pass on this stack yields one
        "logit_t at anchor_k given full context up to anchor_k" per row.
        """
        n_recent = self._failure_hybrid_n_recent
        n_uniform = self._pref_clip_len - n_recent
        anchors = [int(i) for i in fail["subsampled_indices"] if int(i) >= n_recent]
        if not anchors:
            raise ValueError(
                f"Failure pair has no manifest index >= n_recent={n_recent}; "
                f"shrink failure_hybrid_n_recent or rebuild the manifest."
            )
        stack: list[list[int]] = []
        for anchor in anchors:
            raw_end = anchor + 1
            uniform_idx = uniform_subsample_indices(0, raw_end - n_recent, n_uniform)
            recent_idx = list(range(raw_end - n_recent, raw_end))
            stack.append([int(i) for i in uniform_idx] + recent_idx)
        return stack

    def _compute_success_anchor_stack(self, succ: dict) -> list[list[int]]:
        """For success-hybrid mode: return ONE hybrid clip per success anchor.

        Anchor source: success candidates in existing manifests only carry
        `[start, end)`, not `subsampled_indices` (that field is failure-side
        only). When `subsampled_indices` is absent, derive `preference_clip_len`
        anchors uniformly spaced across `[start, end)` — same fallback pattern
        as `_load_pref_clip`. When present (newer manifests), use them as-is.
        """
        n_recent = self._success_hybrid_n_recent
        n_uniform = self._pref_clip_len - n_recent
        if "subsampled_indices" in succ:
            raw_anchors = [int(i) for i in succ["subsampled_indices"]]
        else:
            start = int(succ["start"])
            end = int(succ["end"])
            raw_anchors = list(uniform_subsample_indices(start, end, self._pref_clip_len))
        anchors = [a for a in raw_anchors if a >= n_recent]
        if not anchors:
            raise ValueError(
                f"Success pair has no anchor >= n_recent={n_recent}; "
                f"shrink success_hybrid_n_recent or rebuild the manifest "
                f"(raw_anchors={raw_anchors})."
            )
        stack: list[list[int]] = []
        for anchor in anchors:
            raw_end = anchor + 1
            uniform_idx = uniform_subsample_indices(0, raw_end - n_recent, n_uniform)
            recent_idx = list(range(raw_end - n_recent, raw_end))
            stack.append([int(i) for i in uniform_idx] + recent_idx)
        return stack

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

    def _load_pref_anchor_stack(
        self,
        handle: LeRobotDataset,
        ep_idx: int,
        anchor_indices_list: list[list[int]],
    ) -> dict[str, Any]:
        """Load N_anchors clips and stack each tensor field along new dim 0.

        Returns a dict with the SAME keys as `_load_pref_clip`, but tensors
        carry an extra leading anchor dim:
            observation.images.<cam>: (N_anchors, T_clip, C, H, W)
            observation.state:        (N_anchors, T_clip, state_dim)
            task:                     str  (identical across anchors)
        """
        clips = [
            self._load_pref_clip(handle, ep_idx, start=0, subsampled_indices=indices)
            for indices in anchor_indices_list
        ]
        stacked: dict[str, Any] = {}
        for k in clips[0]:
            v = clips[0][k]
            if isinstance(v, torch.Tensor):
                stacked[k] = torch.stack([c[k] for c in clips], dim=0)
            else:
                stacked[k] = v
        return stacked

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)

        n_pairs = len(self._manifest.pairs)
        pi = int(self._pref_rng.integers(0, n_pairs))
        pair = self._manifest.pairs[pi]

        n_cands = min(self._pref_topk, len(pair["success_candidates"]))
        ci = int(self._pref_rng.integers(0, n_cands))
        succ = pair["success_candidates"][ci]
        fail = pair["failure"]

        if self._failure_sampling_mode == "hybrid":
            anchor_stack = self._compute_failure_anchor_stack(fail)
            fail_clip = self._load_pref_anchor_stack(
                self._failure_handle,
                ep_idx=int(fail["episode_index"]),
                anchor_indices_list=anchor_stack,
            )
        else:
            fail_indices = self._compute_failure_indices(fail)
            fail_clip = self._load_pref_clip(
                self._failure_handle,
                ep_idx=int(fail["episode_index"]),
                start=int(fail["start"]),
                subsampled_indices=fail_indices,
            )

        # Direction tag — used by the model's mono loss to flip the hinge
        # direction per-batch-element (leaving→decrease, approaching→increase).
        # Default "leaving" for older manifests without the field.
        direction = str(fail.get("direction", "leaving"))
        fail_clip["_is_approaching"] = torch.tensor(
            1 if direction == "approaching" else 0, dtype=torch.long
        )
        if self._success_sampling_mode == "hybrid":
            succ_anchor_stack = self._compute_success_anchor_stack(succ)
            succ_clip = self._load_pref_anchor_stack(
                self._success_handle,
                ep_idx=int(succ["episode_index"]),
                anchor_indices_list=succ_anchor_stack,
            )
            # Per-anchor absolute (episode-relative) frame index — last frame
            # of each hybrid clip. Used by the model's MSE target.
            succ_clip["_anchor_frames"] = torch.tensor(
                [s[-1] for s in succ_anchor_stack], dtype=torch.long,
            )
            # Per-episode length, so the model can normalize the anchor frame
            # to a [0, 1] progress target.
            succ_ep = self._success_handle.meta.episodes[int(succ["episode_index"])]
            ep_len = int(succ_ep["length"]) if "length" in succ_ep else (
                int(succ_ep["dataset_to_index"]) - int(succ_ep["dataset_from_index"])
            )
            succ_clip["_episode_length"] = torch.tensor(ep_len, dtype=torch.long)
        else:
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
