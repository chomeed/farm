"""
Render the failure<->success clip pairs from a precompute_preference_pairs.py
manifest as side-by-side MP4 videos.

For each sampled pair, one MP4 is written showing a row of panels:

    FAIL | GOOD#1 | GOOD#2 | GOOD#3

The failure clip is the failure-critical query; GOOD#k are the top-K
success-anchor candidates the manifest matched to it by cosine. Each panel
shows both camera views stacked, labeled with manifest metadata.

No model is loaded — the manifest already contains every clip's episode
index and start/end, so only raw image data is needed.

Run:
    python -m lerobot_policy_rewind_state_ours_variant8.common.visualize_preference_pairs \
        --manifest /scratch2/chomeed/manifests/rewind_state_ours_variant8/preference_pairs.json \
        --n_pairs 8 \
        --output_dir /tmp/pref_pairs_viz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .manifest import load_manifest
from .preference_datasets import _strided, _query_video
from .make_good_vs_bad_clip_pair import (
    CAMERA_KEYS,
    PANEL_H,
    _frame_to_rgb,
    _resize_h,
    _label_strip,
)

DEFAULT_MANIFEST = (
    "/scratch2/chomeed/manifests/rewind_state_ours_variant8/"
    "preference_pairs.json"
)

_GAP = 8          # px gap between panels / camera views
_BG = 30          # background grey value


def _panel_frame(panel: dict, t: int) -> np.ndarray:
    """One panel's frame at time t: its camera views concatenated H-wise."""
    imgs = [_resize_h(_frame_to_rgb(panel[cam][t]), PANEL_H)
            for cam in CAMERA_KEYS]
    out = imgs[0]
    for img in imgs[1:]:
        gap = np.full((out.shape[0], _GAP, 3), _BG, dtype=np.uint8)
        out = np.concatenate([out, gap, img], axis=1)
    return out


def _render_row(
    panels: list[dict],
    labels: list[str],
    out_path: "Path | str",
    fps: int,
) -> None:
    """Write a T-frame MP4: panels concatenated horizontally, each labeled.

    `panels` — list of dicts mapping each CAMERA_KEYS entry to a
    (T, C, H, W) float tensor in [0, 1]. All panels must share the same T.
    `labels` — one label string per panel.
    """
    if len(panels) != len(labels):
        raise ValueError(
            f"Got {len(panels)} panels but {len(labels)} labels; "
            f"they must match."
        )
    if not panels:
        raise ValueError("No panels to render.")
    T = panels[0][CAMERA_KEYS[0]].shape[0]

    frames = []
    for t in range(T):
        panel_imgs = [_panel_frame(p, t) for p in panels]
        h = max(img.shape[0] for img in panel_imgs)

        body = panel_imgs[0]
        title = _label_strip(labels[0], body.shape[1])
        for img, lab in zip(panel_imgs[1:], labels[1:]):
            body_gap = np.full((h, _GAP, 3), _BG, dtype=np.uint8)
            body = np.concatenate([body, body_gap, img], axis=1)
            title_gap = np.full((title.shape[0], _GAP, 3), _BG, dtype=np.uint8)
            title = np.concatenate(
                [title, title_gap, _label_strip(lab, img.shape[1])], axis=1
            )

        frame = np.concatenate([title, body], axis=0)
        if frame.shape[1] % 2:
            frame = np.pad(frame, ((0, 0), (0, 1), (0, 0)), mode="edge")
        if frame.shape[0] % 2:
            frame = np.pad(frame, ((0, 1), (0, 0), (0, 0)), mode="edge")
        frames.append(frame)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264", pixelformat="yuv420p",
        macro_block_size=1, ffmpeg_params=["-movflags", "+faststart"],
    )
    try:
        for f in frames:
            writer.append_data(f)
    finally:
        writer.close()


def _sample_pairs(pairs: list[dict], n_pairs: int, seed: int) -> list[dict]:
    """Randomly sample up to n_pairs entries from the manifest pair list."""
    if not pairs:
        raise ValueError("Manifest pair list is empty; nothing to render.")
    rng = np.random.default_rng(seed)
    n = min(n_pairs, len(pairs))
    idx = rng.choice(len(pairs), size=n, replace=False)
    return [pairs[int(i)] for i in idx]


def _load_clip(
    handle: LeRobotDataset,
    ep_idx: int,
    start: int,
    clip_len: int,
    frame_stride: int,
    subsampled_indices: list[int] | None = None,
) -> dict:
    """Load one clip's camera frames as a panel dict.

    Returns {camera_key: (T, C, H, W) float tensor}. When
    `subsampled_indices` is provided (new variable-length manifests), those
    episode-relative indices are used verbatim and `start/clip_len/frame_stride`
    are ignored. Otherwise frame selection matches
    PreferenceTaggedRewindDataset._load_pref_clip (strided from `start`).
    """
    n_eps = handle.meta.total_episodes
    if ep_idx < 0 or ep_idx >= n_eps:
        raise ValueError(
            f"Manifest references episode_index={ep_idx} but the dataset "
            f"handle has only {n_eps} episodes. Regenerate the manifest."
        )
    ep = handle.meta.episodes[ep_idx]
    ep_start = int(ep["dataset_from_index"])
    ep_end = int(ep["dataset_to_index"])
    if subsampled_indices is not None:
        rel = [int(i) for i in subsampled_indices]
    else:
        rel = _strided(start, clip_len, frame_stride)
    abs_global = [ep_start + i for i in rel]
    if abs_global[-1] >= ep_end:
        raise ValueError(
            f"Manifest clip overshoots episode bounds: ep_idx={ep_idx} "
            f"last_rel={rel[-1]} -> hf row {abs_global[-1]} but episode ends "
            f"at {ep_end - 1}. Regenerate the manifest."
        )
    panel: dict = {}
    for cam in CAMERA_KEYS:
        panel[cam] = _query_video(handle, ep_idx, abs_global, cam).float()
    return panel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=DEFAULT_MANIFEST,
                   help="Manifest JSON produced by precompute_preference_pairs.py")
    p.add_argument("--n_pairs", type=int, default=8,
                   help="Number of manifest pairs to render (clamped to "
                        "manifest size).")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for pair sampling.")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--output_dir", required=True,
                   help="Directory to write one MP4 per pair into.")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[1/3] Loading manifest {args.manifest}")
    manifest = load_manifest(args.manifest)
    clip_len = int(manifest.clip_len)
    frame_stride = int(manifest.frame_stride)
    print(f"      {len(manifest.pairs)} pairs, top_k={manifest.top_k}, "
          f"clip_len={clip_len}, frame_stride={frame_stride}")

    print(f"[2/3] Opening dataset handles")
    success_handle = LeRobotDataset(repo_id=manifest.success_repo_id)
    failure_handle = LeRobotDataset(repo_id=manifest.failure_repo_id)

    sampled = _sample_pairs(manifest.pairs, args.n_pairs, args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[3/3] Rendering {len(sampled)} pairs -> {out_dir}")

    for i, pair in enumerate(sampled):
        fail = pair["failure"]
        fail_ep = int(fail["episode_index"])
        fail_start = int(fail["start"])
        fail_end = int(fail["end"])
        fail_crit = int(fail.get("critical_t", -1))

        panels = [_load_clip(
            failure_handle, fail_ep, fail_start, clip_len, frame_stride,
            subsampled_indices=fail.get("subsampled_indices"),
        )]
        labels = [
            f"FAIL ep={fail_ep} crit_t={fail_crit} [{fail_start},{fail_end})"
        ]

        for k, cand in enumerate(pair["success_candidates"], start=1):
            s_ep = int(cand["episode_index"])
            s_start = int(cand["start"])
            panels.append(_load_clip(
                success_handle, s_ep, s_start, clip_len, frame_stride,
                subsampled_indices=cand.get("subsampled_indices"),
            ))
            cos_val = cand.get("cosine")
            dtw_val = cand.get("dtw_distance")
            if cos_val is not None:
                metric_str = f"cos={float(cos_val):.3f}"
            elif dtw_val is not None:
                metric_str = f"dtw={float(dtw_val):.3f}"
            else:
                metric_str = ""
            labels.append(
                f"GOOD#{k} ep={s_ep} {metric_str} "
                f"dD={float(cand['reward_delta']):.3f}"
            )

        out_path = out_dir / f"pair_{i:03d}_failep{fail_ep}.mp4"
        _render_row(panels, labels, out_path, args.fps)
        print(f"  [{i + 1}/{len(sampled)}] wrote {out_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
