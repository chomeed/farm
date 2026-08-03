"""
Score every preference-pair clip in the variant7 manifest with both the base
`rewind_state` reward model and the `rewind_state_ours_variant7` reward model,
and report per-pair overlays plus aggregate success-vs-failure separation.

Unlike `compare_base_vs_ours_dense_reward.py` (full success episodes), this
script evaluates ONLY the short `clip_len`-frame windows named in the manifest.

Usage:
    python -m lerobot_policy_rewind_state_ours_variant7.common.compare_rewind_vs_variant7_clips \
        --variant7_ckpt /scratch2/chomeed/outputs/rewind/rewind_state_ours_variant7_robomimic_square_v1/checkpoints/last/pretrained_model \
        --output_dir /scratch2/chomeed/outputs/rewind/compare_rewind_vs_variant7_clips
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors

from lerobot_policy_rewind_state.common.eval_rewind_reward_visualization_dense import (
    dense_predict_per_frame,
)
from .manifest import load_manifest
from .preference_datasets import (
    _episode_task,
    _query_feature,
    _query_video,
    _strided,
)

CAMERA_KEYS = [
    "observation.images.agentview",
    "observation.images.eye_in_hand",
]


def _clip_scalars(curve: np.ndarray) -> dict[str, float]:
    """Two scalars summarising one clip's per-frame reward curve."""
    if curve.size == 0:
        raise ValueError("_clip_scalars: curve must be non-empty")
    return {"clip_mean": float(np.mean(curve)), "final": float(curve[-1])}


def aggregate_stats(records: list[dict]) -> dict:
    """Aggregate per-clip scored records into per-model summary stats.

    Each record is a dict:
        {"clip_class": "success"|"failure",
         "base":     {"clip_mean": float, "final": float},
         "variant7": {"clip_mean": float, "final": float}}

    Returns, for each of "base" and "variant7":
        {"success": {"clip_mean": {"mean","std"}, "final": {"mean","std"}},
         "failure": {...},
         "separation_clip_mean": mean(success.clip_mean) - mean(failure.clip_mean),
         "separation_final":     mean(success.final)     - mean(failure.final)}
    """
    models = ("base", "variant7")
    metrics = ("clip_mean", "final")
    classes = ("success", "failure")
    out: dict = {}
    for model in models:
        block: dict = {}
        for cls in classes:
            cls_recs = [r for r in records if r["clip_class"] == cls]
            block[cls] = {}
            for metric in metrics:
                vals = np.array(
                    [r[model][metric] for r in cls_recs], dtype=np.float64
                )
                block[cls][metric] = {
                    "mean": float(vals.mean()) if vals.size else float("nan"),
                    "std": float(vals.std()) if vals.size else float("nan"),  # population std (ddof=0)
                }
        for metric in metrics:
            block[f"separation_{metric}"] = (
                block["success"][metric]["mean"]
                - block["failure"][metric]["mean"]
            )
        out[model] = block
    return out


def _move_to_device(batch: dict, device: str) -> dict:
    return {
        k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def _load_policy_and_preproc(ckpt_path: str, ds_meta, device: str):
    cfg = PreTrainedConfig.from_pretrained(ckpt_path)
    cfg.pretrained_path = ckpt_path
    cfg.device = device
    policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    policy.eval()
    policy.to(device)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=None,
        dataset_stats=ds_meta.stats if hasattr(ds_meta, "stats") else None,
    )
    return policy, preprocessor


def _load_clip(
    handle: LeRobotDataset,
    ep_idx: int,
    start: int,
    clip_len: int,
    frame_stride: int,
) -> dict:
    """Load one `clip_len`-frame window. Mirrors
    `preference_datasets.PreferenceTaggedRewindDataset._load_pref_clip`."""
    n_eps = handle.meta.total_episodes
    if ep_idx < 0 or ep_idx >= n_eps:
        raise ValueError(
            f"Manifest references episode_index={ep_idx} but the dataset "
            f"handle has only {n_eps} episodes. Regenerate the manifest."
        )
    abs_indices = _strided(start, clip_len, frame_stride)
    ep = handle.meta.episodes[ep_idx]
    ep_start = int(ep["dataset_from_index"])
    ep_end = int(ep["dataset_to_index"])
    abs_global = [ep_start + i for i in abs_indices]
    if abs_global[-1] >= ep_end:
        raise ValueError(
            f"Manifest clip overshoots episode bounds: ep_idx={ep_idx} "
            f"start={start} clip_len={clip_len} frame_stride={frame_stride}. "
            f"Regenerate the manifest."
        )
    out: dict = {}
    # CAMERA_KEYS is fixed to the two Robomimic Square cameras both models use.
    for cam in CAMERA_KEYS:
        out[cam] = _query_video(handle, ep_idx, abs_global, cam).float()
    out["observation.state"] = _query_feature(
        handle, ep_idx, abs_global, "observation.state"
    ).float()
    out["task"] = _episode_task(handle, ep_idx)
    return out


@torch.no_grad()
def _score_clip(policy, preprocessor, clip: dict, device: str) -> np.ndarray:
    """Per-frame reward (numpy, length == clip_len) for one clip."""
    batch = {
        **{cam: clip[cam].unsqueeze(0) for cam in CAMERA_KEYS},
        "observation.state": clip["observation.state"].unsqueeze(0),
        "task": [clip["task"]],
        "episode_index": torch.tensor([0]),  # not used by preprocessor; placeholder
    }
    batch = _move_to_device(batch, device)
    batch = preprocessor(batch)
    raw_images = {k: batch[k] for k in CAMERA_KEYS}
    lang = batch["task.minilm"]
    if lang.ndim == 1:
        lang = lang.unsqueeze(0)
    state = batch.get("observation.state")
    pred = dense_predict_per_frame(
        policy, raw_images, lang, state=state, sampling="uniform"
    )
    return pred.detach().float().cpu().numpy()


def _plot_pair(pair_index: int, fail_rec: dict, succ_rec: dict,
               out_path: Path) -> None:
    """Two panels: failure clip | matched (top-1) success clip, each
    overlaying the base and variant7 per-frame reward curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, rec, title in (
        (axes[0], fail_rec, "failure"),
        (axes[1], succ_rec, "success"),
    ):
        ax.plot(rec["curve_base"], label="base rewind_state",
                color="tab:blue", linewidth=1.5)
        ax.plot(rec["curve_v3"], label="variant7",
                color="tab:orange", linewidth=1.5)
        if title == "failure" and rec.get("critical_t") is not None:
            ci = int(rec["critical_t"]) - int(rec["start"])
            n = min(len(rec["curve_base"]), len(rec["curve_v3"]))
            if 0 <= ci < n:
                ax.axvline(ci, color="tab:red", linestyle="--",
                           linewidth=1.0, label="critical_t")
        ax.set_xlabel("frame within clip")
        ax.set_title(f"{title}  ep{rec['episode_index']}  start{rec['start']}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel("pred reward")
    fig.suptitle(f"manifest pair {pair_index}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_aggregate(records: list[dict], out_path: Path) -> None:
    """2x2 boxplot grid: rows = {clip_mean, final}, cols = {base, variant7};
    each subplot contrasts success vs failure clips."""
    metrics = ("clip_mean", "final")
    models = (("base", "base rewind_state"), ("variant7", "variant7"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    rng = np.random.default_rng(0)
    for ri, metric in enumerate(metrics):
        for ci, (mkey, mlabel) in enumerate(models):
            ax = axes[ri][ci]
            succ = [r[mkey][metric] for r in records
                    if r["clip_class"] == "success"]
            fail = [r[mkey][metric] for r in records
                    if r["clip_class"] == "failure"]
            ax.boxplot([succ, fail], tick_labels=["success", "failure"])
            for x, vals in ((1, succ), (2, fail)):
                if vals:
                    jitter = rng.normal(x, 0.04, size=len(vals))
                    ax.scatter(jitter, vals, alpha=0.4, s=12,
                               color="tab:gray", zorder=3)
            ax.set_title(f"{mlabel} — {metric}")
            ax.set_ylabel("pred reward")
            ax.grid(True, alpha=0.3)
    fig.suptitle("clip-level reward: success vs failure")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _clip_img(clip: dict, cam: str, t: int) -> np.ndarray:
    """One clip frame as an (H, W, 3) float array clipped to [0, 1]."""
    frame = clip[cam][t].permute(1, 2, 0).cpu().numpy()
    return np.clip(frame, 0.0, 1.0)


def _open_video_writer(out_path: Path, fps: int):
    """imageio MP4 writer with the libx264/yuv420p settings used everywhere."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        str(out_path), fps=fps, codec="libx264", pixelformat="yuv420p",
        macro_block_size=1, ffmpeg_params=["-movflags", "+faststart"],
    )


def _fig_to_frame(fig) -> np.ndarray:
    """Draw a figure and return its RGB buffer, padded to even dimensions."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    if buf.shape[0] % 2 or buf.shape[1] % 2:
        buf = np.pad(
            buf, ((0, buf.shape[0] % 2), (0, buf.shape[1] % 2), (0, 0)),
            mode="edge")
    return buf.copy()


def _render_clip_panel_frames(pair_index: int, clip: dict, rec: dict,
                              title: str) -> list[np.ndarray]:
    """1x3 panel frames for a single clip: agentview and eye_in_hand images
    beside the base/variant7 reward curves, with a marker sweeping the clip.
    Used to build the concatenated per-class aggregate videos."""
    n_frames = min(
        clip[CAMERA_KEYS[0]].shape[0],
        len(rec["curve_base"]),
        len(rec["curve_v3"]),
    )
    frames: list[np.ndarray] = []
    for t in range(n_frames):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].imshow(_clip_img(clip, CAMERA_KEYS[0], t))
        axes[0].set_title(f"{title} agentview")
        axes[0].axis("off")
        axes[1].imshow(_clip_img(clip, CAMERA_KEYS[1], t))
        axes[1].set_title(f"{title} eye_in_hand")
        axes[1].axis("off")

        ax = axes[2]
        ax.plot(rec["curve_base"], label="base rewind_state",
                color="tab:blue", linewidth=1.5)
        ax.plot(rec["curve_v3"], label="variant7",
                color="tab:orange", linewidth=1.5)
        ax.axvline(t, color="black", linewidth=1.2)
        if title == "failure" and rec.get("critical_t") is not None:
            ci = int(rec["critical_t"]) - int(rec["start"])
            n = min(len(rec["curve_base"]), len(rec["curve_v3"]))
            if 0 <= ci < n:
                ax.axvline(ci, color="tab:red", linestyle="--",
                           linewidth=1.0, label="critical_t")
        ax.set_xlabel("frame within clip")
        ax.set_ylabel("pred reward")
        ax.set_title(f"{title}  ep{rec['episode_index']}  "
                     f"start{rec['start']}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.suptitle(f"pair {pair_index} — {title} — "
                     f"frame {t}/{n_frames - 1}")
        fig.tight_layout()
        frames.append(_fig_to_frame(fig))
        plt.close(fig)
    return frames


def _render_pair_video(pair_index: int, fail_clip: dict, fail_rec: dict,
                       succ_clip: dict, succ_rec: dict,
                       out_path: Path, fps: int) -> None:
    """Per-pair synced MP4. One 2x3 figure per clip frame: agentview and
    eye_in_hand images beside the base/variant7 reward curves, with a marker
    sweeping the clip. Row 0 = failure clip, row 1 = top-1 success clip."""
    n_frames = min(
        fail_clip[CAMERA_KEYS[0]].shape[0],
        succ_clip[CAMERA_KEYS[0]].shape[0],
        len(fail_rec["curve_base"]),
        len(succ_rec["curve_base"]),
    )
    if n_frames == 0:
        print(f"[warn] pair {pair_index}: empty clip, skipping video.")
        return

    rows = (
        (fail_clip, fail_rec, "failure"),
        (succ_clip, succ_rec, "success"),
    )
    frames: list[np.ndarray] = []
    for t in range(n_frames):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for ri, (clip, rec, title) in enumerate(rows):
            axes[ri][0].imshow(_clip_img(clip, CAMERA_KEYS[0], t))
            axes[ri][0].set_title(f"{title} agentview")
            axes[ri][0].axis("off")
            axes[ri][1].imshow(_clip_img(clip, CAMERA_KEYS[1], t))
            axes[ri][1].set_title(f"{title} eye_in_hand")
            axes[ri][1].axis("off")

            ax = axes[ri][2]
            ax.plot(rec["curve_base"], label="base rewind_state",
                    color="tab:blue", linewidth=1.5)
            ax.plot(rec["curve_v3"], label="variant7",
                    color="tab:orange", linewidth=1.5)
            ax.axvline(t, color="black", linewidth=1.2)
            if title == "failure" and rec.get("critical_t") is not None:
                ci = int(rec["critical_t"]) - int(rec["start"])
                n = min(len(rec["curve_base"]), len(rec["curve_v3"]))
                if 0 <= ci < n:
                    ax.axvline(ci, color="tab:red", linestyle="--",
                               linewidth=1.0, label="critical_t")
            ax.set_xlabel("frame within clip")
            ax.set_ylabel("pred reward")
            ax.set_title(f"{title}  ep{rec['episode_index']}  "
                         f"start{rec['start']}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)
        fig.suptitle(f"manifest pair {pair_index} — frame {t}/{n_frames - 1}")
        fig.tight_layout()
        frames.append(_fig_to_frame(fig))
        plt.close(fig)

    writer = _open_video_writer(out_path, fps)
    try:
        for f in frames:
            writer.append_data(f)
    finally:
        writer.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare rewind_state vs variant7 reward models on "
                    "preference-pair manifest clips."
    )
    p.add_argument(
        "--manifest",
        default="/scratch2/chomeed/manifests/rewind_state_ours_variant7/"
                "preference_pairs.json",
        help="preference-pair manifest JSON",
    )
    p.add_argument(
        "--base_ckpt", default=None,
        help="base rewind_state .../pretrained_model. "
             "Defaults to the manifest's base_ckpt.",
    )
    p.add_argument(
        "--variant7_ckpt", required=True,
        help="rewind_state_ours_variant7 .../pretrained_model",
    )
    p.add_argument(
        "--use_all_candidates", action="store_true",
        help="score every success candidate per pair (default: top-1 only)",
    )
    p.add_argument(
        "--max_pairs", type=int, default=None,
        help="cap on number of manifest pairs processed (debugging). "
             "Use a value large enough that at least one success and one "
             "failure clip are scored, else separation stats are NaN.",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device for both reward models",
    )
    p.add_argument(
        "--fps", type=int, default=10,
        help="frame rate of the per-pair synced MP4",
    )
    p.add_argument(
        "--output_dir", required=True,
        help="directory for per-pair PNGs/MP4s, aggregate success.mp4/"
             "failure.mp4, aggregate.png and summary.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest)
    base_ckpt = args.base_ckpt or manifest.base_ckpt
    if args.base_ckpt is not None:
        same = (Path(args.base_ckpt).resolve()
                == Path(manifest.base_ckpt).resolve())
        if not same:
            print(f"[warn] --base_ckpt {args.base_ckpt!r} differs from "
                  f"manifest base_ckpt {manifest.base_ckpt!r}; "
                  f"comparing against the supplied one anyway.")

    pairs = manifest.pairs
    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]
    print(f"[1/4] Manifest {args.manifest}: {len(pairs)} pairs "
          f"(clip_len={manifest.clip_len}, frame_stride={manifest.frame_stride})")

    print(f"[2/4] Opening dataset handles: "
          f"{manifest.success_repo_id} / {manifest.failure_repo_id}")
    success_handle = LeRobotDataset(repo_id=manifest.success_repo_id)
    failure_handle = LeRobotDataset(repo_id=manifest.failure_repo_id)
    ds_meta = success_handle.meta

    print(f"[3/4] Loading checkpoints\n      base:     {base_ckpt}\n"
          f"      variant7: {args.variant7_ckpt}")
    # Both models use the success handle's ds_meta (and its normalization
    # stats); valid here because both reward models share the same dataset.
    base_policy, base_preproc = _load_policy_and_preproc(
        base_ckpt, ds_meta, args.device)
    v3_policy, v3_preproc = _load_policy_and_preproc(
        args.variant7_ckpt, ds_meta, args.device)

    def _score(clip: dict, clip_class: str, ep_idx: int, start: int,
               critical_t: int | None) -> dict:
        curve_base = _score_clip(base_policy, base_preproc, clip, args.device)
        curve_v3 = _score_clip(v3_policy, v3_preproc, clip, args.device)
        return {
            "clip_class": clip_class,
            "episode_index": int(ep_idx),
            "start": int(start),
            "critical_t": (None if critical_t is None else int(critical_t)),
            "curve_base": curve_base,
            "curve_v3": curve_v3,
            "base": _clip_scalars(curve_base),
            "variant7": _clip_scalars(curve_v3),
        }

    print(f"[4/4] Scoring clips and writing per-pair plots + videos "
          f"and aggregate success/failure videos")
    records: list[dict] = []
    # Aggregate per-class videos: every scored clip's panel frames are
    # streamed straight into these writers so clip tensors are never held
    # in memory beyond the iteration that loads them.
    success_writer = _open_video_writer(out_dir / "success.mp4", args.fps)
    failure_writer = _open_video_writer(out_dir / "failure.mp4", args.fps)
    try:
        for pi, pair in enumerate(tqdm(pairs, desc="pairs")):
            fail = pair["failure"]
            fail_clip = _load_clip(
                failure_handle, int(fail["episode_index"]),
                int(fail["start"]), manifest.clip_len, manifest.frame_stride)
            fail_rec = _score(fail_clip, "failure", fail["episode_index"],
                              fail["start"], fail.get("critical_t"))
            records.append(fail_rec)
            for fr in _render_clip_panel_frames(pi, fail_clip, fail_rec,
                                                "failure"):
                failure_writer.append_data(fr)

            cands = pair["success_candidates"]
            kept = cands if args.use_all_candidates else cands[:1]
            succ_recs = []
            top1_succ_clip = None
            for cand in kept:
                succ_clip = _load_clip(
                    success_handle, int(cand["episode_index"]),
                    int(cand["start"]), manifest.clip_len,
                    manifest.frame_stride)
                succ_rec = _score(succ_clip, "success", cand["episode_index"],
                                  cand["start"], None)
                records.append(succ_rec)
                succ_recs.append(succ_rec)
                for fr in _render_clip_panel_frames(pi, succ_clip, succ_rec,
                                                    "success"):
                    success_writer.append_data(fr)
                if top1_succ_clip is None:
                    top1_succ_clip = succ_clip

            # Per-pair plot and video always use the top-1 success candidate.
            if not succ_recs:
                print(f"[warn] pair {pi}: no success candidates, "
                      f"skipping per-pair plot/video.")
                continue
            _plot_pair(pi, fail_rec, succ_recs[0],
                       out_dir / f"pair_{pi:04d}.png")
            _render_pair_video(pi, fail_clip, fail_rec, top1_succ_clip,
                               succ_recs[0], out_dir / f"pair_{pi:04d}.mp4",
                               args.fps)
    finally:
        success_writer.close()
        failure_writer.close()

    _plot_aggregate(records, out_dir / "aggregate.png")

    stats = aggregate_stats(records)
    n_fail = sum(1 for r in records if r["clip_class"] == "failure")
    n_succ = sum(1 for r in records if r["clip_class"] == "success")
    summary = {
        "manifest_path": args.manifest,
        "base_ckpt": base_ckpt,
        "variant7_ckpt": args.variant7_ckpt,
        "use_all_candidates": bool(args.use_all_candidates),
        "n_failure_clips": n_fail,
        "n_success_clips": n_succ,
        "models": stats,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nScored {n_succ} success clips / {n_fail} failure clips")
    for model in ("base", "variant7"):
        b = stats[model]
        print(f"\n[{model}]")
        for cls in ("success", "failure"):
            cm = b[cls]["clip_mean"]
            fn = b[cls]["final"]
            print(f"  {cls:8s}  clip_mean={cm['mean']:.4f}±{cm['std']:.4f}  "
                  f"final={fn['mean']:.4f}±{fn['std']:.4f}")
        print(f"  separation  clip_mean={b['separation_clip_mean']:+.4f}  "
              f"final={b['separation_final']:+.4f}")
    print(f"\nSaved per-pair plots/videos, aggregate success.mp4/failure.mp4, "
          f"aggregate.png and summary.json to {out_dir}")


if __name__ == "__main__":
    main()
