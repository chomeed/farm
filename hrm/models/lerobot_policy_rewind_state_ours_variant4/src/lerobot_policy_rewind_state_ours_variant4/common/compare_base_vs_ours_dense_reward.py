"""
Overlay per-frame predicted reward from the base `rewind_state` checkpoint and
the `rewind_state_ours` checkpoint on the same SUCCESS episodes, and surface
frames where the two disagree.

For each episode we plot three rows:
  (a) base pred reward vs. ours pred reward, overlaid
  (b) diff = base - ours (positive = ours dips below base; candidate
      "bad-within-success" frame the ours model down-weights)
  (c) the top-K frames ranked by diff, marked on top of the base curve

Usage:
    python -m lerobot_policy_rewind_state_ours_variant4.common.compare_base_vs_ours_dense_reward \
        --base_ckpt /home/chomeed/workspace/insertion_generation/ckpts/rewind/rewind_state_robomimic_square_v1_pro6000/checkpoints/010000/pretrained_model \
        --ours_ckpt /scratch2/chomeed/outputs/rewind/rewind_state_ours_robomimic_square_v1/checkpoints/last/pretrained_model \
        --repo_id chomeed/robomimic_square_resized_peg_success \
        --num_episodes 10 \
        --output_dir /scratch2/chomeed/outputs/rewind/compare_base_vs_ours
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot_policy_rewind_state.common.rewind_datasets import EpisodeDataset
from lerobot_policy_rewind_state.common.eval_rewind_reward_visualization_dense import (
    dense_predict_per_frame,
)

CAMERA_KEYS = [
    "observation.images.agentview",
    "observation.images.eye_in_hand",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_ckpt",
                   default="/home/chomeed/workspace/insertion_generation/ckpts/rewind/rewind_state_robomimic_square_v1_pro6000/checkpoints/010000/pretrained_model",
                   help="path to base rewind_state .../pretrained_model")
    p.add_argument("--ours_ckpt", required=True,
                   help="path to rewind_state_ours .../pretrained_model")
    p.add_argument("--repo_id", default="chomeed/robomimic_square_resized_peg_success",
                   help="success dataset to evaluate on")
    p.add_argument("--num_episodes", type=int, default=10)
    p.add_argument("--episode_offset", type=int, default=0,
                   help="skip the first K episodes (useful to look at the val tail)")
    p.add_argument("--top_k_marks", type=int, default=5,
                   help="how many largest-diff frames to highlight per episode")
    p.add_argument("--smooth_window", type=int, default=1,
                   help="rolling mean width applied to each reward curve before diffing (1 = no smoothing)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def _move_to_device(batch, device):
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


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or x.size < w:
        return x
    kernel = np.ones(w, dtype=np.float32) / w
    return np.convolve(x, kernel, mode="same")


@torch.no_grad()
def _episode_pred(policy, preprocessor, item: dict, device: str) -> np.ndarray:
    batch = {
        **{cam: item[cam].unsqueeze(0) for cam in CAMERA_KEYS},
        "observation.state": item["observation.state"].unsqueeze(0),
        "task": [item["task"]],
        "episode_index": torch.tensor([item["episode_index"]]),
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


def _plot_episode(ep_idx: int, base: np.ndarray, ours: np.ndarray,
                  top_k: int, out_path: Path) -> dict:
    T = min(len(base), len(ours))
    base = base[:T]
    ours = ours[:T]
    diff = base - ours
    order = np.argsort(-diff)[:top_k]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.plot(base, label="base rewind_state", color="tab:blue", linewidth=1.4)
    ax.plot(ours, label="ours (rewind_state_ours)", color="tab:orange", linewidth=1.4)
    ax.set_ylabel("pred reward")
    ax.set_title(f"episode {ep_idx} — per-frame reward overlay")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    ax.plot(diff, color="tab:red", linewidth=1.2, label="base − ours")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylabel("base − ours")
    ax.set_title("disagreement (positive = ours dips below base)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    ax = axes[2]
    ax.plot(base, color="tab:blue", linewidth=1.0, alpha=0.7, label="base")
    ax.plot(ours, color="tab:orange", linewidth=1.0, alpha=0.7, label="ours")
    ax.scatter(order, ours[order], color="tab:red", zorder=5, s=40,
               label=f"top-{top_k} disagreement frames")
    for t in order:
        ax.axvline(t, color="tab:red", linewidth=0.5, alpha=0.4)
    ax.set_xlabel("timestep")
    ax.set_ylabel("pred reward")
    ax.set_title("top-K candidate bad-within-success frames")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    return {
        "episode_index": int(ep_idx),
        "T": int(T),
        "diff_mean": float(diff.mean()),
        "diff_max": float(diff.max()),
        "diff_min": float(diff.min()),
        "diff_l1": float(np.abs(diff).mean()),
        "top_k_frames": [int(t) for t in order.tolist()],
        "top_k_diffs": [float(diff[t]) for t in order.tolist()],
    }


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = list(range(args.episode_offset,
                          args.episode_offset + args.num_episodes))
    print(f"[1/3] Loading dataset {args.repo_id} for episodes {episodes[0]}..{episodes[-1]}")
    dataset = EpisodeDataset(
        repo_id=args.repo_id,
        episodes=episodes,
        camera_keys=CAMERA_KEYS,
        text_embedding_key=None,
        state_key="observation.state",
        use_precomputed_embeddings_only=False,
        pad_to_length=None,
    )
    ds_meta = dataset.meta

    print(f"[2/3] Loading base ckpt: {args.base_ckpt}")
    base_policy, base_preproc = _load_policy_and_preproc(
        args.base_ckpt, ds_meta, args.device
    )
    print(f"      Loading ours ckpt: {args.ours_ckpt}")
    ours_policy, ours_preproc = _load_policy_and_preproc(
        args.ours_ckpt, ds_meta, args.device
    )

    print(f"[3/3] Running dense reward inference on {len(dataset)} episodes")
    summary = []
    for i in tqdm(range(len(dataset)), desc="episodes"):
        item = dataset[i]
        ep_idx = int(item["episode_index"])

        base_r = _smooth(_episode_pred(base_policy, base_preproc, item, args.device),
                         args.smooth_window)
        ours_r = _smooth(_episode_pred(ours_policy, ours_preproc, item, args.device),
                         args.smooth_window)

        out_path = out_dir / f"compare_ep{ep_idx:05d}.png"
        stats = _plot_episode(ep_idx, base_r, ours_r, args.top_k_marks, out_path)
        stats["plot"] = str(out_path)
        summary.append(stats)
        print(f"  ep={ep_idx:4d}  T={stats['T']:4d}  "
              f"|diff|.mean={stats['diff_l1']:.3f}  "
              f"diff.max={stats['diff_max']:+.3f}  "
              f"diff.min={stats['diff_min']:+.3f}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {
                "base_ckpt": args.base_ckpt,
                "ours_ckpt": args.ours_ckpt,
                "repo_id": args.repo_id,
                "smooth_window": args.smooth_window,
                "episodes": summary,
            },
            f, indent=2,
        )
    print(f"\nSaved {len(summary)} plots and summary.json to {out_dir}")


if __name__ == "__main__":
    main()
