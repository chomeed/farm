"""
Overlay variant 4's potential-shaping term for success and failure demos.

For each episode this computes:
  - phi: the per-frame progress reward sigmoid(fusion_net), via
    dense_predict_per_frame
  - P[t] = gamma * phi[t + delta] - phi[t]   (variant 4's BT discriminand)
  - the steepest single-step phi drop frame (variant 4's failure-clip anchor)

It draws one figure with two rows (raw phi, potential P) overlaying every
success episode (green) and every failure episode (red), with the drop
anchor marked on each curve.

Usage:
    python -m lerobot_policy_rewind_state_ours_variant4.common.visualize_potential_shaping \
        --ckpt /path/to/variant4/.../pretrained_model \
        --output_dir /scratch2/chomeed/outputs/rewind/potential_shaping
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot_policy_rewind_state.common.eval_rewind_reward_visualization_dense import (
    dense_predict_per_frame,
)
from lerobot_policy_rewind_state.common.rewind_datasets import EpisodeDataset
from lerobot_policy_rewind_state_ours_variant4.common._match_helpers import (
    potential_drop_timestep,
)

CAMERA_KEYS = [
    "observation.images.agentview",
    "observation.images.eye_in_hand",
]

DEFAULT_CKPT = (
    "/home/chomeed/workspace/insertion_generation/ckpts/rewind/"
    "rewind_state_ours_variant4_robomimic_square_v1_eps200_clip100_longer/"
    "checkpoints/050000/pretrained_model"
)


def potential_curve(phi: np.ndarray, gamma: float, delta: int) -> np.ndarray:
    """Per-frame potential gamma*phi[t+delta] - phi[t].

    Returns an array of length len(phi) - delta (empty when len(phi) <= delta).
    Requires delta >= 1.
    """
    if delta < 1:
        raise ValueError(f"delta must be >= 1, got {delta}")
    if phi.shape[0] <= delta:
        return np.empty(0, dtype=phi.dtype)
    return gamma * phi[delta:] - phi[:-delta]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=DEFAULT_CKPT,
                   help="variant 4 .../pretrained_model directory")
    p.add_argument("--success_repo_id",
                   default="chomeed/robomimic_square_resized_peg_success",
                   help="success demo dataset")
    p.add_argument("--failure_repo_id",
                   default="chomeed/robomimic_square_resized_peg_failure",
                   help="failure demo dataset")
    p.add_argument("--num_episodes", type=int, default=10,
                   help="episodes per class")
    p.add_argument("--episode_offset", type=int, default=0,
                   help="skip the first K episodes of each source")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="potential discount (config reward_pref_gamma)")
    p.add_argument("--delta", type=int, default=8,
                   help="potential interval (config reward_pref_potential_delta)")
    p.add_argument("--smooth_window", type=int, default=7,
                   help="rolling-mean width for the drop-anchor helper")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def _move_to_device(batch, device):
    return {
        k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def _build_dataset(repo_id: str, episodes: list[int]) -> EpisodeDataset:
    return EpisodeDataset(
        repo_id=repo_id,
        episodes=episodes,
        camera_keys=CAMERA_KEYS,
        text_embedding_key=None,
        state_key="observation.state",
        use_precomputed_embeddings_only=False,
        pad_to_length=None,
    )


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


@torch.no_grad()
def _episode_phi(policy, preprocessor, item: dict, device: str) -> np.ndarray:
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


def _run(policy, preproc, dataset, args) -> list[tuple]:
    """Inference over one class; returns [(ep_idx, phi, P, t_drop), ...]."""
    out = []
    for i in tqdm(range(len(dataset)), desc="episodes"):
        item = dataset[i]
        ep_idx = int(item["episode_index"])
        phi = _episode_phi(policy, preproc, item, args.device)
        # need T >= delta+1 for a non-empty P and T >= 2 for the drop-anchor
        # np.diff -> require T >= delta+2
        if phi.shape[0] < args.delta + 2:
            print(f"  skip ep={ep_idx}: T={phi.shape[0]} "
                  f"< delta+2={args.delta + 2}")
            continue
        P = potential_curve(phi, args.gamma, args.delta)
        t_drop = potential_drop_timestep(phi, args.smooth_window)
        out.append((ep_idx, phi, P, t_drop))
    return out


def _mean_potential(runs: list[tuple]) -> float:
    if not runs:
        return 0.0
    return float(np.mean([P.mean() for _, _, P, _ in runs]))


def _draw_class(axes, runs: list[tuple], color: str) -> None:
    for _ep_idx, phi, P, t_drop in runs:
        axes[0].plot(phi, color=color, alpha=0.35, linewidth=1.1)
        axes[0].scatter([t_drop], [phi[t_drop]], color=color, s=22,
                        edgecolors="black", linewidths=0.4, zorder=5)
        axes[1].plot(P, color=color, alpha=0.35, linewidth=1.1)
        if t_drop < len(P):
            axes[1].scatter([t_drop], [P[t_drop]], color=color, s=22,
                            edgecolors="black", linewidths=0.4, zorder=5)


def _plot(success: list[tuple], failure: list[tuple], gamma: float,
          delta: int, out_path: Path) -> None:
    green, red = "tab:green", "tab:red"
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    _draw_class(axes, success, green)
    _draw_class(axes, failure, red)

    axes[0].set_ylabel("phi = sigmoid(fusion_net)")
    axes[0].set_title("per-frame progress phi  —  "
                      "success (green) vs failure (red)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(handles=[
        plt.Line2D([], [], color=green, label="success"),
        plt.Line2D([], [], color=red, label="failure"),
        plt.Line2D([], [], marker="o", linestyle="None", color="gray",
                   markeredgecolor="black", markersize=5,
                   label="steepest-phi-drop anchor"),
    ], loc="best", fontsize=9)

    axes[1].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel(f"potential  gamma*phi[t+{delta}] - phi[t]")
    axes[1].set_xlabel("frame index t")
    axes[1].set_title(f"potential-shaping term "
                      f"(gamma={gamma}, delta={delta}); "
                      f"dots = steepest-phi-drop anchor")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = list(range(args.episode_offset,
                          args.episode_offset + args.num_episodes))

    print(f"[1/4] Loading success dataset {args.success_repo_id}")
    succ_ds = _build_dataset(args.success_repo_id, episodes)

    print(f"[2/4] Loading variant 4 ckpt {args.ckpt}")
    policy, preproc = _load_policy_and_preproc(
        args.ckpt, succ_ds.meta, args.device
    )

    print("[3/4] Inference on success episodes")
    success = _run(policy, preproc, succ_ds, args)
    print("      Inference on failure episodes")
    fail_ds = _build_dataset(args.failure_repo_id, episodes)
    failure = _run(policy, preproc, fail_ds, args)

    print("[4/4] Plotting")
    out_path = out_dir / "potential_shaping_overlay.png"
    _plot(success, failure, args.gamma, args.delta, out_path)

    s_mean = _mean_potential(success)
    f_mean = _mean_potential(failure)
    print(f"\nsuccess: {len(success):3d} episodes  mean potential = {s_mean:+.4f}")
    print(f"failure: {len(failure):3d} episodes  mean potential = {f_mean:+.4f}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
