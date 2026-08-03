"""
Compare the potential-shaping term of the base rewind_state model and the
variant 4 model, side by side, on success and failure demos.

For each (model, episode) it computes the per-frame progress phi
(sigmoid(fusion_net), via dense_predict_per_frame) and the potential term
P[t] = gamma * phi[t + delta] - phi[t]. It draws a 2x2 figure:
  - columns: base rewind (left), variant4 (right)
  - rows:    raw phi (top), potential P (bottom)
each panel overlaying every success episode (green) and failure episode
(red), with the steepest-phi-drop anchor marked on each curve.

The loading / inference / drawing helpers are reused from
visualize_potential_shaping.py; this script only adds the second checkpoint
and the 2x2 layout.

Usage:
    python -m lerobot_policy_rewind_state_ours_variant4.common.compare_rewind_vs_variant4_potential \
        --output_dir /scratch2/chomeed/outputs/rewind/compare_potential
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from lerobot_policy_rewind_state_ours_variant4.common.visualize_potential_shaping import (
    DEFAULT_CKPT,
    _build_dataset,
    _draw_class,
    _load_policy_and_preproc,
    _mean_potential,
    _run,
)

DEFAULT_BASE_CKPT = (
    "/home/chomeed/workspace/insertion_generation/ckpts/rewind/"
    "rewind_state_robomimic_square_v1_pro6000/checkpoints/010000/pretrained_model"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant4_ckpt", default=DEFAULT_CKPT,
                   help="variant 4 .../pretrained_model directory")
    p.add_argument("--base_ckpt", default=DEFAULT_BASE_CKPT,
                   help="base rewind_state .../pretrained_model directory")
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


def _plot_2x2(base_runs: dict, v4_runs: dict, gamma: float, delta: int,
              out_path: Path) -> None:
    """base_runs / v4_runs are dicts {"success": runs, "failure": runs},
    each runs being a list of (ep_idx, phi, P, t_drop) tuples."""
    green, red = "tab:green", "tab:red"
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True, sharey="row")

    for col, (title, runs) in enumerate(
        [("base rewind", base_runs), ("variant4", v4_runs)]
    ):
        col_axes = axes[:, col]
        _draw_class(col_axes, runs["success"], green)
        _draw_class(col_axes, runs["failure"], red)
        col_axes[0].set_title(title)
        col_axes[0].grid(True, alpha=0.3)
        col_axes[1].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
        col_axes[1].grid(True, alpha=0.3)
        col_axes[1].set_xlabel("frame index t")

    axes[0, 0].set_ylabel("phi = sigmoid(fusion_net)")
    axes[1, 0].set_ylabel(f"potential  gamma*phi[t+{delta}] - phi[t]")
    axes[0, 0].legend(handles=[
        plt.Line2D([], [], color=green, label="success"),
        plt.Line2D([], [], color=red, label="failure"),
        plt.Line2D([], [], marker="o", linestyle="None", color="gray",
                   markeredgecolor="black", markersize=5,
                   label="steepest-phi-drop anchor"),
    ], loc="best", fontsize=9)

    fig.suptitle(f"potential-shaping: base rewind vs variant4  "
                 f"(gamma={gamma}, delta={delta})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = list(range(args.episode_offset,
                          args.episode_offset + args.num_episodes))

    print("[1/5] Loading datasets")
    succ_ds = _build_dataset(args.success_repo_id, episodes)
    fail_ds = _build_dataset(args.failure_repo_id, episodes)

    print(f"[2/5] Loading base rewind ckpt {args.base_ckpt}")
    base_policy, base_preproc = _load_policy_and_preproc(
        args.base_ckpt, succ_ds.meta, args.device
    )
    print(f"[3/5] Loading variant 4 ckpt {args.variant4_ckpt}")
    v4_policy, v4_preproc = _load_policy_and_preproc(
        args.variant4_ckpt, succ_ds.meta, args.device
    )

    print("[4/5] Inference: base x {success,failure}, variant4 x {success,failure}")
    base_runs = {
        "success": _run(base_policy, base_preproc, succ_ds, args),
        "failure": _run(base_policy, base_preproc, fail_ds, args),
    }
    v4_runs = {
        "success": _run(v4_policy, v4_preproc, succ_ds, args),
        "failure": _run(v4_policy, v4_preproc, fail_ds, args),
    }

    print("[5/5] Plotting")
    out_path = out_dir / "compare_rewind_vs_variant4_potential.png"
    _plot_2x2(base_runs, v4_runs, args.gamma, args.delta, out_path)

    print()
    for model_name, runs in [("base rewind", base_runs), ("variant4", v4_runs)]:
        for cls in ("success", "failure"):
            r = runs[cls]
            print(f"{model_name:13s} {cls:8s}: {len(r):3d} episodes  "
                  f"mean potential = {_mean_potential(r):+.4f}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
