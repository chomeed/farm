"""
Post-process eval_val_loss_and_visualize outputs: plot

    composite = pred_reward * σ(pref_score)

alongside the raw reward and the σ(pref_score) gate, for the first N success
and failure episodes in trajectories.npz.

Run after eval_val_loss_and_visualize has produced trajectories.npz.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", required=True,
                   help="path to .../eval_val_last50 (contains trajectories.npz)")
    p.add_argument("--num_visualize", type=int, default=5)
    return p.parse_args()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _interp_pref_to_per_frame(
    pref: np.ndarray, last_idx: np.ndarray, T: int
) -> np.ndarray:
    """Linear-interpolate the sliding pref scores onto a per-frame x axis.

    `last_idx[i]` is the last frame of the i-th clip; `pref[i]` is the score.
    Frames before `last_idx[0]` have no clip score; we hold the first value.
    """
    if pref.size == 0:
        return np.full(T, np.nan, dtype=np.float32)
    per_frame = np.full(T, np.nan, dtype=np.float32)
    # Hold the first score for the unattributable head.
    per_frame[: int(last_idx[0])] = float(pref[0])
    per_frame[int(last_idx[0]) :] = np.interp(
        np.arange(int(last_idx[0]), T),
        last_idx.astype(np.float32),
        pref.astype(np.float32),
    )
    return per_frame.astype(np.float32)


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    data = np.load(eval_dir / "trajectories.npz")

    # Discover available episodes per source.
    available: dict[str, list[int]] = {"success": [], "failure": []}
    for k in data.files:
        if "/predicted_reward" in k:
            src, ep, _ = k.split("/")
            available[src].append(int(ep))
    available["success"] = sorted(available["success"])
    available["failure"] = sorted(available["failure"])
    print(f"Found {len(available['success'])} success + "
          f"{len(available['failure'])} failure episodes in {eval_dir}")

    for source in ("success", "failure"):
        ep_ids = available[source][: args.num_visualize]
        n = len(ep_ids)
        if n == 0:
            continue

        fig, axes = plt.subplots(n, 3, figsize=(24, 3 * n), squeeze=False)
        for row, ep in enumerate(ep_ids):
            pred = data[f"{source}/{ep}/predicted_reward"]
            pref = data[f"{source}/{ep}/preference_score"]
            last_idx = data[f"{source}/{ep}/preference_last_idx"]
            T = pred.shape[0]
            pref_per_frame = _interp_pref_to_per_frame(pref, last_idx, T)
            gate = _sigmoid(pref_per_frame)
            composite = pred * gate

            ax0 = axes[row, 0]
            ax0.plot(pred, color="tab:blue", label="raw reward")
            target = np.arange(T, dtype=np.float32) / max(T - 1, 1)
            ax0.plot(target, color="tab:gray", linestyle="--", alpha=0.4,
                     label="frame-progress target")
            ax0.set_ylim(-0.05, 1.05)
            ax0.set_title(f"{source} ep {ep} - raw reward")
            ax0.set_xlabel("frame")
            ax0.grid(True, alpha=0.3)
            ax0.legend(fontsize=8, loc="best")

            ax1 = axes[row, 1]
            ax1.plot(gate, color="tab:purple", label="σ(pref_score)")
            ax1.set_ylim(-0.05, 1.05)
            ax1.axhline(0.5, color="gray", linewidth=0.5)
            ax1.set_title(f"{source} ep {ep} - pref gate σ(pref_score)")
            ax1.set_xlabel("frame")
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=8, loc="best")

            ax2 = axes[row, 2]
            ax2.plot(composite, color="tab:green",
                     label="composite = reward × σ(pref)")
            ax2.plot(pred, color="tab:blue", alpha=0.3, label="raw reward")
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_title(f"{source} ep {ep} - composite signal")
            ax2.set_xlabel("frame")
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=8, loc="best")

        fig.tight_layout()
        out_path = eval_dir / f"{source}_composite_reward.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        print(f"  saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
