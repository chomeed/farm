"""Figure: comparison pair (success + failure) for ReWiND vs ours.

Mirrors the video layout from render_compare_videos_success_vs_failure as a
static figure suitable for inclusion in a paper.

Layout:
  Top row:    success frame | failure frame (both at the divergence timestep)
  Bottom row: shared reward curves — solid = success, dashed = failure,
              two colors for ReWiND vs ours. Vertical line at the
              divergence timestep.

Divergence point = argmax of (ReWiND - ours) on the *failure* curves
(smoothed). Pre-failure both models agree (gap ≈ 0); post-failure ours stays
low while ReWiND jumps back up — that's the discriminating moment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lerobot_policy_rewind_state.common.rewind_datasets import EpisodeDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-eval-dir", required=True, type=Path,
                   help="dir containing trajectories.npz for ReWiND baseline")
    p.add_argument("--ours-eval-dir", required=True, type=Path,
                   help="dir containing trajectories.npz for ours")
    p.add_argument("--failure-repo-id", required=True)
    p.add_argument("--failure-ep", type=int, required=True)
    p.add_argument("--success-repo-id", default=None,
                   help="omit to render failure-only layout")
    p.add_argument("--success-ep", type=int, default=-1,
                   help="omit (or -1) to render failure-only layout")
    p.add_argument("--camera-key", default="observation.images.agentview")
    p.add_argument("--smooth-window", type=int, default=5,
                   help="moving-average window for the (ReWiND - ours) signal "
                        "before argmax")
    p.add_argument("--frame-t", type=int, default=-1,
                   help="override the auto-detected divergence frame "
                        "(-1 = auto)")
    p.add_argument("--output-png", required=True, type=Path)
    return p.parse_args()


def _load_pred(eval_dir: Path, split: str, ep: int) -> np.ndarray:
    npz = np.load(eval_dir / "trajectories.npz")
    key = f"{split}/{ep}/predicted_reward"
    if key not in npz.files:
        raise KeyError(f"{key} not in {eval_dir / 'trajectories.npz'}")
    return np.asarray(npz[key], dtype=np.float32)


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) <= w:
        return x
    kernel = np.ones(w, dtype=np.float32) / w
    return np.convolve(x, kernel, mode="same")


def _detect_divergence(base: np.ndarray, ours: np.ndarray, w: int) -> int:
    gap = _smooth(base, w) - _smooth(ours, w)
    return int(np.argmax(gap))


def _load_frame(repo_id: str, ep: int, camera_key: str, t: int) -> np.ndarray:
    ds = EpisodeDataset(
        repo_id=repo_id,
        episodes=[ep],
        camera_keys=[camera_key],
        text_embedding_key=None,
        state_key=None,
        use_precomputed_embeddings_only=False,
        pad_to_length=None,
    )
    item = ds[0]
    frames = item[camera_key]
    T = frames.shape[0]
    t = max(0, min(T - 1, t))
    frame = frames[t].detach().cpu().numpy()
    frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)
    return np.clip(frame * 255.0, 0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()

    show_success = args.success_ep >= 0 and args.success_repo_id is not None

    base_f = _load_pred(args.base_eval_dir, "failure", args.failure_ep)
    ours_f = _load_pred(args.ours_eval_dir, "failure", args.failure_ep)
    Tf = min(len(base_f), len(ours_f))
    base_f, ours_f = base_f[:Tf], ours_f[:Tf]

    if show_success:
        base_s = _load_pred(args.base_eval_dir, "success", args.success_ep)
        ours_s = _load_pred(args.ours_eval_dir, "success", args.success_ep)
        Ts = min(len(base_s), len(ours_s))
        base_s, ours_s = base_s[:Ts], ours_s[:Ts]
    else:
        base_s = ours_s = None
        Ts = 0

    if args.frame_t >= 0:
        div_t = args.frame_t
    else:
        div_t = _detect_divergence(base_f, ours_f, args.smooth_window)
    div_t_f = min(div_t, Tf - 1)
    div_t_s = min(div_t, Ts - 1) if show_success else 0

    fail_frame = _load_frame(args.failure_repo_id, args.failure_ep,
                             args.camera_key, div_t_f)
    succ_frame = (
        _load_frame(args.success_repo_id, args.success_ep,
                    args.camera_key, div_t_s)
        if show_success else None
    )

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    color_base = "tab:gray"
    color_ours = "tab:red"

    if show_success:
        fig = plt.figure(figsize=(10, 8), dpi=150)
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 2],
                              hspace=0.22, wspace=0.05)

        ax_succ = fig.add_subplot(gs[0, 0])
        ax_succ.imshow(succ_frame)
        ax_succ.set_xticks([]); ax_succ.set_yticks([])
        ax_succ.set_title(f"success ep {args.success_ep}, t={div_t_s}",
                          fontsize=11)

        ax_fail = fig.add_subplot(gs[0, 1])
        ax_fail.imshow(fail_frame)
        ax_fail.set_xticks([]); ax_fail.set_yticks([])
        ax_fail.set_title(f"failure ep {args.failure_ep}, t={div_t_f}",
                          fontsize=11)

        ax = fig.add_subplot(gs[1, :])
        ax.plot(np.arange(Ts), base_s, color=color_base, linewidth=1.8,
                linestyle="-",  label="ReWiND (success)")
        ax.plot(np.arange(Tf), base_f, color=color_base, linewidth=1.8,
                linestyle="--", label="ReWiND (failure)")
        ax.plot(np.arange(Ts), ours_s, color=color_ours, linewidth=1.8,
                linestyle="-",  label="ours (success)")
        ax.plot(np.arange(Tf), ours_f, color=color_ours, linewidth=1.8,
                linestyle="--", label="ours (failure)")
    else:
        fig = plt.figure(figsize=(7, 7), dpi=150)
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.22)

        ax_fail = fig.add_subplot(gs[0])
        ax_fail.imshow(fail_frame)
        ax_fail.set_xticks([]); ax_fail.set_yticks([])
        ax_fail.set_title(f"failure ep {args.failure_ep}, t={div_t_f}",
                          fontsize=11)

        ax = fig.add_subplot(gs[1])
        ax.plot(np.arange(Tf), base_f, color=color_base, linewidth=1.8,
                linestyle="-", label="ReWiND")
        ax.plot(np.arange(Tf), ours_f, color=color_ours, linewidth=1.8,
                linestyle="-", label="ours")

    ax.axvline(div_t, color="black", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.scatter([div_t_f], [base_f[div_t_f]], color=color_base,
               edgecolor="black", s=40, zorder=5)
    ax.scatter([div_t_f], [ours_f[div_t_f]], color=color_ours,
               edgecolor="black", s=40, zorder=5)
    ax.set_xlabel("step in episode")
    ax.set_ylabel("predicted reward")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=True, fontsize=9)

    fig.savefig(args.output_png, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output_png}")
    print(f"  failure len={Tf}{'  success len=' + str(Ts) if show_success else ''}")
    print(f"  divergence at t={div_t}: failure ReWiND={base_f[div_t_f]:.3f}, "
          f"ours={ours_f[div_t_f]:.3f}, gap={base_f[div_t_f] - ours_f[div_t_f]:.3f}")


if __name__ == "__main__":
    main()
