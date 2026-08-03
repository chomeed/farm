"""
Compare per-frame predicted rewards from several trained `rewind` checkpoints
on the same set of robomimic_square success and failure validation demos.

Each checkpoint must already have been evaluated by
`lerobot_policy_rewind_state_ours_variant6.common.eval_val_loss_and_visualize`,
which writes `trajectories.npz` and `summary.json` under an `eval_val_last*/`
directory. This script only reads those cached artifacts (no model loading,
no GPU). It produces:

  - success_overlay.png  : first N success episodes, all models overlaid per subplot
  - failure_overlay.png  : same for failure episodes
  - mean_curve.png       : mean predicted reward over progress-normalized time
                            (one line per model; success on left, failure on right)
  - summary.csv          : per-model val_mse mean/median on success and failure

Episodes are intersected across the supplied evals before plotting, so the
comparison is apples-to-apples even when different evals covered different
episode ranges.

Example:
  python -m lerobot_policy_rewind_state_ours_variant6.common.compare_rewind_models_reward_predictions \\
      --output_dir /home/chomeed/workspace/insertion_generation/ckpts/rewind/_compare_six \\
      --num_visualize 8 \\
      --model "variant6:/home/chomeed/workspace/insertion_generation/ckpts/rewind/rewind_state_ours_variant6_robomimic_square_v1_eps200_clip100_longer/eval_val_last" \\
      --model "varlen_blend:/.../rewind_state_ours_variant6_varlen_blend_..._longer/eval_val_last" \\
      ...
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", action="append", required=True,
        help="'label:/abs/path/to/eval_val_last' - repeat for each model",
    )
    p.add_argument("--num_visualize", type=int, default=8,
                   help="how many episodes per source to overlay in the grid")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def _parse_models(specs: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"--model must be 'label:/path', got: {spec!r}")
        label, path = spec.split(":", 1)
        out.append((label.strip(), Path(path.strip())))
    return out


def _load_one(eval_dir: Path) -> dict:
    npz = np.load(eval_dir / "trajectories.npz")
    success: dict[int, np.ndarray] = {}
    failure: dict[int, np.ndarray] = {}
    for key in npz.files:
        # key format: "{split}/{ep_idx}/predicted_reward"
        split, ep_str, _ = key.split("/")
        ep = int(ep_str)
        arr = np.asarray(npz[key], dtype=np.float32)
        (success if split == "success" else failure)[ep] = arr

    summary_path = eval_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return {"success": success, "failure": failure, "summary": summary}


def _intersect_episodes(per_model: list[dict], split: str) -> list[int]:
    sets = [set(d[split].keys()) for d in per_model]
    common = sorted(set.intersection(*sets))
    return common


def _plot_overlay_grid(
    split: str,
    models: list[tuple[str, dict]],
    episodes: list[int],
    num_visualize: int,
    out_path: Path,
):
    ep_ids = episodes[:num_visualize]
    if not ep_ids:
        print(f"[{split}] no common episodes - skipping {out_path.name}")
        return

    n = len(ep_ids)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.6 * n), squeeze=False)
    cmap = plt.get_cmap("tab10")

    for row, ep_idx in enumerate(ep_ids):
        ax = axes[row, 0]
        Ts: list[int] = []
        for mi, (label, data) in enumerate(models):
            pred = data[split][ep_idx]
            Ts.append(int(pred.shape[0]))
            ax.plot(pred, color=cmap(mi % 10), linewidth=1.1, label=label)
        T = max(Ts)
        target = np.arange(T, dtype=np.float32) / max(T - 1, 1)
        ax.plot(target, color="black", linestyle="--", linewidth=0.8,
                alpha=0.5, label="frame-progress target")
        ax.set_title(f"{split} ep {ep_idx}  (T={T})")
        ax.set_xlabel("frame")
        ax.set_ylabel("pred reward")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.legend(loc="upper left", fontsize=7, ncol=2)

    fig.suptitle(f"{split.title()} demos - predicted reward (overlay across {len(models)} models)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved {out_path}")


def _resample_to_unit(curve: np.ndarray, n_bins: int) -> np.ndarray:
    """Linear-interpolate a length-T curve onto n_bins evenly spaced points
    on the unit progress axis [0, 1].
    """
    T = curve.shape[0]
    if T < 2:
        return np.full((n_bins,), float(curve[0]) if T == 1 else np.nan, dtype=np.float32)
    src_x = np.linspace(0.0, 1.0, T, dtype=np.float32)
    tgt_x = np.linspace(0.0, 1.0, n_bins, dtype=np.float32)
    return np.interp(tgt_x, src_x, curve.astype(np.float32)).astype(np.float32)


def _plot_mean_curves(
    models: list[tuple[str, dict]],
    success_eps: list[int],
    failure_eps: list[int],
    out_path: Path,
    n_bins: int = 200,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    cmap = plt.get_cmap("tab10")

    for split, ax, eps in (
        ("success", axes[0], success_eps),
        ("failure", axes[1], failure_eps),
    ):
        for mi, (label, data) in enumerate(models):
            stacked = np.stack(
                [_resample_to_unit(data[split][ep], n_bins) for ep in eps],
                axis=0,
            )  # (E, n_bins)
            mean = np.nanmean(stacked, axis=0)
            std = np.nanstd(stacked, axis=0)
            x = np.linspace(0.0, 1.0, n_bins)
            ax.plot(x, mean, color=cmap(mi % 10), linewidth=1.4, label=label)
            ax.fill_between(x, mean - std, mean + std,
                            color=cmap(mi % 10), alpha=0.10, linewidth=0)
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8,
                alpha=0.5, label="frame-progress target")
        ax.set_title(f"{split} (mean +/- std over {len(eps)} episodes)")
        ax.set_xlabel("normalized progress")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if split == "success":
            ax.set_ylabel("pred reward")
            ax.legend(loc="upper left", fontsize=8, ncol=2)

    fig.suptitle("Mean predicted reward over progress-normalized time", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved {out_path}")


def _write_summary_csv(
    models: list[tuple[str, dict]],
    success_eps: list[int],
    failure_eps: list[int],
    out_path: Path,
):
    rows = []
    for label, data in models:
        # Recompute from the intersected episode set so the numbers compare
        # like-for-like across models, regardless of what each model's own
        # summary.json reported.
        s_target_mse = []
        for ep in success_eps:
            pred = data["success"][ep]
            T = pred.shape[0]
            tgt = np.arange(T, dtype=np.float32) / max(T - 1, 1)
            s_target_mse.append(float(np.mean((pred - tgt) ** 2)))
        f_target_mse = []
        for ep in failure_eps:
            pred = data["failure"][ep]
            T = pred.shape[0]
            tgt = np.arange(T, dtype=np.float32) / max(T - 1, 1)
            f_target_mse.append(float(np.mean((pred - tgt) ** 2)))

        s_last = [float(data["success"][ep][-1]) for ep in success_eps]
        f_last = [float(data["failure"][ep][-1]) for ep in failure_eps]
        s_max = [float(data["success"][ep].max()) for ep in success_eps]
        f_max = [float(data["failure"][ep].max()) for ep in failure_eps]

        rows.append({
            "model": label,
            "n_success": len(success_eps),
            "n_failure": len(failure_eps),
            "success_mse_to_progress_mean": float(np.mean(s_target_mse)),
            "success_mse_to_progress_median": float(np.median(s_target_mse)),
            "failure_mse_to_progress_mean": float(np.mean(f_target_mse)),
            "failure_mse_to_progress_median": float(np.median(f_target_mse)),
            "success_last_frame_mean": float(np.mean(s_last)),
            "failure_last_frame_mean": float(np.mean(f_last)),
            "success_max_mean": float(np.mean(s_max)),
            "failure_max_mean": float(np.mean(f_max)),
            "success_last_minus_failure_last": float(np.mean(s_last) - np.mean(f_last)),
        })

    fields = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"  saved {out_path}")

    print()
    print("Per-model summary (computed on common episode set):")
    header = ("model", "S last", "F last", "S-F last", "S MSE", "F MSE")
    print("  {:<28} {:>7} {:>7} {:>9} {:>7} {:>7}".format(*header))
    for r in rows:
        print("  {:<28} {:>7.3f} {:>7.3f} {:>9.3f} {:>7.4f} {:>7.4f}".format(
            r["model"],
            r["success_last_frame_mean"],
            r["failure_last_frame_mean"],
            r["success_last_minus_failure_last"],
            r["success_mse_to_progress_mean"],
            r["failure_mse_to_progress_mean"],
        ))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_specs = _parse_models(args.model)
    print(f"[1/3] Loading {len(model_specs)} model evals")
    per_model: list[tuple[str, dict]] = []
    for label, eval_dir in model_specs:
        if not (eval_dir / "trajectories.npz").exists():
            raise FileNotFoundError(
                f"{label}: missing {eval_dir / 'trajectories.npz'}. "
                f"Run eval_val_loss_and_visualize for this checkpoint first."
            )
        data = _load_one(eval_dir)
        print(f"  {label:<32} success={len(data['success'])}  "
              f"failure={len(data['failure'])}  from {eval_dir}")
        per_model.append((label, data))

    success_eps = _intersect_episodes([d for _, d in per_model], "success")
    failure_eps = _intersect_episodes([d for _, d in per_model], "failure")
    print(f"  Common success episodes: {len(success_eps)} "
          f"(range {success_eps[0]}..{success_eps[-1]} if any)" if success_eps else
          "  Common success episodes: 0")
    print(f"  Common failure episodes: {len(failure_eps)} "
          f"(range {failure_eps[0]}..{failure_eps[-1]} if any)" if failure_eps else
          "  Common failure episodes: 0")

    print(f"[2/3] Plotting overlay grids (first {args.num_visualize} episodes)")
    _plot_overlay_grid("success", per_model, success_eps, args.num_visualize,
                       output_dir / "success_overlay.png")
    _plot_overlay_grid("failure", per_model, failure_eps, args.num_visualize,
                       output_dir / "failure_overlay.png")

    print("[3/3] Mean-curve panel + summary CSV")
    _plot_mean_curves(per_model, success_eps, failure_eps,
                      output_dir / "mean_curve.png")
    _write_summary_csv(per_model, success_eps, failure_eps,
                       output_dir / "summary.csv")
    print("Done.")


if __name__ == "__main__":
    main()
