"""
Evaluate a trained `rewind_state_ours_variant8` checkpoint:

  1. Compute per-frame reward MSE on the LAST N success episodes (the
     "validation slice"); report mean and per-episode breakdown.
  2. Compute the per-frame predicted reward on each of those episodes plus
     the LAST N failure episodes.
  3. Plot M success + M failure episodes side-by-side as PNGs showing
     predicted reward vs frame-progress target.

Usage:
    python -m lerobot_policy_rewind_state_ours_variant8.common.eval_val_loss_and_visualize \
        --ckpt /scratch2/chomeed/outputs/rewind/rewind_state_ours_robomimic_square_v1/checkpoints/last/pretrained_model \
        --num_val_episodes 50 --num_visualize 5 \
        --output_dir /scratch2/chomeed/outputs/rewind/rewind_state_ours_robomimic_square_v1/eval_val
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
    MAX_WINDOW,
)

CAMERA_KEYS = [
    "observation.images.agentview",
    "observation.images.eye_in_hand",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="path to .../checkpoints/<step>/pretrained_model")
    p.add_argument("--success_repo_id", default="chomeed/robomimic_square_resized_peg_success")
    p.add_argument("--failure_repo_id", default="chomeed/robomimic_square_resized_peg_failure")
    p.add_argument("--num_val_episodes", type=int, default=50,
                   help="how many episodes from the TAIL of each dataset")
    p.add_argument("--num_visualize", type=int, default=5,
                   help="how many demos per source to plot (taken from the val slice)")
    p.add_argument("--camera_keys", type=str, nargs="+", default=CAMERA_KEYS,
                   help="Camera observation keys to load and embed. Defaults "
                        "to the 2 robomimic-square cameras. Pass the dexmg "
                        "3-camera keys for dexmg tasks.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--sampling", choices=["uniform", "recent", "hybrid"], default="uniform",
        help="Window-index sampling for dense prediction. 'hybrid' uses 8 uniform + 8 latest "
             "(controlled by --n_recent).",
    )
    p.add_argument(
        "--n_recent", type=int, default=8,
        help="When --sampling=hybrid, number of most-recent frames in the 16-frame window. "
             "Ignored otherwise.",
    )
    return p.parse_args()


def _move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


def _build_full_episode_batch(item: dict, device: str, camera_keys: list[str]) -> dict:
    """Pack an EpisodeDataset item into a (1, T, ...) batch dict."""
    batch = {
        **{cam: item[cam].unsqueeze(0) for cam in camera_keys},
        "observation.state": item["observation.state"].unsqueeze(0),
        "task": [item["task"]],
        "episode_index": torch.tensor([item["episode_index"]]),
    }
    return _move_batch_to_device(batch, device)


@torch.no_grad()
def _dense_reward(
    policy,
    preprocessor,
    item: dict,
    device: str,
    camera_keys: list[str],
    sampling: str = "uniform",
    n_recent: int = 8,
) -> np.ndarray:
    """Per-frame predicted reward in [0, 1], shape (T,)."""
    batch = _build_full_episode_batch(item, device, camera_keys)
    batch = preprocessor(batch)
    raw_images = {k: batch[k] for k in camera_keys}
    pred = dense_predict_per_frame(
        reward_model=policy,
        raw_images=raw_images,
        lang_emb=batch["task.minilm"],
        state=batch.get("observation.state"),
        max_window=MAX_WINDOW,
        sampling=sampling,
        n_recent=n_recent,
    )
    return pred.detach().float().cpu().numpy().astype(np.float32)


def _val_loss(pred: np.ndarray) -> float:
    """Per-frame MSE between dense predicted reward and the synthetic
    frame-progress target used by RewindDataset for the forward branch.
    """
    T = pred.shape[0]
    target = np.arange(T, dtype=np.float32) / max(T - 1, 1)
    return float(np.mean((pred - target) ** 2))


def _process_source(
    label: str,
    repo_id: str,
    num_episodes: int,
    policy,
    preprocessor,
    args,
):
    print(f"\n[{label}] Discovering total episodes in {repo_id}")
    meta_only = EpisodeDataset(
        repo_id=repo_id,
        episodes=[0],
        camera_keys=args.camera_keys,
        text_embedding_key=None,
        state_key="observation.state",
        use_precomputed_embeddings_only=False,
        pad_to_length=None,
    )
    total = int(meta_only.meta.total_episodes)
    start = max(0, total - num_episodes)
    val_eps = list(range(start, total))
    print(f"[{label}] Using episodes {val_eps[0]}..{val_eps[-1]} (n={len(val_eps)} of {total})")

    dataset = EpisodeDataset(
        repo_id=repo_id,
        episodes=val_eps,
        camera_keys=args.camera_keys,
        text_embedding_key=None,
        state_key="observation.state",
        use_precomputed_embeddings_only=False,
        pad_to_length=None,
    )

    out: dict[int, dict] = {}
    for i in tqdm(range(len(dataset)), desc=label):
        item = dataset[i]
        ep_idx = int(item["episode_index"])
        T = int(item[args.camera_keys[0]].shape[0])

        pred = _dense_reward(
            policy, preprocessor, item, args.device,
            camera_keys=args.camera_keys,
            sampling=args.sampling, n_recent=args.n_recent,
        )

        out[ep_idx] = {
            "T": T,
            "predicted_reward": pred,
            "val_mse": _val_loss(pred),
        }
    return out


def _plot_demos(
    label: str,
    per_episode: dict[int, dict],
    num_visualize: int,
    output_dir: Path,
):
    ep_ids = sorted(per_episode.keys())[:num_visualize]
    n = len(ep_ids)
    if n == 0:
        return None

    fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n), squeeze=False)
    for row, ep_idx in enumerate(ep_ids):
        d = per_episode[ep_idx]
        pred = d["predicted_reward"]
        T = d["T"]

        ax0 = axes[row, 0]
        ax0.plot(pred, color="tab:blue", label="pred reward")
        target = np.arange(T, dtype=np.float32) / max(T - 1, 1)
        ax0.plot(target, color="tab:gray", linestyle="--", alpha=0.5,
                 label="frame-progress target")
        ax0.set_title(f"{label} ep {ep_idx} - predicted reward "
                      f"(MSE={d['val_mse']:.4f})")
        ax0.set_xlabel("frame")
        ax0.set_ylabel("reward in [0,1]")
        ax0.set_ylim(-0.05, 1.05)
        ax0.grid(True, alpha=0.3)
        ax0.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out_path = output_dir / f"{label}_reward.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved {out_path}")
    return out_path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading policy from {args.ckpt}")
    policy_cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    policy_cfg.pretrained_path = args.ckpt
    policy_cfg.device = args.device

    # Need a ds_meta to call make_policy; the success dataset is the natural choice
    # (its stats were used at training time).
    bootstrap_ds = EpisodeDataset(
        repo_id=args.success_repo_id,
        episodes=[0],
        camera_keys=args.camera_keys,
        text_embedding_key=None,
        state_key="observation.state",
        use_precomputed_embeddings_only=False,
        pad_to_length=None,
    )
    ds_meta = bootstrap_ds.meta

    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    policy.eval()
    policy.to(args.device)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=None,
        dataset_stats=ds_meta.stats if hasattr(ds_meta, "stats") else None,
    )
    print(f"      Policy on {args.device}; d_model={policy.config.d_model}, "
          f"num_cameras={policy.config.num_cameras}")

    print(f"[2/3] Running eval over success+failure tails (n={args.num_val_episodes} each)")
    success_per_ep = _process_source(
        "success", args.success_repo_id, args.num_val_episodes,
        policy, preprocessor, args,
    )
    failure_per_ep = _process_source(
        "failure", args.failure_repo_id, args.num_val_episodes,
        policy, preprocessor, args,
    )

    success_mses = [d["val_mse"] for d in success_per_ep.values()]
    print(f"\nValidation reward MSE on last {len(success_per_ep)} success episodes:")
    print(f"  mean={np.mean(success_mses):.5f}  "
          f"median={np.median(success_mses):.5f}  "
          f"min={np.min(success_mses):.5f}  "
          f"max={np.max(success_mses):.5f}")

    # Failure-side reward "MSE" is reported for reference but it's not a real
    # training target — failure episodes aren't supervised by the synthetic
    # frame-progress target.
    failure_mses = [d["val_mse"] for d in failure_per_ep.values()]
    print(f"\nReference reward MSE on last {len(failure_per_ep)} failure episodes "
          f"(NOT a training target; failure clips were not supervised):")
    print(f"  mean={np.mean(failure_mses):.5f}  "
          f"median={np.median(failure_mses):.5f}")

    print(f"\n[3/3] Plotting {args.num_visualize} demos per source")
    _plot_demos("success", success_per_ep, args.num_visualize, output_dir)
    _plot_demos("failure", failure_per_ep, args.num_visualize, output_dir)

    print(f"\nWriting summary -> {output_dir / 'summary.json'}")
    summary = {
        "ckpt": str(args.ckpt),
        "success_repo_id": args.success_repo_id,
        "failure_repo_id": args.failure_repo_id,
        "num_val_episodes": int(args.num_val_episodes),
        "sampling": args.sampling,
        "n_recent": int(args.n_recent),
        "val_mse_success": {
            "mean": float(np.mean(success_mses)),
            "median": float(np.median(success_mses)),
            "min": float(np.min(success_mses)),
            "max": float(np.max(success_mses)),
            "per_episode": {int(ep): float(d["val_mse"])
                            for ep, d in success_per_ep.items()},
        },
        "val_mse_failure_reference": {
            "mean": float(np.mean(failure_mses)),
            "median": float(np.median(failure_mses)),
            "per_episode": {int(ep): float(d["val_mse"])
                            for ep, d in failure_per_ep.items()},
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Writing trajectories -> {output_dir / 'trajectories.npz'}")
    np.savez(
        output_dir / "trajectories.npz",
        **{
            f"success/{ep}/predicted_reward": d["predicted_reward"]
            for ep, d in success_per_ep.items()
        },
        **{
            f"failure/{ep}/predicted_reward": d["predicted_reward"]
            for ep, d in failure_per_ep.items()
        },
    )
    print("Done.")


if __name__ == "__main__":
    main()
