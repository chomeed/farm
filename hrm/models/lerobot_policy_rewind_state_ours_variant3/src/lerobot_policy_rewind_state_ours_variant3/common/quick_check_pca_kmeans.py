"""
Quick visual sanity check for the PCA + K-means failure-onset detector.

What it does (no reward model load, no GPU needed):
  1. Loads `observation.state` per frame for all success and failure episodes
     in the robomimic_square dataset.
  2. Fits the PCA+KMeans detector on success-episode states.
  3. Scores every failure episode and finds the PCA first-cross critical_t.
  4. If --eval_dir is given, also loads cached predicted rewards from that
     dir's `trajectories.npz` and computes the smooth+argmax critical_t for
     comparison.
  5. Renders a 2xN grid: top row = predicted reward + smooth-argmax critical_t,
     bottom row = PCA OOD score + threshold + first-cross critical_t.

The bottom row tells you whether the detector "actually works":
  - Curve should sit near zero through the early/mid frames (in-distribution)
    and ramp up near the failure moment.
  - The horizontal threshold line is the (1-alpha) quantile from success
    frames; the red vertical line is the chosen critical_t (first cross,
    with argmax fallback).

Example:
  python -m lerobot_policy_rewind_state_ours_variant3.common.quick_check_pca_kmeans \\
      --success_repo_id chomeed/robomimic_square_resized_peg_success \\
      --failure_repo_id chomeed/robomimic_square_resized_peg_failure \\
      --num_episodes 200 \\
      --num_visualize 8 \\
      --alpha 0.05 \\
      --eval_dir /home/chomeed/workspace/insertion_generation/ckpts/rewind/rewind_state_ours_variant3_robomimic_square_v1_eps200_clip100_longer/eval_val_last_hybrid \\
      --output_dir /tmp/pca_kmeans_check
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

from lerobot_policy_rewind_state.common.rewind_datasets import EpisodeDataset
from ._failure_detector import fit_pca_kmeans_detector
from ._match_helpers import critical_timestep, smooth


CAMERA_KEYS_DEFAULT = [
    "observation.images.agentview",
    "observation.images.eye_in_hand",
]
# DINOv2 ImageNet normalization stats (same as modeling_rewind_state._DINO_MEAN/_STD).
_DINO_MEAN = (0.485, 0.456, 0.406)
_DINO_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--success_repo_id", default="chomeed/robomimic_square_resized_peg_success")
    p.add_argument("--failure_repo_id", default="chomeed/robomimic_square_resized_peg_failure")
    p.add_argument("--num_episodes", type=int, default=200,
                   help="how many episodes per source to fit / score on")
    p.add_argument("--num_visualize", type=int, default=8,
                   help="how many failure episodes to plot")
    p.add_argument("--n_components", type=int, default=4,
                   help="raise to ~16/32 when --obs_mode=state_image since "
                        "the DINOv2 embedding adds hundreds of dims")
    p.add_argument("--n_clusters", type=int, default=64)
    p.add_argument("--alpha", type=float, default=0.05,
                   help="threshold = (1-alpha) quantile of success scores")
    p.add_argument("--smooth_window", type=int, default=7,
                   help="for smooth+argmax baseline on cached rewards")
    p.add_argument(
        "--eval_dir", type=str, default=None,
        help="optional: path to an eval_val_last*/ dir with trajectories.npz; "
             "if set, overlays cached predicted reward + smooth+argmax baseline",
    )
    p.add_argument("--obs_mode", choices=["state", "state_image"], default="state_image",
                   help="state-only PCA, or augment with per-frame finetuned-DINOv2 "
                        "image embeddings from --base_ckpt (one per camera, "
                        "CLS-attention pooled, then concatenated to the state)")
    p.add_argument("--base_ckpt", type=str,
                   default="/home/chomeed/workspace/insertion_generation/ckpts/rewind/"
                           "rewind_state_robomimic_square_v1_pro6000/checkpoints/"
                           "010000/pretrained_model",
                   help="Only used when --obs_mode=state_image. Loads the "
                        "policy's vision_encoder (finetuned DINOv2) for image "
                        "embedding. Defaults to the pro6000 baseline ckpt.")
    p.add_argument("--camera_keys", nargs="+", default=CAMERA_KEYS_DEFAULT)
    p.add_argument("--encode_chunk_size", type=int, default=64,
                   help="frames per vision_encoder forward (bounds GPU memory)")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


@torch.no_grad()
def _encode_episode_images(
    policy,
    cam_frames: dict[str, torch.Tensor],
    device: str,
    chunk_size: int,
) -> np.ndarray:
    """Per-frame DINOv2 image embedding, concatenated across cameras.

    cam_frames[cam] = (T, C, H, W) in [0, 1] from the dataset. Returns a
    (T, n_cams * D_vis) float32 array. Uses the policy's vision_encoder
    (finetuned DINOv2 from the supplied base ckpt) with CLS-attention
    weighted patch pooling, matching `RewindStatePolicy._encode_images`.
    """
    cam_keys = list(cam_frames.keys())
    T = cam_frames[cam_keys[0]].shape[0]
    mean = torch.tensor(_DINO_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_DINO_STD, device=device).view(1, 3, 1, 1)

    per_cam_embs = []
    for cam in cam_keys:
        imgs = cam_frames[cam].to(device, non_blocking=True)   # (T, C, H, W)
        chunks = []
        for start in range(0, T, chunk_size):
            chunk = imgs[start : start + chunk_size]
            chunk = (chunk - mean) / std
            outputs = policy.vision_encoder(
                pixel_values=chunk, output_attentions=True
            )
            patch_tokens = outputs.last_hidden_state[:, 1:]
            attn = outputs.attentions[-1][:, :, 0, 1:].mean(dim=1)
            attn = attn / attn.sum(dim=-1, keepdim=True)
            emb = (patch_tokens * attn.unsqueeze(-1)).sum(dim=1)   # (chunk, D)
            chunks.append(emb.detach().to("cpu", dtype=torch.float32))
        per_cam_embs.append(torch.cat(chunks, dim=0))              # (T, D)
    return torch.cat(per_cam_embs, dim=-1).numpy()                 # (T, n_cams * D)


def _collect_observations(
    repo_id: str,
    num_episodes: int,
    obs_mode: str,
    camera_keys: list[str],
    policy=None,
    device: str = "cpu",
    encode_chunk_size: int = 64,
) -> dict[int, np.ndarray]:
    """Per-episode observation arrays. Each value is (T, feature_dim) with
    feature_dim = state_dim (state mode) or state_dim + n_cams * D_vis
    (state_image mode)."""
    ds = EpisodeDataset(
        repo_id=repo_id,
        episodes=list(range(num_episodes)),
        camera_keys=camera_keys if obs_mode == "state_image" else [],
        text_embedding_key=None,
        state_key="observation.state",
        use_precomputed_embeddings_only=False,
        pad_to_length=None,
    )
    out: dict[int, np.ndarray] = {}
    desc = f"obs[{obs_mode}] {repo_id.split('/')[-1]}"
    for i in tqdm(range(len(ds)), desc=desc):
        item = ds[i]
        ep_idx = int(item["episode_index"])
        state = item["observation.state"]
        if hasattr(state, "detach"):
            state = state.detach().cpu().numpy()
        state = np.asarray(state, dtype=np.float32)
        if obs_mode == "state":
            out[ep_idx] = state
            continue
        # state_image: concat per-frame DINOv2 embeddings.
        cam_frames = {cam: item[cam] for cam in camera_keys}
        img_emb = _encode_episode_images(
            policy, cam_frames, device=device, chunk_size=encode_chunk_size,
        )                                                          # (T, n_cams * D)
        T_state, T_img = state.shape[0], img_emb.shape[0]
        if T_state != T_img:
            raise RuntimeError(
                f"ep {ep_idx}: state has T={T_state} but image embeddings "
                f"have T={T_img}; check the dataset"
            )
        out[ep_idx] = np.concatenate([state, img_emb], axis=1)
    return out


def _load_cached_rewards(eval_dir: Path, split: str) -> dict[int, np.ndarray]:
    npz_path = eval_dir / "trajectories.npz"
    if not npz_path.exists():
        return {}
    npz = np.load(npz_path)
    out: dict[int, np.ndarray] = {}
    for key in npz.files:
        s, ep_str, _ = key.split("/")
        if s != split:
            continue
        out[int(ep_str)] = np.asarray(npz[key], dtype=np.float32)
    return out


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = None
    if args.obs_mode == "state_image":
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_policy
        print(f"[0/4] Loading vision encoder from {args.base_ckpt}")
        policy_cfg = PreTrainedConfig.from_pretrained(args.base_ckpt)
        policy_cfg.pretrained_path = args.base_ckpt
        policy_cfg.device = args.device
        # Need a ds_meta to construct the policy; the success dataset's meta
        # is the natural choice (matches training-time stats).
        bootstrap = EpisodeDataset(
            repo_id=args.success_repo_id,
            episodes=[0],
            camera_keys=args.camera_keys,
            text_embedding_key=None,
            state_key="observation.state",
            use_precomputed_embeddings_only=False,
            pad_to_length=None,
        )
        policy = make_policy(cfg=policy_cfg, ds_meta=bootstrap.meta)
        policy.eval()
        policy.to(args.device)
        # We don't need the full reward head, only the vision_encoder.
        for p_ in policy.parameters():
            p_.requires_grad_(False)

    print(f"[1/4] Loading observations (success+failure, n={args.num_episodes} each, "
          f"obs_mode={args.obs_mode})")
    success_obs = _collect_observations(
        args.success_repo_id, args.num_episodes, args.obs_mode, args.camera_keys,
        policy=policy, device=args.device, encode_chunk_size=args.encode_chunk_size,
    )
    failure_obs = _collect_observations(
        args.failure_repo_id, args.num_episodes, args.obs_mode, args.camera_keys,
        policy=policy, device=args.device, encode_chunk_size=args.encode_chunk_size,
    )
    print(f"  success episodes={len(success_obs)}  "
          f"failure episodes={len(failure_obs)}  "
          f"feature_dim={next(iter(success_obs.values())).shape[1]}")

    print(f"[2/4] Fitting PCA(n={args.n_components}) + KMeans(k={args.n_clusters}) on success obs, "
          f"alpha={args.alpha}")
    det = fit_pca_kmeans_detector(
        list(success_obs.values()),
        n_components=args.n_components,
        n_clusters=args.n_clusters,
        alpha=args.alpha,
    )
    print(f"  threshold={det.threshold:.4f}  "
          f"explained_var={[f'{v:.3f}' for v in det.pca.explained_variance_ratio_]}")

    cached_rewards: dict[int, np.ndarray] = {}
    if args.eval_dir:
        eval_dir = Path(args.eval_dir)
        cached_rewards = _load_cached_rewards(eval_dir, split="failure")
        print(f"[3/4] Loaded {len(cached_rewards)} cached failure rewards from {eval_dir}")
    else:
        print("[3/4] No --eval_dir: skipping cached-reward overlay")

    # Score every failure episode + summarize.
    print("[4/4] Scoring failure episodes")
    per_ep: list[dict] = []
    for ep_idx in sorted(failure_obs.keys()):
        obs = failure_obs[ep_idx]
        scores = det.score(obs)
        above = np.where(scores > det.threshold)[0]
        crit_pca = int(above[0]) if above.size else int(np.argmax(scores))
        smooth_argmax_t = None
        if ep_idx in cached_rewards:
            r = cached_rewards[ep_idx]
            smooth_argmax_t = int(critical_timestep(r, args.smooth_window))
        per_ep.append({
            "ep": ep_idx,
            "T": int(obs.shape[0]),
            "scores": scores,
            "crit_pca": crit_pca,
            "first_cross": bool(above.size > 0),
            "smooth_argmax_t": smooth_argmax_t,
            "reward": cached_rewards.get(ep_idx),
        })

    # Aggregate stats.
    cross_frac = float(np.mean([float(e["first_cross"]) for e in per_ep]))
    pcas = [e["crit_pca"] for e in per_ep]
    print(f"\nSummary across {len(per_ep)} failure episodes:")
    print(f"  fraction with PCA first-cross (not argmax fallback): {cross_frac:.2%}")
    print(f"  PCA critical_t distribution: "
          f"min={min(pcas)}  median={int(np.median(pcas))}  max={max(pcas)}")
    if cached_rewards:
        eps_with_baseline = [e for e in per_ep if e["smooth_argmax_t"] is not None]
        diffs = [e["crit_pca"] - e["smooth_argmax_t"] for e in eps_with_baseline]
        print(f"  smooth+argmax baseline available for {len(eps_with_baseline)} episodes")
        if diffs:
            print(f"  PCA - smooth_argmax (frames): "
                  f"mean={np.mean(diffs):+.1f}  "
                  f"median={int(np.median(diffs)):+d}  "
                  f"std={np.std(diffs):.1f}")

    def _draw_score_panel(ax, entry, *, with_legend=True, ylabel=True):
        axb = ax
        axb.plot(entry["scores"], color="tab:purple", linewidth=1.2,
                 label="PCA OOD score")
        axb.axhline(det.threshold, color="black", linestyle="--", linewidth=0.8,
                    alpha=0.6, label=f"threshold (alpha={args.alpha})")
        axb.axvline(entry["crit_pca"], color="tab:red", linewidth=1.4,
                    label=f"PCA crit_t={entry['crit_pca']}"
                    + ("" if entry["first_cross"] else " (argmax fallback)"))
        axb.set_xlim(0, max(entry["T"] - 1, 1))
        axb.set_title(f"failure ep {entry['ep']} (T={entry['T']})", fontsize=9)
        axb.set_xlabel("frame")
        if ylabel:
            axb.set_ylabel("OOD score")
        axb.grid(True, alpha=0.3)
        if with_legend:
            axb.legend(fontsize=7, loc="upper left")

    def _draw_reward_panel(ax, entry, *, with_legend=True, ylabel=True):
        axt = ax
        if entry["reward"] is not None:
            r = entry["reward"]
            axt.plot(r, color="tab:blue", linewidth=1.2, label="rewind reward")
            axt.plot(smooth(r, args.smooth_window), color="tab:orange",
                     linewidth=1.0, linestyle="--", alpha=0.7,
                     label=f"smooth(w={args.smooth_window})")
            axt.axvline(entry["smooth_argmax_t"], color="tab:green",
                        linewidth=1.4,
                        label=f"smooth+argmax crit_t={entry['smooth_argmax_t']}")
            axt.axvline(entry["crit_pca"], color="tab:red",
                        linewidth=1.0, alpha=0.7, linestyle=":",
                        label=f"PCA crit_t={entry['crit_pca']}")
            axt.set_ylim(-0.05, 1.05)
        else:
            axt.text(0.5, 0.5, "no cached reward\n(pass --eval_dir)",
                     ha="center", va="center", transform=axt.transAxes,
                     fontsize=8, color="gray")
        axt.set_xlim(0, max(entry["T"] - 1, 1))
        axt.set_title(f"failure ep {entry['ep']} -- reward view", fontsize=9)
        axt.set_xlabel("frame")
        if ylabel:
            axt.set_ylabel("pred reward")
        axt.grid(True, alpha=0.3)
        if with_legend and entry["reward"] is not None:
            axt.legend(fontsize=7, loc="lower right")

    feat_dim = next(iter(success_obs.values())).shape[1]
    suptitle_suffix = (
        f"(obs_mode={args.obs_mode}, feat_dim={feat_dim}, "
        f"n_components={args.n_components}, k={args.n_clusters}, "
        f"alpha={args.alpha})"
    )

    # Summary grid of the first --num_visualize failure episodes.
    eps_to_plot = sorted(failure_obs.keys())[: args.num_visualize]
    n = len(eps_to_plot)
    fig, axes = plt.subplots(
        2, n, figsize=(3.4 * n, 5.6), sharex=False, squeeze=False,
    )
    for col, ep_idx in enumerate(eps_to_plot):
        entry = next(e for e in per_ep if e["ep"] == ep_idx)
        _draw_score_panel(axes[1, col], entry,
                          with_legend=(col == 0), ylabel=(col == 0))
        _draw_reward_panel(axes[0, col], entry,
                           with_legend=(col == 0), ylabel=(col == 0))
    fig.suptitle(
        f"PCA+KMeans failure-onset detector -- first {n} failure episodes  "
        + suptitle_suffix,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = output_dir / "pca_kmeans_check.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\nSaved {out_path}")

    # Per-episode PNGs: one figure per failure episode.
    per_ep_dir = output_dir / "per_episode"
    per_ep_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(per_ep)} per-episode PNGs to {per_ep_dir}")
    for entry in tqdm(per_ep, desc="per-episode plots"):
        ep_idx = entry["ep"]
        fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=False)
        _draw_reward_panel(axes[0], entry, with_legend=True, ylabel=True)
        _draw_score_panel(axes[1], entry, with_legend=True, ylabel=True)
        fig.suptitle(
            f"failure ep {ep_idx}  " + suptitle_suffix,
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(per_ep_dir / f"failure_{ep_idx:05d}.png", dpi=110)
        plt.close(fig)

    # Tiny JSON summary.
    summary = {
        "detector": det.to_dict(),
        "num_failure_episodes": len(per_ep),
        "first_cross_fraction": cross_frac,
        "per_episode": [
            {
                "ep": e["ep"], "T": e["T"], "crit_pca": e["crit_pca"],
                "first_cross": e["first_cross"],
                "smooth_argmax_t": e["smooth_argmax_t"],
            }
            for e in per_ep
        ],
    }
    (output_dir / "pca_kmeans_check.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved {output_dir / 'pca_kmeans_check.json'}")


if __name__ == "__main__":
    main()
