"""
Per-frame ReWiND rewards refined by an InsertionPrior success classifier.

For each frame t in an episode:
    r_rewind(t)      = dense per-frame reward from ReWiND
    p_insertion(t)   = P(success | prefix frames [0..t])      — computed by
                       feeding a uniformly-subsampled 16-frame window of the
                       prefix through the InsertionPrior model
    r_refined(t, w)  = r_rewind(t) - (1 - p_insertion(t)) * w

Multiple weights w are swept so we can visually tune the penalty strength.

CLI example (note: all args use `=`, not spaces):
    python -m lerobot_policy_rewind.common.eval_rewind_refined_by_insertion_prior \
    --policy.path=/home/chomeed/workspace/proj-rl-finetuning/hrm/models/lerobot_policy_rewind/src/lerobot_policy_rewind/outputs/train/2026-04-20/13-43-45_rewind_peg_insertion/checkpoints/last/pretrained_model \
    --insertion_prior_path=/home/chomeed/workspace/proj-rl-finetuning/outputs/train/2026-04-15/01-33-41_insertion_prior/checkpoints/last/pretrained_model \
    --dataset.repo_id=chomeed/peg_insertion_success_1000_for_rewind_val \
    --dataset.video_backend=torchcodec \
    --num_workers=4
"""
from dataclasses import dataclass, field
from pathlib import Path

import accelerate
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors

from lerobot_policy_rewind_state.common.eval_rewind_reward_visualization_dense import (
    _extract_episode_index,
    _move_batch_to_device,
    dense_predict_per_frame,
)
from lerobot_policy_rewind_state.common.factory import make_eval_original_dataset


MAX_WINDOW = 16


@dataclass
class RewindRefinedConfig(TrainPipelineConfig):
    insertion_prior_path: str | None = None
    weights: list[float] = field(default_factory=lambda: [0.5])
    render_videos: bool = True
    video_mix_weight: float = 0.5
    video_fps: int = 5
    video_save_dir: str = "videos_refined"
    video_delta: int = 2


# ── Per-frame insertion prior ────────────────────────────────────────────────

@torch.no_grad()
def dense_predict_insertion_per_frame(
    insertion_model,
    raw_images: dict[str, torch.Tensor],
    lang_emb: torch.Tensor,
    max_window: int = MAX_WINDOW,
    chunk_size: int = 8,
) -> torch.Tensor:
    """
    p(success | prefix[0..t]) for every t in [0, N).

    For each target t we build a 16-frame window by uniformly sampling indices
    from [0, t] (via linspace, with repeats for t < max_window). This matches
    the subsampling distribution InsertionPrior was trained on (16-frame clips
    subsampled from 8-30-frame trajectories).

    Args:
        raw_images: {camera_key: (1, N, C, H, W)}
        lang_emb:   (1, D_text)

    Returns:
        (N,) tensor of probabilities in [0, 1].
    """
    cam_keys = list(raw_images.keys())
    first = raw_images[cam_keys[0]]
    if first.ndim != 5 or first.shape[0] != 1:
        raise ValueError(
            f"Expected (1, N, C, H, W) per camera, got {tuple(first.shape)}"
        )
    _, N, _, _, _ = first.shape
    device = first.device

    # Build (N, max_window) index matrix: row t = uniform subsample of [0..t].
    idx_matrix = np.stack(
        [np.linspace(0, t, max_window).astype(np.int64) for t in range(N)],
        axis=0,
    )
    idx_tensor = torch.as_tensor(idx_matrix, device=device, dtype=torch.long)

    scores = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        idx_chunk = idx_tensor[start:end]       # (m, max_window)
        m = idx_chunk.shape[0]

        batch = {
            "task.minilm": lang_emb.expand(m, -1).contiguous(),
            "lengths": torch.full((m,), max_window, dtype=torch.long, device=device),
        }
        for cam in cam_keys:
            base = raw_images[cam][0]           # (N, C, H, W)
            batch[cam] = base[idx_chunk]        # (m, max_window, C, H, W)

        s = insertion_model.predict_reward(batch)  # (m,) in [0, 1]
        scores.append(s)

    return torch.cat(scores, dim=0)


# ── Video rendering ──────────────────────────────────────────────────────────

def _render_episode_video(
    raw_images: dict[str, torch.Tensor],
    r_rewind: np.ndarray,
    p_insertion: np.ndarray,
    mix_weight: float,
    save_path: str,
    fps: int = 10,
    max_alpha: float = 0.45,
    delta_scale: float = 8.0,
    delta_step: int = 2,
) -> None:
    """
    Write an mp4 of the episode with a red/green frame tint whose opacity tracks
    the Δ-step delta of the per-episode min-max-normalized refined reward
    (r_rewind - (1 - p_insertion) * mix_weight).

    delta[t] = r_norm[t + delta_step] - r_norm[t]   (for t >= N - delta_step, delta = 0)

    - increasing reward → green overlay
    - decreasing reward → red overlay
    - alpha = min(max_alpha, |delta| * delta_scale)   (kept subtle by default)
    """
    r_mix = r_rewind - (1.0 - p_insertion) * mix_weight
    lo, hi = float(r_mix.min()), float(r_mix.max())
    scale = hi - lo if hi - lo > 1e-9 else 1.0
    r_norm = (r_mix - lo) / scale
    delta = np.zeros_like(r_norm)
    if delta_step < len(r_norm):
        delta[:-delta_step] = r_norm[delta_step:] - r_norm[:-delta_step]

    cam_keys = list(raw_images.keys())
    T = raw_images[cam_keys[0]].shape[1]

    def _frame_rgb(cam: str, t: int) -> np.ndarray:
        img = raw_images[cam][0, t].detach().cpu().numpy()  # (C, H, W) in [0, 1]
        img = np.transpose(img, (1, 2, 0))
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)

    # Figure out composite size from the first frame
    first = [_frame_rgb(c, 0) for c in cam_keys]
    target_h = max(f.shape[0] for f in first)

    def _resize_h(img: np.ndarray, h: int) -> np.ndarray:
        if img.shape[0] == h:
            return img
        new_w = int(img.shape[1] * h / img.shape[0])
        return cv2.resize(img, (new_w, h))

    composite0 = np.concatenate([_resize_h(f, target_h) for f in first], axis=1)
    H, W = composite0.shape[:2]

    writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
    )
    if not writer.isOpened():
        print(f"[warn] could not open {save_path} for writing; skipping video")
        return

    try:
        for t in range(T):
            frames = [_resize_h(_frame_rgb(c, t), target_h) for c in cam_keys]
            composite = np.concatenate(frames, axis=1)

            d = float(delta[t])
            alpha = float(min(max_alpha, abs(d) * delta_scale))
            tint = np.array([0, 255, 0], dtype=np.float32) if d >= 0 else np.array([255, 0, 0], dtype=np.float32)
            mixed = composite.astype(np.float32) * (1 - alpha) + tint[None, None, :] * alpha
            mixed = np.clip(mixed, 0, 255).astype(np.uint8)

            bgr = cv2.cvtColor(mixed, cv2.COLOR_RGB2BGR)
            cv2.putText(
                bgr,
                f"t={t}  r_norm={r_norm[t]:.3f}  d={d:+.3f}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(bgr)
    finally:
        writer.release()


# ── Collection ───────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_refined_trajectories(
    dataloader: DataLoader,
    reward_model,
    insertion_model,
    camera_keys: list[str],
    preprocessor,
    accelerator: accelerate.Accelerator | None = None,
    sampling: str = "uniform",
    max_episodes: int | None = None,
    video_save_dir: str | Path | None = None,
    video_mix_weight: float = 0.5,
    video_fps: int = 10,
    video_delta: int = 2,
):
    """
    For each episode, compute ReWiND per-frame reward AND InsertionPrior
    per-frame success probability, using the *same* preprocessed raw images.
    """
    episode_to_pred_rewards = {}
    episode_to_p_insertion = {}
    episode_to_gt_reward = {}

    device = accelerator.device if accelerator is not None else next(reward_model.parameters()).device

    if video_save_dir is not None:
        video_save_dir = Path(video_save_dir)
        video_save_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(tqdm(dataloader, desc="Dense reward + insertion prior")):
        if max_episodes is not None and i >= max_episodes:
            break
        batch = _move_batch_to_device(batch, device)
        batch = preprocessor(batch)

        missing = [k for k in camera_keys if k not in batch]
        if missing:
            raise KeyError(
                f"Expected raw images under {missing}; batch has {list(batch.keys())}"
            )
        raw_images = {k: batch[k] for k in camera_keys}

        lang_emb = batch["task.minilm"]
        if lang_emb.ndim == 1:
            lang_emb = lang_emb.unsqueeze(0)

        state = batch.get("observation.state")

        r_dense = dense_predict_per_frame(
            reward_model, raw_images, lang_emb, state=state, sampling=sampling,
        )
        p_ins = dense_predict_insertion_per_frame(
            insertion_model, raw_images, lang_emb
        )

        ep_idx = _extract_episode_index(batch["episode_index"])
        r_np = r_dense.detach().float().cpu().numpy()
        p_np = p_ins.detach().float().cpu().numpy()
        episode_to_pred_rewards[ep_idx] = r_np
        episode_to_p_insertion[ep_idx] = p_np

        if "reward" in batch:
            gt = batch["reward"]
            if isinstance(gt, torch.Tensor):
                if gt.ndim == 2:
                    gt = gt[0]
                episode_to_gt_reward[ep_idx] = gt.detach().float().cpu().numpy()

        if video_save_dir is not None:
            _render_episode_video(
                raw_images=raw_images,
                r_rewind=r_np,
                p_insertion=p_np,
                mix_weight=video_mix_weight,
                save_path=str(video_save_dir / f"episode_{ep_idx:05d}.mp4"),
                fps=video_fps,
                delta_step=video_delta,
            )

    return episode_to_pred_rewards, episode_to_p_insertion, episode_to_gt_reward


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_refined_rewards(
    episode_to_pred_rewards: dict,
    episode_to_p_insertion: dict,
    weights: list[float],
    episode_to_gt_reward: dict | None = None,
    num_episodes_to_plot: int = 10,
    save_path: str = "reward_refined_by_insertion_prior.png",
):
    episode_ids = sorted(episode_to_pred_rewards.keys())[:num_episodes_to_plot]
    if not episode_ids:
        print("No trajectories to plot.")
        return

    n_rows = len(episode_ids)
    fig, axes = plt.subplots(n_rows, 2, figsize=(18, 3 * n_rows), squeeze=False)

    cmap = plt.get_cmap("viridis")
    w_colors = [cmap(i / max(1, len(weights) - 1)) for i in range(len(weights))]

    for row, ep_idx in enumerate(episode_ids):
        pred = np.asarray(episode_to_pred_rewards[ep_idx], dtype=np.float32)
        p_ins = np.asarray(episode_to_p_insertion[ep_idx], dtype=np.float32)
        T = len(pred)
        xs = np.arange(T)

        # Left: raw reward + insertion prior + optional GT
        ax0 = axes[row, 0]
        ax0.plot(xs, pred, label="r_rewind", color="tab:blue", linewidth=1.8)
        ax0.plot(xs, p_ins, label="p_insertion", color="tab:orange", linewidth=1.5, linestyle="--")
        if episode_to_gt_reward is not None and ep_idx in episode_to_gt_reward:
            gt = np.asarray(episode_to_gt_reward[ep_idx], dtype=np.float32)
            ax0.plot(np.arange(len(gt)), gt, label="gt_reward", color="tab:green", alpha=0.7)
        ax0.set_ylim(-0.05, 1.05)
        ax0.set_title(f"Episode {ep_idx} — raw signals")
        ax0.set_xlabel("timestep")
        ax0.set_ylabel("value")
        ax0.grid(True, alpha=0.3)
        ax0.legend(fontsize=8, loc="best")

        # Right: refined rewards at each weight, min-max normalized per episode
        # using a single shared scale across all weights (plus the w=0 curve) so
        # the curves remain comparable within the episode.
        ax1 = axes[row, 1]
        refined_all = [pred] + [pred - (1.0 - p_ins) * w for w in weights]
        stacked = np.stack(refined_all, axis=0)
        lo, hi = float(stacked.min()), float(stacked.max())
        scale = hi - lo if hi - lo > 1e-9 else 1.0
        normed = [(r - lo) / scale for r in refined_all]

        ax1.plot(xs, normed[0], label="r_rewind (w=0)", color="tab:blue", linewidth=1.8, alpha=0.4)
        for r, w, color in zip(normed[1:], weights, w_colors):
            ax1.plot(xs, r, label=f"w={w:g}", color=color, linewidth=1.3)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_title(
            f"Episode {ep_idx} — r_refined (min-max norm, shared scale)\n"
            f"raw range: [{lo:.3f}, {hi:.3f}]"
        )
        ax1.set_xlabel("timestep")
        ax1.set_ylabel("normalized reward")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Saved plot to {save_path}")


# ── Model loading ────────────────────────────────────────────────────────────

def _load_insertion_prior(path: str, ds_meta):
    """Load an InsertionPrior policy from a pretrained checkpoint directory."""
    ip_cfg = PreTrainedConfig.from_pretrained(path)
    ip_cfg.pretrained_path = Path(path)
    return make_policy(cfg=ip_cfg, ds_meta=ds_meta)


# ── Main ─────────────────────────────────────────────────────────────────────

@parser.wrap()
def main_fn(cfg: RewindRefinedConfig):
    cfg.validate()
    if cfg.insertion_prior_path is None:
        raise ValueError("Pass --insertion_prior_path=/path/to/pretrained_model")

    accelerator = accelerate.Accelerator()

    # Dataset — keep raw images so both models can run online DINOv2 encoding.
    dataset = make_eval_original_dataset(cfg)
    dataset.pad_to_length = None
    dataset.use_precomputed_embeddings_only = False
    print(f"Dataset loaded. Num episodes: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers if cfg.num_workers is not None else 4,
        pin_memory=True,
    )

    ds_meta = dataset.meta if hasattr(dataset, "meta") else dataset._datasets[0].meta

    # ReWiND (main reward model, driven by --policy.path)
    reward_model = make_policy(cfg=cfg.policy, ds_meta=ds_meta)

    # InsertionPrior (refiner)
    insertion_model = _load_insertion_prior(cfg.insertion_prior_path, ds_meta)
    insertion_model.eval()

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        dataset_stats=ds_meta.stats if hasattr(ds_meta, "stats") else None,
    )

    print("Models loaded.")

    dataloader, reward_model, insertion_model = accelerator.prepare(
        dataloader, reward_model, insertion_model
    )

    sampling = "uniform"
    video_dir = cfg.video_save_dir if cfg.render_videos else None
    ep2r, ep2p, ep2gt = collect_refined_trajectories(
        dataloader=dataloader,
        reward_model=reward_model,
        insertion_model=insertion_model,
        camera_keys=list(cfg.policy.encoder_camera_keys),
        preprocessor=preprocessor,
        accelerator=accelerator,
        sampling=sampling,
        max_episodes=15,
        video_save_dir=video_dir,
        video_mix_weight=cfg.video_mix_weight,
        video_fps=cfg.video_fps,
        video_delta=cfg.video_delta,
    )

    if accelerator.is_main_process:
        weights = list(cfg.weights)
        print(f"Weights: {weights}")
        weight_tag = "_".join(f"w={w:g}" for w in weights)
        plot_refined_rewards(
            episode_to_pred_rewards=ep2r,
            episode_to_p_insertion=ep2p,
            weights=weights,
            episode_to_gt_reward=ep2gt,
            num_episodes_to_plot=15,
            save_path=f"reward_refined_by_insertion_prior_{sampling}_{weight_tag}.png",
        )


def main():
    main_fn()


if __name__ == "__main__":
    main()
