from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import accelerate

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors

from lerobot_policy_rewind_state.common.factory import make_eval_original_dataset
from lerobot_policy_rewind_state.common.rewind_datasets import EpisodeDataset


MAX_WINDOW = 16


@dataclass
class DenseVizConfig(TrainPipelineConfig):
    num_episodes: int | None = None
    render_sidebyside_videos: bool = False
    video_save_dir: str = "videos_sidebyside"
    video_fps: int = 20
    plot_height: int = 360
    output_tag: str = "eval"


def _make_first_n_episode_dataset(cfg: DenseVizConfig) -> EpisodeDataset:
    return EpisodeDataset(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=list(range(cfg.num_episodes)) if cfg.num_episodes is not None else None,
        video_backend=cfg.dataset.video_backend,
        use_precomputed_embeddings_only=False,
        pad_to_length=None,
    )


def _move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _extract_episode_index(x) -> int:
    if isinstance(x, torch.Tensor):
        return int(x.flatten()[0].item())
    if isinstance(x, np.ndarray):
        return int(x.flatten()[0])
    if isinstance(x, (list, tuple)):
        return int(x[0])
    return int(x)


@torch.no_grad()
def dense_predict_per_frame(
    reward_model,
    raw_images: dict[str, torch.Tensor],
    lang_emb: torch.Tensor,
    state: torch.Tensor | None = None,
    max_window: int = MAX_WINDOW,
    sampling: str = "uniform",
    chunk_size: int = 8,
    n_recent: int = 8,
) -> torch.Tensor:
    """
    Per-frame reward over a full episode, exploiting the causal transformer.

    Online-encoding variant: takes raw images and lets the model's vision_encoder
    run forward on each window (compatible with finetune_encoder=True).

    - Frames [0..min(max_window, N)-1]: one forward pass on the first min(max_window, N)
      frames; take r[0, :k] as per-frame rewards (causal => r[0, t] uses frames 0..t).
    - Frames [max_window..N-1]: one window per target t with indices chosen by ``sampling``.
      The M windows are processed in chunks of ``chunk_size`` to bound peak memory.

    Sampling modes (all windows still end at index t so the prediction is causal):
      "uniform" : max_window indices spaced linearly across [0, t]
      "recent"  : sqrt-biased toward t, snapped to t at the end
      "hybrid"  : ``max_window - n_recent`` indices uniformly across [0, t-n_recent]
                  concatenated with the ``n_recent`` most recent frames [t-n_recent+1, t]

    Args:
        raw_images: dict {camera_key: (1, N, C, H, W)}
        lang_emb:   (1, D_text)
    """
    if sampling not in ("uniform", "recent", "hybrid"):
        raise ValueError(
            f"sampling must be 'uniform', 'recent', or 'hybrid', got '{sampling}'"
        )
    if sampling == "hybrid":
        if not (1 <= n_recent < max_window):
            raise ValueError(
                f"hybrid sampling requires 1 <= n_recent < max_window, "
                f"got n_recent={n_recent}, max_window={max_window}"
            )
    if not raw_images:
        raise ValueError("raw_images is empty")

    cam_keys = list(raw_images.keys())
    first = raw_images[cam_keys[0]]
    if first.ndim != 5 or first.shape[0] != 1:
        raise ValueError(
            f"Expected each camera tensor to have shape (1, N, C, H, W), got {tuple(first.shape)}"
        )
    _, N, _, _, _ = first.shape
    device = first.device

    outputs = []

    if state is not None and (state.ndim != 3 or state.shape[0] != 1):
        raise ValueError(
            f"Expected state shape (1, N, state_dim), got {tuple(state.shape)}"
        )

    k = min(max_window, N)
    part1_batch = {
        **{cam: raw_images[cam][:, :k].contiguous() for cam in cam_keys},
        "task.minilm": lang_emb,
        "lengths": torch.tensor([k], dtype=torch.long, device=device),
    }
    if state is not None:
        part1_batch["observation.state"] = state[:, :k].contiguous()
    r1 = reward_model.predict_reward(part1_batch)
    outputs.append(r1[0, :k])

    if N > max_window:
        targets = np.arange(max_window, N, dtype=np.int64)
        M = len(targets)

        if sampling == "uniform":
            idx_matrix = np.stack(
                [np.linspace(0, int(t), max_window).astype(np.int64) for t in targets],
                axis=0,
            )  # (M, max_window)
        elif sampling == "recent":
            unit = np.linspace(0, 1, max_window)
            biased = unit ** 0.5
            idx_matrix = np.stack(
                [(int(t) * biased).astype(np.int64) for t in targets],
                axis=0,
            )
            idx_matrix[:, -1] = targets
        else:  # "hybrid": n_uniform from [0, t-n_recent] + n_recent latest frames ending at t
            n_uniform = max_window - n_recent
            recent_offsets = np.arange(-n_recent + 1, 1, dtype=np.int64)  # [-n_recent+1, ..., 0]
            rows = []
            for t in targets:
                t_int = int(t)
                uniform_part = np.linspace(0, t_int - n_recent, n_uniform).astype(np.int64)
                recent_part = t_int + recent_offsets
                rows.append(np.concatenate([uniform_part, recent_part]))
            idx_matrix = np.stack(rows, axis=0)  # (M, max_window)
        idx_tensor = torch.as_tensor(idx_matrix, device=device, dtype=torch.long)  # (M, max_window)

        r2_last = []
        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            idx_chunk = idx_tensor[start:end]  # (m, max_window)
            m = idx_chunk.shape[0]

            part2_batch = {
                "task.minilm": lang_emb.expand(m, -1).contiguous(),
                "lengths": torch.full((m,), max_window, dtype=torch.long, device=device),
            }
            for cam in cam_keys:
                base = raw_images[cam][0]            # (N, C, H, W)
                part2_batch[cam] = base[idx_chunk]   # (m, max_window, C, H, W)
            if state is not None:
                state_base = state[0]                            # (N, state_dim)
                part2_batch["observation.state"] = state_base[idx_chunk]  # (m, max_window, state_dim)

            r2 = reward_model.predict_reward(part2_batch)  # (m, max_window)
            r2_last.append(r2[:, -1])

        outputs.append(torch.cat(r2_last, dim=0))

    return torch.cat(outputs, dim=0)


@torch.no_grad()
def collect_dense_reward_predictions(
    dataloader: DataLoader,
    reward_model,
    camera_keys: list[str],
    accelerator: accelerate.Accelerator | None = None,
    preprocessor=None,
    sampling: str = "uniform",
):
    assert preprocessor is not None, "preprocessor must be provided"

    episode_to_pred_rewards = {}
    episode_to_gt_reward = {}

    if accelerator is not None:
        device = accelerator.device
    else:
        device = next(reward_model.parameters()).device

    for batch in tqdm(dataloader, desc="Collecting dense reward trajectories"):
        batch = _move_batch_to_device(batch, device)
        # print(f"Batch keys: {list(batch.keys())}")  # Debug print to check available keys
        batch = preprocessor(batch)

        missing = [k for k in camera_keys if k not in batch]
        if missing:
            raise KeyError(
                f"Dense eval expects raw images under {missing} but the batch only has: "
                f"{list(batch.keys())}"
            )
        raw_images = {k: batch[k] for k in camera_keys}
        first_cam = raw_images[camera_keys[0]]
        if first_cam.ndim != 5 or first_cam.shape[0] != 1:
            raise ValueError(
                f"Dense eval requires batch_size=1 and image shape (1, N, C, H, W), "
                f"got {tuple(first_cam.shape)} for {camera_keys[0]}"
            )

        lang_emb = batch["task.minilm"]
        if lang_emb.ndim == 1:
            lang_emb = lang_emb.unsqueeze(0)

        state = batch.get("observation.state")

        dense = dense_predict_per_frame(
            reward_model, raw_images, lang_emb, state=state, sampling=sampling,
        )

        ep_idx = _extract_episode_index(batch["episode_index"])
        episode_to_pred_rewards[ep_idx] = dense.detach().float().cpu().tolist()

        if "reward" in batch:
            gt = batch["reward"]
            if isinstance(gt, torch.Tensor):
                if gt.ndim == 2:
                    gt = gt[0]
                episode_to_gt_reward[ep_idx] = gt.detach().float().cpu().tolist()

    return episode_to_pred_rewards, episode_to_gt_reward


def plot_reward_trajectories(
    episode_to_pred_rewards,
    episode_to_gt_reward=None,
    num_episodes_to_plot: int = 5,
    save_path: str = "reward_trajectories_dense.png",
):
    episode_ids = sorted(episode_to_pred_rewards.keys())[:num_episodes_to_plot]
    if len(episode_ids) == 0:
        print("No trajectories to plot.")
        return

    plt.figure(figsize=(15, 3 * len(episode_ids)))

    for row, ep_idx in enumerate(episode_ids, start=1):
        plt.subplot(len(episode_ids), 1, row)

        pred = np.asarray(episode_to_pred_rewards[ep_idx], dtype=np.float32)
        plt.plot(pred, label="pred_reward")

        if episode_to_gt_reward is not None and ep_idx in episode_to_gt_reward:
            gt = np.asarray(episode_to_gt_reward[ep_idx], dtype=np.float32)
            plt.plot(gt, label="gt_reward")

        plt.title(f"Episode {ep_idx} Dense Reward Trajectory")
        plt.xlabel("Timestep")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


def compute_rabc_signals(
    pred_reward: np.ndarray,
    delta: int = 8,
    eps: float = 1e-6,
) -> dict[str, np.ndarray]:
    """
    Compute RA-BC style per-frame signals for a single episode.

    - r_hat[i] = pred_reward[i + delta] - pred_reward[i]          (length N - delta)
    - r_hat_norm[i] = min-max normalized r_hat over the episode   (length N - delta)
    - weight[i]     = clip((r_hat_norm - (mu - 2*sigma)) / (4*sigma + eps), 0, 1)
      where mu, sigma are the episode-level mean and std of r_hat_norm.

    Returns a dict with keys: "r_hat", "r_hat_norm", "weight".
    All arrays are length N - delta, aligned to the first index of the delta window.
    """
    pred = np.asarray(pred_reward, dtype=np.float32)
    N = len(pred)
    if N <= delta:
        empty = np.zeros((0,), dtype=np.float32)
        return {"r_hat": empty, "r_hat_norm": empty, "weight": empty}

    r_hat = pred[delta:] - pred[:-delta]

    r_min, r_max = float(r_hat.min()), float(r_hat.max())
    if r_max - r_min < eps:
        r_hat_norm = np.zeros_like(r_hat)
    else:
        r_hat_norm = (r_hat - r_min) / (r_max - r_min)

    mu = float(r_hat_norm.mean())
    sigma = float(r_hat_norm.std())
    weight = np.clip(
        (r_hat_norm - (mu - 2.0 * sigma)) / (4.0 * sigma + eps),
        0.0,
        1.0,
    )

    return {"r_hat": r_hat, "r_hat_norm": r_hat_norm, "weight": weight}


def plot_rabc_visualization(
    episode_to_pred_rewards,
    delta: int = 8,
    num_episodes_to_plot: int = 5,
    save_path: str = "rabc_weights_dense.png",
):
    """
    Visualize for each episode:
      (row a) pred reward
      (row b) delta r_hat = pred[i+delta] - pred[i]
      (row c) RA-BC weight w_i
    """
    episode_ids = sorted(episode_to_pred_rewards.keys())[:num_episodes_to_plot]
    if len(episode_ids) == 0:
        print("No trajectories to plot.")
        return

    n_rows = len(episode_ids)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 3 * n_rows), squeeze=False)

    for row, ep_idx in enumerate(episode_ids):
        pred = np.asarray(episode_to_pred_rewards[ep_idx], dtype=np.float32)
        signals = compute_rabc_signals(pred, delta=delta)
        r_hat = signals["r_hat"]
        weight = signals["weight"]
        x_delta = np.arange(len(r_hat))

        ax0 = axes[row, 0]
        ax0.plot(pred, label="pred_reward", color="tab:blue")
        ax0.set_title(f"Episode {ep_idx} — pred reward")
        ax0.set_xlabel("Timestep")
        ax0.set_ylabel("Reward")
        ax0.grid(True)
        ax0.legend()

        ax1 = axes[row, 1]
        ax1.plot(x_delta, r_hat, label=f"r_hat (Δ={delta})", color="tab:orange")
        ax1.axhline(0, color="gray", linewidth=0.5)
        ax1.set_title(f"Episode {ep_idx} — delta reward")
        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("r_hat")
        ax1.grid(True)
        ax1.legend()

        ax2 = axes[row, 2]
        ax2.plot(x_delta, weight, label="weight w_i", color="tab:green")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_title(f"Episode {ep_idx} — RA-BC weight")
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("weight")
        ax2.grid(True)
        ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")
    print(f"Delta used for RABC signals: {delta} timesteps")


# ── Side-by-side video rendering ─────────────────────────────────────────────

def _frame_rgb(raw_images: dict[str, torch.Tensor], cam: str, t: int) -> np.ndarray:
    img = raw_images[cam][0, t].detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def _resize_h(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] == h:
        return img
    new_w = int(round(img.shape[1] * h / img.shape[0]))
    return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_LINEAR)


def _render_plot_frame(
    pred: np.ndarray,
    gt: np.ndarray | None,
    t: int,
    width_px: int,
    height_px: int,
    title: str,
) -> np.ndarray:
    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)

    xs = np.arange(len(pred))
    ax.plot(xs, pred, label="pred_reward", color="tab:blue", linewidth=1.6)
    if gt is not None:
        ax.plot(np.arange(len(gt)), gt, label="gt_reward", color="tab:green",
                linewidth=1.2, alpha=0.8)

    ax.axvline(t, color="red", linewidth=1.2, alpha=0.8)
    ax.scatter([t], [pred[t]], color="red", zorder=5, s=20)

    ax.set_xlim(0, max(len(pred) - 1, 1))
    y_min = float(min(pred.min(), 0.0))
    y_max = float(max(pred.max(), 1.0))
    pad = 0.05 * max(y_max - y_min, 1e-3)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlabel("timestep")
    ax.set_ylabel("reward")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(title, fontsize=10)

    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[..., :3].copy()
    plt.close(fig)
    return rgb


def render_episode_sidebyside(
    raw_images: dict[str, torch.Tensor],
    pred: np.ndarray,
    gt: np.ndarray | None,
    save_path: str,
    fps: int = 20,
    plot_height: int = 360,
    title: str = "",
) -> None:
    cam_keys = list(raw_images.keys())
    T = raw_images[cam_keys[0]].shape[1]

    cam0 = [_resize_h(_frame_rgb(raw_images, c, 0), plot_height) for c in cam_keys]
    cam_strip0 = np.concatenate(cam0, axis=1)
    plot_w = max(cam_strip0.shape[1], 480)

    plot0 = _render_plot_frame(pred, gt, 0, plot_w, plot_height, title)
    plot0 = _resize_h(plot0, plot_height)
    composite0 = np.concatenate([cam_strip0, plot0], axis=1)
    H, W = composite0.shape[:2]

    # H.264 + yuv420p for QuickTime/Safari/Chrome compatibility on macOS.
    # H.264 requires even width/height.
    pad_w = W % 2
    pad_h = H % 2
    out_W, out_H = W + pad_w, H + pad_h

    writer = imageio.get_writer(
        save_path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_params=["-movflags", "+faststart"],
    )

    try:
        for t in range(T):
            cam_t = [_resize_h(_frame_rgb(raw_images, c, t), plot_height) for c in cam_keys]
            cam_strip = np.concatenate(cam_t, axis=1)
            plot_img = _resize_h(
                _render_plot_frame(pred, gt, t, plot_w, plot_height, title),
                plot_height,
            )
            composite = np.concatenate([cam_strip, plot_img], axis=1)
            if composite.shape[:2] != (H, W):
                composite = cv2.resize(composite, (W, H))
            if pad_w or pad_h:
                composite = np.pad(
                    composite,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="edge",
                )
            writer.append_data(composite)
    finally:
        writer.close()


@torch.no_grad()
def collect_and_render_sidebyside(
    dataloader: DataLoader,
    reward_model,
    camera_keys: list[str],
    preprocessor,
    accelerator: accelerate.Accelerator | None,
    video_save_dir: Path,
    video_fps: int,
    plot_height: int,
    dataset_tag: str,
    sampling: str = "uniform",
):
    device = accelerator.device if accelerator is not None else next(reward_model.parameters()).device
    video_save_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for batch in tqdm(dataloader, desc=f"Rendering {dataset_tag}"):
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

        pred = dense_predict_per_frame(
            reward_model, raw_images, lang_emb, state=state, sampling=sampling,
        ).detach().float().cpu().numpy()

        gt = None
        if "reward" in batch:
            gt_t = batch["reward"]
            if isinstance(gt_t, torch.Tensor):
                if gt_t.ndim == 2:
                    gt_t = gt_t[0]
                gt = gt_t.detach().float().cpu().numpy()

        ep_idx = _extract_episode_index(batch["episode_index"])
        out_path = video_save_dir / f"{dataset_tag}_episode_{ep_idx:05d}.mp4"
        title = f"{dataset_tag}  ep={ep_idx}  T={len(pred)}"
        render_episode_sidebyside(
            raw_images=raw_images,
            pred=pred,
            gt=gt,
            save_path=str(out_path),
            fps=video_fps,
            plot_height=plot_height,
            title=title,
        )
        summary[ep_idx] = {
            "video_path": str(out_path),
            "T": int(len(pred)),
            "pred_min": float(pred.min()),
            "pred_max": float(pred.max()),
            "pred_last": float(pred[-1]),
        }

    return summary


# ── Main ─────────────────────────────────────────────────────────────────────

"""
Success val
python -m lerobot_policy_rewind_state.common.eval_rewind_reward_visualization_dense \
  --policy.type rewind_state \
  --policy.pretrained_path /home/chomeed/workspace/insertion_generation/ckpts/rewind/rewind_state_peg_insertion_v2/checkpoints/007000/pretrained_model \
  --policy.finetune_encoder=True \
  --policy.num_cameras=2 \
  --dataset.repo_id chomeed/peg_insertion_full_500_unprivileged_fixed_v2_lerobot_success_for_rewind_val \
  --dataset.video_backend torchcodec \
  --num_workers 4

Side-by-side video render (first N episodes):
python -m lerobot_policy_rewind_state.common.eval_rewind_reward_visualization_dense \
  --policy.type=rewind_state \
  --policy.pretrained_path=/home/chomeed/workspace/insertion_generation/ckpts/rewind/rewind_state_robomimic_square_v1_pro6000/checkpoints/010000/pretrained_model \
  --policy.finetune_encoder=True \
  --policy.num_cameras=2 \
  --policy.encoder_camera_keys='["observation.images.agentview","observation.images.eye_in_hand"]' \
  --dataset.repo_id=chomeed/robomimic_square_resized_peg_success \
  --dataset.video_backend=torchcodec \
  --num_workers=2 \
  --num_episodes=5 \
  --render_sidebyside_videos=True \
  --video_save_dir=videos_sidebyside_success \
  --output_tag=robomimic_success
"""
@parser.wrap()
def infer_reward_trajectory_main(cfg: DenseVizConfig):
    accelerator = accelerate.Accelerator()

    if cfg.render_sidebyside_videos and cfg.num_episodes is not None:
        dataset = _make_first_n_episode_dataset(cfg)
    else:
        dataset = make_eval_original_dataset(cfg)
        dataset.pad_to_length = None
        dataset.use_precomputed_embeddings_only = False
    print(f"Dataset {cfg.dataset.repo_id} loaded. Num episodes: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers if cfg.num_workers is not None else 4,
        pin_memory=True,
    )

    ds_meta = dataset.meta if hasattr(dataset, "meta") else dataset._datasets[0].meta

    reward_model = make_policy(
        cfg=cfg.policy,
        ds_meta=ds_meta,
    )
    reward_model.eval()

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        dataset_stats=ds_meta.stats if hasattr(ds_meta, "stats") else None,
    )

    print("Model loaded successfully")

    dataloader, reward_model = accelerator.prepare(dataloader, reward_model)

    SAMPLING = "uniform"
    tag = cfg.output_tag

    if cfg.render_sidebyside_videos:
        save_dir = Path(cfg.video_save_dir)
        summary = collect_and_render_sidebyside(
            dataloader=dataloader,
            reward_model=reward_model,
            camera_keys=list(cfg.policy.encoder_camera_keys),
            preprocessor=preprocessor,
            accelerator=accelerator,
            video_save_dir=save_dir,
            video_fps=cfg.video_fps,
            plot_height=cfg.plot_height,
            dataset_tag=tag,
            sampling=SAMPLING,
        )
        if accelerator.is_main_process:
            print(f"\nRendered {len(summary)} episode videos to {save_dir.resolve()}")
            for ep_idx, info in sorted(summary.items()):
                print(
                    f"  ep={ep_idx}: T={info['T']:4d}  "
                    f"pred_min={info['pred_min']:+.3f}  "
                    f"pred_max={info['pred_max']:+.3f}  "
                    f"pred_last={info['pred_last']:+.3f}  -> {info['video_path']}"
                )
        return

    print("Start dense inference...")
    episode_to_pred_rewards, episode_to_gt_reward = collect_dense_reward_predictions(
        dataloader=dataloader,
        reward_model=reward_model,
        camera_keys=list(cfg.policy.encoder_camera_keys),
        accelerator=accelerator,
        preprocessor=preprocessor,
        sampling=SAMPLING,
    )

    if accelerator.is_main_process:
        plot_reward_trajectories(
            episode_to_pred_rewards=episode_to_pred_rewards,
            episode_to_gt_reward=episode_to_gt_reward,
            num_episodes_to_plot=15,
            save_path=f"reward_trajectories_dense_{SAMPLING}_{tag}.png",
        )

        DELTA = 2
        plot_rabc_visualization(
            episode_to_pred_rewards=episode_to_pred_rewards,
            delta=DELTA,
            num_episodes_to_plot=15,
            save_path=f"rabc_weights_dense_delta={DELTA}_{SAMPLING}_{tag}.png",
        )

        threshold = 0.75
        total = len(episode_to_pred_rewards)
        max_over = 0
        last_over = 0
        for ep_idx, preds in episode_to_pred_rewards.items():
            preds_np = np.asarray(preds, dtype=np.float32)
            if preds_np.size == 0:
                continue
            if preds_np.max() > threshold:
                max_over += 1
            if preds_np[-1] > threshold:
                last_over += 1

        pct_max = (max_over / total * 100) if total > 0 else 0.0
        pct_last = (last_over / total * 100) if total > 0 else 0.0
        print(
            f"Episodes with max pred reward > {threshold}: "
            f"{max_over}/{total} ({pct_max:.1f}%)"
        )
        print(
            f"Episodes with last-frame pred reward > {threshold}: "
            f"{last_over}/{total} ({pct_last:.1f}%)"
        )


def main():
    infer_reward_trajectory_main()


if __name__ == "__main__":
    main()
