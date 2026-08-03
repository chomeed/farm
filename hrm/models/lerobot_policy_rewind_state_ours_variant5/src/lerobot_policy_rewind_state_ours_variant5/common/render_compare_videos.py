"""
Per-episode side-by-side comparison videos: camera footage on the left, all
N model reward curves overlaid on the right with a moving cursor synced to
the current frame.

Reads each model's `eval_val_last*/trajectories.npz` (no model loading) and
the success / failure LeRobot datasets for the camera footage. Writes mp4s
to `<output_dir>/videos/<split>/ep_<idx>.mp4`.

Usage mirrors `compare_rewind_models_reward_predictions.py`:

    python -m lerobot_policy_rewind_state_ours_variant5.common.render_compare_videos \\
        --output_dir /.../_compare_six_hybrid \\
        --num_episodes 20 \\
        --splits success,failure \\
        --success_repo_id chomeed/robomimic_square_resized_peg_success \\
        --failure_repo_id chomeed/robomimic_square_resized_peg_failure \\
        --model "variant5_base:.../eval_val_last_hybrid" \\
        --model "varlen_blend:.../eval_val_last_hybrid" \\
        ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imageio.v2 as imageio
import torch
from tqdm import tqdm

from lerobot_policy_rewind_state.common.rewind_datasets import EpisodeDataset


DEFAULT_CAMERA_KEYS = [
    "observation.images.agentview",
    "observation.images.eye_in_hand",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", action="append", required=True,
        help="'label:/abs/path/to/eval_val_last*' - repeat for each model",
    )
    p.add_argument("--success_repo_id", default="chomeed/robomimic_square_resized_peg_success")
    p.add_argument("--failure_repo_id", default="chomeed/robomimic_square_resized_peg_failure")
    p.add_argument("--splits", default="success,failure",
                   help="comma-separated subset of {success,failure}")
    p.add_argument("--num_episodes", type=int, default=20,
                   help="how many episodes per split (taken from the common episode set)")
    p.add_argument("--episodes", type=str, default=None,
                   help="comma-separated explicit episode indices (overrides --num_episodes)")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--plot_height", type=int, default=360,
                   help="pixel height of the chart panel; camera strip is resized to match")
    p.add_argument("--cameras", default=",".join(DEFAULT_CAMERA_KEYS),
                   help="comma-separated camera keys to render")
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
    npz_path = eval_dir / "trajectories.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"missing {npz_path}. Run eval_val_loss_and_visualize for this checkpoint first."
        )
    npz = np.load(npz_path)
    success: dict[int, np.ndarray] = {}
    failure: dict[int, np.ndarray] = {}
    for key in npz.files:
        split, ep_str, _ = key.split("/")
        ep = int(ep_str)
        arr = np.asarray(npz[key], dtype=np.float32)
        (success if split == "success" else failure)[ep] = arr
    return {"success": success, "failure": failure}


def _intersect_episodes(per_model: list[dict], split: str) -> list[int]:
    sets = [set(d[split].keys()) for d in per_model]
    return sorted(set.intersection(*sets)) if sets else []


def _frame_to_uint8(frame: torch.Tensor) -> np.ndarray:
    """(C, H, W) float in [0, 1] -> (H, W, 3) uint8 RGB."""
    arr = frame.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _resize_h(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] == h:
        return img
    new_w = max(1, int(round(img.shape[1] * h / img.shape[0])))
    return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_LINEAR)


def _render_static_chart(
    models: list[tuple[str, np.ndarray]],
    title: str,
    width_px: int,
    height_px: int,
):
    """Render the per-episode multi-model reward chart once.

    Returns: (base_rgb [H,W,3] uint8, x_pixel_per_frame_fn).
    The fn maps frame index t in [0, T-1] to integer pixel x in [0, W-1].
    """
    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    cmap = plt.get_cmap("tab10")

    T = max(len(pred) for _, pred in models)
    for mi, (label, pred) in enumerate(models):
        xs = np.arange(len(pred))
        ax.plot(xs, pred, color=cmap(mi % 10), linewidth=1.2, label=label)
    target = np.arange(T, dtype=np.float32) / max(T - 1, 1)
    ax.plot(np.arange(T), target, color="black", linestyle="--",
            linewidth=0.8, alpha=0.5, label="frame-progress target")

    ax.set_xlim(0, max(T - 1, 1))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("frame")
    ax.set_ylabel("pred reward")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.set_title(title, fontsize=10)

    fig.tight_layout()
    fig.canvas.draw()

    rgba = np.asarray(fig.canvas.buffer_rgba())
    base = rgba[..., :3].copy()  # (H, W, 3) uint8

    bbox = ax.get_window_extent(fig.canvas.get_renderer())
    img_h = base.shape[0]
    # matplotlib display coords have origin bottom-left; arrays are top-left.
    x0 = float(bbox.x0)
    x1 = float(bbox.x1)
    y0 = img_h - float(bbox.y1)
    y1 = img_h - float(bbox.y0)
    plt.close(fig)

    def t_to_px(t: int) -> int:
        if T <= 1:
            return int(round(x0))
        frac = t / float(T - 1)
        return int(round(x0 + frac * (x1 - x0)))

    cursor_y0 = max(0, int(round(y0)))
    cursor_y1 = min(base.shape[0] - 1, int(round(y1)))
    return base, t_to_px, (cursor_y0, cursor_y1)


def _draw_cursor(chart_base: np.ndarray, x_px: int, y_range: tuple[int, int]) -> np.ndarray:
    out = chart_base.copy()
    y0, y1 = y_range
    x_px = int(np.clip(x_px, 1, out.shape[1] - 2))
    cv2.line(out, (x_px, y0), (x_px, y1), color=(220, 30, 30), thickness=2)
    return out


def _render_one_video(
    cam_frames_list: list[np.ndarray],
    models_for_ep: list[tuple[str, np.ndarray]],
    save_path: Path,
    title: str,
    fps: int,
    plot_height: int,
):
    """cam_frames_list[i] has shape (T, H, W, 3) uint8 for one camera."""
    T = cam_frames_list[0].shape[0]
    # Build sample camera strip to size the chart.
    sample_strip = np.concatenate(
        [_resize_h(cam_frames_list[ci][0], plot_height) for ci in range(len(cam_frames_list))],
        axis=1,
    )
    plot_w = max(sample_strip.shape[1], 540)

    chart_base, t_to_px, y_range = _render_static_chart(
        models_for_ep, title, plot_w, plot_height,
    )
    # Chart may render at slightly different height due to tight_layout; resize.
    chart_base = _resize_h(chart_base, plot_height)

    sample_composite = np.concatenate([sample_strip, chart_base], axis=1)
    H, W = sample_composite.shape[:2]
    pad_w, pad_h = W % 2, H % 2  # libx264 needs even dims
    out_W, out_H = W + pad_w, H + pad_h

    save_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(save_path),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_params=["-movflags", "+faststart"],
    )
    try:
        for t in range(T):
            cam_strip = np.concatenate(
                [_resize_h(cam_frames_list[ci][t], plot_height) for ci in range(len(cam_frames_list))],
                axis=1,
            )
            chart_t = _draw_cursor(chart_base, t_to_px(t), y_range)
            composite = np.concatenate([cam_strip, chart_t], axis=1)
            if composite.shape[:2] != (H, W):
                composite = cv2.resize(composite, (W, H))
            if pad_w or pad_h:
                composite = np.pad(
                    composite, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge",
                )
            writer.append_data(composite)
    finally:
        writer.close()


def _load_episode_cam_frames(
    repo_id: str,
    ep_idx: int,
    camera_keys: list[str],
) -> list[np.ndarray]:
    """Open a one-episode EpisodeDataset and return one (T, H, W, 3) uint8 array per camera."""
    ds = EpisodeDataset(
        repo_id=repo_id,
        episodes=[ep_idx],
        camera_keys=camera_keys,
        text_embedding_key=None,
        state_key=None,
        use_precomputed_embeddings_only=False,
        pad_to_length=None,
    )
    item = ds[0]
    out: list[np.ndarray] = []
    for cam in camera_keys:
        frames = item[cam]  # (T, C, H, W) float in [0, 1]
        T = frames.shape[0]
        arr = np.zeros((T, frames.shape[2], frames.shape[3], 3), dtype=np.uint8)
        for t in range(T):
            arr[t] = _frame_to_uint8(frames[t])
        out.append(arr)
    return out


def _pick_episodes(
    args: argparse.Namespace,
    common: list[int],
) -> list[int]:
    if args.episodes:
        explicit = [int(x) for x in args.episodes.split(",") if x.strip()]
        missing = [e for e in explicit if e not in set(common)]
        if missing:
            raise ValueError(
                f"requested episodes {missing} are not in the common set "
                f"(common: {common[0]}..{common[-1]} if any)"
            )
        return explicit
    return common[: args.num_episodes]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    videos_root = output_dir / "videos"
    videos_root.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    camera_keys = [c.strip() for c in args.cameras.split(",") if c.strip()]
    repo_id_for_split = {
        "success": args.success_repo_id,
        "failure": args.failure_repo_id,
    }
    for s in splits:
        if s not in repo_id_for_split:
            raise ValueError(f"unknown split {s!r}; expected success or failure")

    print(f"[1/3] Loading {len(args.model)} model evals")
    model_specs = _parse_models(args.model)
    per_model: list[tuple[str, dict]] = []
    for label, eval_dir in model_specs:
        data = _load_one(eval_dir)
        print(f"  {label:<32} success={len(data['success']):>3}  "
              f"failure={len(data['failure']):>3}  from {eval_dir}")
        per_model.append((label, data))

    manifest: dict[str, list[dict]] = {s: [] for s in splits}
    for split in splits:
        common = _intersect_episodes([d for _, d in per_model], split)
        ep_ids = _pick_episodes(args, common)
        if not ep_ids:
            print(f"[2/3] {split}: no episodes selected, skipping")
            continue
        print(f"[2/3] {split}: rendering {len(ep_ids)} episodes "
              f"(eps {ep_ids[0]}..{ep_ids[-1]}) from {repo_id_for_split[split]}")

        for ep_idx in tqdm(ep_ids, desc=f"{split}"):
            cam_frames = _load_episode_cam_frames(
                repo_id_for_split[split], ep_idx, camera_keys,
            )
            T_cam = cam_frames[0].shape[0]
            models_for_ep = [
                (label, data[split][ep_idx][:T_cam]) for label, data in per_model
            ]
            T_pred = min(len(p) for _, p in models_for_ep)
            if T_pred != T_cam:
                # Cameras and predictions should already be the same length; clip both for safety.
                T_min = min(T_cam, T_pred)
                cam_frames = [c[:T_min] for c in cam_frames]
                models_for_ep = [(label, p[:T_min]) for label, p in models_for_ep]
                T_cam = T_min

            save_path = videos_root / split / f"ep_{ep_idx:05d}.mp4"
            title = f"{split} ep {ep_idx}  (T={T_cam})"
            _render_one_video(
                cam_frames_list=cam_frames,
                models_for_ep=models_for_ep,
                save_path=save_path,
                title=title,
                fps=args.fps,
                plot_height=args.plot_height,
            )
            manifest[split].append({
                "ep": int(ep_idx),
                "T": int(T_cam),
                "video": str(save_path.relative_to(output_dir)),
            })

    print("[3/3] Writing manifest")
    (videos_root / "manifest.json").write_text(json.dumps(
        {
            "splits": splits,
            "cameras": camera_keys,
            "models": [label for label, _ in per_model],
            "fps": args.fps,
            "episodes": manifest,
        },
        indent=2,
    ))
    total = sum(len(v) for v in manifest.values())
    print(f"Done. Rendered {total} videos under {videos_root}")


if __name__ == "__main__":
    main()
