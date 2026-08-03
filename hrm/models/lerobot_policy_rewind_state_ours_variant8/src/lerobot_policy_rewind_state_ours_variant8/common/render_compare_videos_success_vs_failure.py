"""
Per-pair side-by-side comparison videos: ONE video per (success_ep, failure_ep)
pair. Layout:

    [  success cameras (3 horiz)  |  failure cameras (3 horiz)  ]
    [  shared chart: 4 models, success=solid, failure=dashed,
       two synced cursors (one per split)                        ]

Reads each model's `eval_step*_hybrid*/trajectories.npz` (no model loading) and
loads camera footage from the success / failure LeRobot datasets. Writes mp4s
to `<output_dir>/videos/ep_s{S}_f{F}.mp4`.

Usage (mirrors render_compare_videos.py for --model specs):

    python -m lerobot_policy_rewind_state_ours_variant8.common.render_compare_videos_success_vs_failure \\
        --output_dir /.../_compare_4_anchorstack_step002000_hybrid8_pairs \\
        --num_pairs 10 \\
        --success_repo_id chomeed/dexmg_two_arm_threading \\
        --failure_repo_id chomeed/dexmg_two_arm_threading_failures \\
        --cameras observation.images.agentview,observation.images.robot0_eye_in_hand,observation.images.robot1_eye_in_hand \\
        --model "blend_eps0.2:.../eval_step002000_hybrid8" \\
        --model "blend_eps0.3:.../eval_step002000_hybrid8" \\
        --model "fb_eps0.2_0.0:.../eval_step002000_hybrid8" \\
        --model "fb_eps0.3_0.0:.../eval_step002000_hybrid8"
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
    "observation.images.robot0_eye_in_hand",
    "observation.images.robot1_eye_in_hand",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", action="append", required=True,
        help="'label:/abs/path/to/eval_step*_hybrid*' - repeat for each model",
    )
    p.add_argument("--success_repo_id", default="chomeed/dexmg_two_arm_threading")
    p.add_argument("--failure_repo_id", default="chomeed/dexmg_two_arm_threading_failures")
    p.add_argument("--num_pairs", type=int, default=10,
                   help="number of (success, failure) pairs to render, taken in order "
                        "from the common-across-models episode sets")
    p.add_argument("--pairs", type=str, default=None,
                   help="explicit pairs as 's_ep:f_ep,s_ep:f_ep,...' (overrides --num_pairs)")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--plot_height", type=int, default=360,
                   help="pixel height of both the camera strip and the chart panel")
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


def _resize_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    if img.shape[:2] == (h, w):
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def _load_episode_cam_frames(
    repo_id: str,
    ep_idx: int,
    camera_keys: list[str],
) -> list[np.ndarray]:
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
        frames = item[cam]
        T = frames.shape[0]
        arr = np.zeros((T, frames.shape[2], frames.shape[3], 3), dtype=np.uint8)
        for t in range(T):
            arr[t] = _frame_to_uint8(frames[t])
        out.append(arr)
    return out


def _render_static_chart(
    models_success: list[tuple[str, np.ndarray]],
    models_failure: list[tuple[str, np.ndarray]],
    title: str,
    width_px: int,
    height_px: int,
):
    """Render the shared chart with success (solid) and failure (dashed) curves
    for every model. x-axis is frame index, range [0, T_max-1] where
    T_max = max(T_success, T_failure).

    Returns (base_rgb, t_to_px_fn, (y0_px, y1_px)).
    """
    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    cmap = plt.get_cmap("tab10")

    T_s = max((len(p) for _, p in models_success), default=1)
    T_f = max((len(p) for _, p in models_failure), default=1)
    T_max = max(T_s, T_f)

    for mi, (label, pred) in enumerate(models_success):
        ax.plot(
            np.arange(len(pred)), pred,
            color=cmap(mi % 10), linewidth=1.2, linestyle="-",
            label=f"{label} (succ)",
        )
    for mi, (label, pred) in enumerate(models_failure):
        ax.plot(
            np.arange(len(pred)), pred,
            color=cmap(mi % 10), linewidth=1.2, linestyle="--",
            label=f"{label} (fail)",
        )
    # Linear frame-progress target on the success length only — the natural
    # ground-truth target for the success episode (failure has no monotonic
    # progress reference).
    target = np.arange(T_s, dtype=np.float32) / max(T_s - 1, 1)
    ax.plot(np.arange(T_s), target, color="black", linestyle=":",
            linewidth=0.8, alpha=0.5, label="succ frame-progress target")

    ax.set_xlim(0, max(T_max - 1, 1))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("frame")
    ax.set_ylabel("pred reward")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, loc="upper left", ncol=2)
    ax.set_title(title, fontsize=10)

    fig.tight_layout()
    fig.canvas.draw()

    rgba = np.asarray(fig.canvas.buffer_rgba())
    base = rgba[..., :3].copy()

    bbox = ax.get_window_extent(fig.canvas.get_renderer())
    img_h = base.shape[0]
    x0 = float(bbox.x0)
    x1 = float(bbox.x1)
    y0 = img_h - float(bbox.y1)
    y1 = img_h - float(bbox.y0)
    plt.close(fig)

    def t_to_px(t: int) -> int:
        if T_max <= 1:
            return int(round(x0))
        frac = t / float(T_max - 1)
        return int(round(x0 + frac * (x1 - x0)))

    cursor_y0 = max(0, int(round(y0)))
    cursor_y1 = min(base.shape[0] - 1, int(round(y1)))
    return base, t_to_px, (cursor_y0, cursor_y1)


def _draw_two_cursors(
    chart_base: np.ndarray,
    x_succ_px: int,
    x_fail_px: int,
    y_range: tuple[int, int],
) -> np.ndarray:
    out = chart_base.copy()
    y0, y1 = y_range
    x_s = int(np.clip(x_succ_px, 1, out.shape[1] - 2))
    x_f = int(np.clip(x_fail_px, 1, out.shape[1] - 2))
    # success: green, failure: red. Draw failure first so success draws on top
    # in case they coincide.
    cv2.line(out, (x_f, y0), (x_f, y1), color=(220, 30, 30), thickness=2)
    cv2.line(out, (x_s, y0), (x_s, y1), color=(30, 160, 30), thickness=2)
    return out


def _label_strip(width: int, height: int, text: str, color: tuple[int, int, int]) -> np.ndarray:
    """A small colored bar with a centered text label, used to mark which
    half of the camera row is success vs failure."""
    bar = np.full((height, width, 3), 30, dtype=np.uint8)
    cv2.rectangle(bar, (0, 0), (width - 1, height - 1), color, thickness=-1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, height / 32.0)
    thick = max(1, int(round(scale * 1.5)))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    org = ((width - tw) // 2, (height + th) // 2 - 2)
    cv2.putText(bar, text, org, font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return bar


def _build_cam_row(
    succ_cams: list[np.ndarray],
    fail_cams: list[np.ndarray],
    t_succ: int,
    t_fail: int,
    cam_height: int,
    label_h: int,
) -> np.ndarray:
    """Build the top camera row: [succ_label/succ_cams | fail_label/fail_cams].
    Both halves are sized to share the same final width so the chart below
    aligns cleanly.
    """
    succ_strip = np.concatenate(
        [_resize_h(c[t_succ], cam_height) for c in succ_cams], axis=1,
    )
    fail_strip = np.concatenate(
        [_resize_h(c[t_fail], cam_height) for c in fail_cams], axis=1,
    )
    # Equalize halves to the larger width.
    half_w = max(succ_strip.shape[1], fail_strip.shape[1])
    succ_strip = _resize_to(succ_strip, cam_height, half_w)
    fail_strip = _resize_to(fail_strip, cam_height, half_w)

    succ_label = _label_strip(half_w, label_h, "SUCCESS", (40, 110, 40))
    fail_label = _label_strip(half_w, label_h, "FAILURE", (110, 40, 40))
    succ_half = np.concatenate([succ_label, succ_strip], axis=0)
    fail_half = np.concatenate([fail_label, fail_strip], axis=0)
    return np.concatenate([succ_half, fail_half], axis=1)


def _pick_pairs(
    args: argparse.Namespace,
    common_succ: list[int],
    common_fail: list[int],
) -> list[tuple[int, int]]:
    if args.pairs:
        out: list[tuple[int, int]] = []
        s_set, f_set = set(common_succ), set(common_fail)
        for tok in args.pairs.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if ":" not in tok:
                raise ValueError(f"--pairs entry must be 's_ep:f_ep', got: {tok!r}")
            s_str, f_str = tok.split(":", 1)
            s_ep, f_ep = int(s_str), int(f_str)
            if s_ep not in s_set:
                raise ValueError(f"success ep {s_ep} not in common set")
            if f_ep not in f_set:
                raise ValueError(f"failure ep {f_ep} not in common set")
            out.append((s_ep, f_ep))
        return out
    n = min(args.num_pairs, len(common_succ), len(common_fail))
    return list(zip(common_succ[:n], common_fail[:n]))


def _render_one_pair(
    succ_cam_frames: list[np.ndarray],
    fail_cam_frames: list[np.ndarray],
    models_succ: list[tuple[str, np.ndarray]],
    models_fail: list[tuple[str, np.ndarray]],
    save_path: Path,
    title: str,
    fps: int,
    plot_height: int,
):
    T_s = succ_cam_frames[0].shape[0]
    T_f = fail_cam_frames[0].shape[0]
    T = max(T_s, T_f)

    # Build a sample camera row to determine chart width.
    label_h = max(18, plot_height // 14)
    cam_h = plot_height - label_h
    sample_row = _build_cam_row(
        succ_cam_frames, fail_cam_frames, 0, 0, cam_h, label_h,
    )
    plot_w = max(sample_row.shape[1], 720)

    chart_base, t_to_px, y_range = _render_static_chart(
        models_succ, models_fail, title, plot_w, plot_height,
    )
    # Match chart width to camera-row width for a clean stack.
    if chart_base.shape[1] != sample_row.shape[1]:
        chart_base = _resize_to(chart_base, chart_base.shape[0], sample_row.shape[1])

    sample_composite = np.concatenate([sample_row, chart_base], axis=0)
    H, W = sample_composite.shape[:2]
    pad_w, pad_h = W % 2, H % 2
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
            t_s = min(t, T_s - 1)
            t_f = min(t, T_f - 1)
            cam_row = _build_cam_row(
                succ_cam_frames, fail_cam_frames, t_s, t_f, cam_h, label_h,
            )
            chart_t = _draw_two_cursors(chart_base, t_to_px(t_s), t_to_px(t_f), y_range)
            composite = np.concatenate([cam_row, chart_t], axis=0)
            if composite.shape[:2] != (H, W):
                composite = cv2.resize(composite, (W, H))
            if pad_w or pad_h:
                composite = np.pad(
                    composite, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge",
                )
            writer.append_data(composite)
    finally:
        writer.close()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    videos_root = output_dir / "videos"
    videos_root.mkdir(parents=True, exist_ok=True)
    camera_keys = [c.strip() for c in args.cameras.split(",") if c.strip()]

    print(f"[1/3] Loading {len(args.model)} model evals")
    model_specs = _parse_models(args.model)
    per_model: list[tuple[str, dict]] = []
    for label, eval_dir in model_specs:
        data = _load_one(eval_dir)
        print(f"  {label:<32} success={len(data['success']):>3}  "
              f"failure={len(data['failure']):>3}  from {eval_dir}")
        per_model.append((label, data))

    common_succ = _intersect_episodes([d for _, d in per_model], "success")
    common_fail = _intersect_episodes([d for _, d in per_model], "failure")
    pairs = _pick_pairs(args, common_succ, common_fail)
    if not pairs:
        print("[2/3] no pairs selected (empty intersection or num_pairs=0); exiting")
        return

    print(f"[2/3] rendering {len(pairs)} pair(s); "
          f"|common success|={len(common_succ)}  |common failure|={len(common_fail)}")

    manifest: list[dict] = []
    for s_ep, f_ep in tqdm(pairs, desc="pairs"):
        succ_cams = _load_episode_cam_frames(args.success_repo_id, s_ep, camera_keys)
        fail_cams = _load_episode_cam_frames(args.failure_repo_id, f_ep, camera_keys)
        T_s_cam = succ_cams[0].shape[0]
        T_f_cam = fail_cams[0].shape[0]
        models_succ = [(label, d["success"][s_ep][:T_s_cam]) for label, d in per_model]
        models_fail = [(label, d["failure"][f_ep][:T_f_cam]) for label, d in per_model]
        # Defensive truncation in case predictions are shorter than cameras.
        T_s = min(T_s_cam, min(len(p) for _, p in models_succ))
        T_f = min(T_f_cam, min(len(p) for _, p in models_fail))
        succ_cams = [c[:T_s] for c in succ_cams]
        fail_cams = [c[:T_f] for c in fail_cams]
        models_succ = [(l, p[:T_s]) for l, p in models_succ]
        models_fail = [(l, p[:T_f]) for l, p in models_fail]

        save_path = videos_root / f"ep_s{s_ep:05d}_f{f_ep:05d}.mp4"
        title = f"success ep {s_ep} (T={T_s})   vs   failure ep {f_ep} (T={T_f})"
        _render_one_pair(
            succ_cam_frames=succ_cams,
            fail_cam_frames=fail_cams,
            models_succ=models_succ,
            models_fail=models_fail,
            save_path=save_path,
            title=title,
            fps=args.fps,
            plot_height=args.plot_height,
        )
        manifest.append({
            "success_ep": int(s_ep),
            "failure_ep": int(f_ep),
            "T_success": int(T_s),
            "T_failure": int(T_f),
            "video": str(save_path.relative_to(output_dir)),
        })

    print("[3/3] Writing manifest")
    (videos_root / "manifest.json").write_text(json.dumps(
        {
            "cameras": camera_keys,
            "models": [label for label, _ in per_model],
            "success_repo_id": args.success_repo_id,
            "failure_repo_id": args.failure_repo_id,
            "fps": args.fps,
            "pairs": manifest,
        },
        indent=2,
    ))
    print(f"Done. Rendered {len(manifest)} pair videos under {videos_root}")


if __name__ == "__main__":
    main()
