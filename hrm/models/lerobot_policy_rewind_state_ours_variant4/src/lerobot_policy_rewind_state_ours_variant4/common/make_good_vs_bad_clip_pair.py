"""
Build one (bad clip, good clip) pair where:

  * The bad clip ends at a frame whose rabc.weight is 0 in the ours-rewind
    RABC-tagged dataset (i.e. RABC says "no progress here").
  * The good clip ends at a frame whose rabc.weight is exactly 1 (RABC says
    "definitely progressing").
  * The two clips are picked to be at *similar task stage* — the good clip
    is the cosine-nearest match (in mean-pooled ours-trunk embedding) among
    a random pool of weight=1 candidates.

Side-by-side MP4 is rendered, with each frame labeled by (episode_index,
frame_index, rabc.weight).

By default we draw the bad clip from the mid-trajectory bin (40-60% relative
position) since that's where the ours/baseline divergence is concentrated.

Run:
    python -m lerobot_policy_rewind_state_ours_variant4.common.make_good_vs_bad_clip_pair \
        --ckpt /home/chomeed/workspace/insertion_generation/ckpts/rewind/rewind_state_ours_robomimic_square_v1_eps50_clip20/checkpoints/010000/pretrained_model \
        --tagged_repo_id chomeed/robomimic_square_resized_peg_success_rabc_delta8_kappa0.10_using_rewind_ours \
        --output_mp4 /tmp/bad_vs_good_clip.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_policy_rewind_state_ours_variant4.modeling_rewind_state_ours_variant4 import (
    RewindStateOursVariant4Policy,
)


CAMERA_KEYS = [
    "observation.images.agentview",
    "observation.images.eye_in_hand",
]
PANEL_H = 240


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tagged_repo_id", required=True,
                   help="The _rabc_..._using_rewind_ours dataset that has the "
                        "rabc.weight column. Must contain images.")
    p.add_argument("--clip_len", type=int, default=16)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--bad_rel_pos_lo", type=float, default=0.4)
    p.add_argument("--bad_rel_pos_hi", type=float, default=0.6,
                   help="Pick the bad clip's last frame from this relative-"
                        "position window in its episode.")
    p.add_argument("--n_good_candidates", type=int, default=80,
                   help="Random weight=1 clips to match against.")
    p.add_argument("--bad_seed", type=int, default=0)
    p.add_argument("--good_seed", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--output_mp4", required=True)
    return p.parse_args()


def _build_clip_batch(ds: LeRobotDataset, ep_idx: int, end_frame: int,
                      clip_len: int, frame_stride: int, device: str) -> dict:
    """Slice clip_len frames ending at `end_frame` (inclusive) with given stride."""
    ep = ds.meta.episodes[ep_idx]
    ep_start = int(ep["dataset_from_index"])
    end_abs = ep_start + int(end_frame)
    start_abs = end_abs - (clip_len - 1) * frame_stride
    abs_indices = [start_abs + i * frame_stride for i in range(clip_len)]

    rel = [i - ep_start for i in abs_indices]
    timestamps = ds.hf_dataset[rel]["timestamp"]
    timestamps = [t.item() if isinstance(t, torch.Tensor) else float(t)
                  for t in timestamps]

    cams = {
        cam: ds._query_videos({cam: timestamps}, ep_idx)[cam]
        for cam in CAMERA_KEYS
    }
    state_items = ds.hf_dataset[rel]["observation.state"]
    state = (state_items if isinstance(state_items, torch.Tensor)
             else torch.stack(state_items)).float()

    task_idx = int(ds.hf_dataset[rel[0]]["task_index"])
    task_str = ds.meta.tasks.iloc[task_idx].name

    batch = {
        **{cam: cams[cam].float().unsqueeze(0).to(device) for cam in CAMERA_KEYS},
        "observation.state": state.unsqueeze(0).to(device),
        "task": [task_str],
        "episode_index": torch.tensor([ep_idx], device=device),
    }
    return batch, cams


@torch.no_grad()
def _clip_embedding(policy, preprocessor, batch) -> np.ndarray:
    """Mean-pooled hidden-state embedding of the clip via ours' trunk."""
    batch = preprocessor(batch)
    flattened_h, _ = policy._trunk(batch)
    emb = flattened_h.mean(dim=1).squeeze(0).detach().float().cpu().numpy()
    norm = float(np.linalg.norm(emb))
    if norm < 1e-8:
        return emb
    return emb / norm


def _frame_to_rgb(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _resize_h(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] == h:
        return img
    new_w = int(round(img.shape[1] * h / img.shape[0]))
    return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_LINEAR)


def _label_strip(text: str, width: int, height: int = 28) -> np.ndarray:
    img = np.full((height, width, 3), 30, dtype=np.uint8)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (240, 240, 240), 1, cv2.LINE_AA)
    return img


def _render_pair(
    cams_bad: dict, cams_good: dict,
    bad_label: str, good_label: str,
    out_path: Path, fps: int,
):
    T = cams_bad[CAMERA_KEYS[0]].shape[0]
    frames = []
    for t in range(T):
        left_imgs = [_resize_h(_frame_to_rgb(cams_bad[k][t]), PANEL_H)
                     for k in CAMERA_KEYS]
        right_imgs = [_resize_h(_frame_to_rgb(cams_good[k][t]), PANEL_H)
                      for k in CAMERA_KEYS]
        left = np.concatenate(left_imgs, axis=1)
        right = np.concatenate(right_imgs, axis=1)
        h = max(left.shape[0], right.shape[0])
        gap = np.full((h, 8, 3), 30, dtype=np.uint8)
        body = np.concatenate([left, gap, right], axis=1)

        title_left = _label_strip(bad_label, left.shape[1])
        title_right = _label_strip(good_label, right.shape[1])
        title_gap = np.full((title_left.shape[0], 8, 3), 30, dtype=np.uint8)
        title = np.concatenate([title_left, title_gap, title_right], axis=1)

        frame = np.concatenate([title, body], axis=0)
        if frame.shape[1] % 2:
            frame = np.pad(frame, ((0, 0), (0, 1), (0, 0)), mode="edge")
        if frame.shape[0] % 2:
            frame = np.pad(frame, ((0, 1), (0, 0), (0, 0)), mode="edge")
        frames.append(frame)

    writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264", pixelformat="yuv420p",
        macro_block_size=1, ffmpeg_params=["-movflags", "+faststart"],
    )
    try:
        for f in frames:
            writer.append_data(f)
    finally:
        writer.close()


def main():
    args = parse_args()

    print(f"[1/5] Loading policy from {args.ckpt}")
    policy_cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    policy_cfg.pretrained_path = args.ckpt
    policy_cfg.device = args.device

    print(f"[2/5] Loading dataset {args.tagged_repo_id}")
    ds = LeRobotDataset(repo_id=args.tagged_repo_id)
    policy = make_policy(cfg=policy_cfg, ds_meta=ds.meta)
    policy.eval(); policy.to(args.device)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=None,
        dataset_stats=ds.meta.stats if hasattr(ds.meta, "stats") else None,
    )

    print(f"[3/5] Scanning parquet for bad/good frame candidates")
    weights = []
    ep_indices = []
    for k in range(ds.meta.total_episodes):
        ep = ds.meta.episodes[k]
        start = int(ep["dataset_from_index"]); end = int(ep["dataset_to_index"])
        ep_len = end - start
        # Load weight column for this episode (avoids loading everything at once).
        w_col = []
        for f in range(ep_len):
            sample = ds.hf_dataset[start + f]
            w = sample["rabc.weight"]
            w_col.append(float(w.item()) if isinstance(w, torch.Tensor) else float(w))
        w_col = np.array(w_col, dtype=np.float32)
        weights.append(w_col)
        ep_indices.append((k, ep_len))

    # Pick a bad clip: random frame with w==0 whose end_frame is in the
    # requested rel-pos window AND has enough lead-in for the clip.
    span = (args.clip_len - 1) * args.frame_stride + 1
    bad_rng = np.random.default_rng(args.bad_seed)
    bad_candidates = []
    for (ep, ep_len), w in zip(ep_indices, weights):
        for f in range(span - 1, ep_len):
            if w[f] != 0.0:
                continue
            rel = f / max(ep_len - 1, 1)
            if args.bad_rel_pos_lo <= rel <= args.bad_rel_pos_hi:
                bad_candidates.append((ep, f, rel))
    if not bad_candidates:
        raise SystemExit(
            f"No bad frames (w==0) found in rel-pos "
            f"[{args.bad_rel_pos_lo}, {args.bad_rel_pos_hi}]. "
            f"Widen the window."
        )
    bad_ep, bad_end_frame, bad_rel = bad_candidates[
        bad_rng.integers(0, len(bad_candidates))
    ]
    print(f"      chose BAD clip: ep={bad_ep} end_frame={bad_end_frame} "
          f"rel_pos={bad_rel:.2f}  ({len(bad_candidates)} candidates total)")

    # Pick N good candidates: random frames with w==1, anywhere in their episode
    # with enough lead-in.
    good_rng = np.random.default_rng(args.good_seed)
    good_pool = []
    for (ep, ep_len), w in zip(ep_indices, weights):
        for f in range(span - 1, ep_len):
            if w[f] == 1.0:
                good_pool.append((ep, f))
    if not good_pool:
        raise SystemExit("No good frames (w==1) found.")
    n = min(args.n_good_candidates, len(good_pool))
    pick = good_rng.choice(len(good_pool), size=n, replace=False)
    good_candidates = [good_pool[i] for i in pick]
    print(f"      sampled {n} good clip candidates from a pool of {len(good_pool)}")

    print(f"[4/5] Computing embeddings (1 bad + {len(good_candidates)} good)")
    bad_batch, bad_cams = _build_clip_batch(
        ds, bad_ep, bad_end_frame, args.clip_len, args.frame_stride, args.device,
    )
    bad_emb = _clip_embedding(policy, preprocessor, bad_batch)

    good_embs = []
    good_data = []
    for (ep, f) in tqdm(good_candidates, desc="good"):
        try:
            batch, cams = _build_clip_batch(
                ds, ep, f, args.clip_len, args.frame_stride, args.device,
            )
            emb = _clip_embedding(policy, preprocessor, batch)
        except Exception as e:
            print(f"  [skip] ep={ep} f={f}: {e}")
            continue
        good_embs.append(emb)
        good_data.append((ep, f, cams))
    good_embs = np.stack(good_embs, axis=0)

    sims = good_embs @ bad_emb
    best = int(np.argmax(sims))
    best_ep, best_f, best_cams = good_data[best]
    print(f"      best match: ep={best_ep} end_frame={best_f}  cos={sims[best]:.4f}")

    bad_label = (f"BAD  ep={bad_ep}  end_frame={bad_end_frame}  "
                 f"rel_pos={bad_rel:.2f}  w=0.00")
    good_label = (f"GOOD ep={best_ep}  end_frame={best_f}  "
                  f"w=1.00  cos={sims[best]:.3f}")

    out = Path(args.output_mp4)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[5/5] Rendering pair -> {out}")
    _render_pair(bad_cams, best_cams, bad_label, good_label, out, args.fps)
    print("Done.")


if __name__ == "__main__":
    main()
