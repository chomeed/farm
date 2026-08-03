"""
Build a preference-pair manifest against the BASE rewind_state checkpoint.

For each failure episode:
  1. dense per-frame predicted reward (base model, MAX_WINDOW=16).
  2. critical_t = argmax(smoothed_reward, smooth_window).
  3. extract a centered clip of length clip_len * frame_stride.
For each success episode:
  4. random_clips_per_ep clip starts.
  5. drop those with smoothed reward delta <= reward_delta_eps (improving-only).
For all clips:
  6. compute the last-`match_window`-frame matching embedding. By default
     (match_state_weight=0) this is the per-stream camera embedding. With
     match_state_weight>0 it becomes a weighted blend of a robot-state
     pose-trajectory block and the per-camera stage blocks.
For each failure critical clip:
  7. take top-K success matches by cosine, optionally blended with a DTW
     distance on the robot-state trajectory (see --match_dtw_weight).
Output: JSON manifest read by PreferenceTaggedRewindDataset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

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
from lerobot_policy_rewind_state.modeling_rewind_state import RewindStatePolicy

from ._match_helpers import (
    centered_window,
    critical_timestep,
    sample_random_clip_starts,
    smooth,
    strided_indices,
    success_critical_timestep,
)
from .manifest import write_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_ckpt", required=True,
                   help="path to .../checkpoints/<step>/pretrained_model")
    p.add_argument("--success_repo_id", required=True)
    p.add_argument("--failure_repo_id", required=True)
    p.add_argument("--num_episodes", type=int, default=50,
                   help="episodes per source")
    p.add_argument("--clip_len", type=int, default=16)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--smooth_window", type=int, default=7)
    p.add_argument("--success_clips_per_ep", type=int, default=20)
    p.add_argument("--reward_delta_eps", type=float, default=0.0,
                   help="minimum smoothed reward improvement on success bank clips")
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--match_window", type=int, default=5,
                   help="Match clips on the per-stream mean of the LAST N "
                        "frames' transformer hidden states, instead of the "
                        "whole clip. Distinct from --smooth_window (which "
                        "denoises the reward curve for clip anchoring).")
    p.add_argument("--match_state_weight", type=float, default=0.0,
                   help="Weight (0..1) of the robot-state pose-trajectory "
                        "block in the matching similarity. 0.0 (default) = "
                        "no state weighting: match on cameras only. e.g. 0.4 "
                        "= 0.4*cos(state_trajectory) + 0.3*cos(cam) per "
                        "camera, so clips match more on how the robot moves "
                        "while the cameras still carry task stage.")
    p.add_argument("--match_dtw_weight", type=float, default=0.0,
                   help="Weight (0..1) of a Dynamic Time Warping distance on "
                        "the robot-state trajectory, blended into the match "
                        "score. 0.0 (default) = pure cosine. >0 = per query, "
                        "score = (1-w)*cosine + w*(1-dtw), each min-max "
                        "normalized over candidates. DTW captures temporal "
                        "dynamics that mean/cosine miss.")
    # Variant 2: success-side critical-point anchoring.
    p.add_argument("--success_anchor_mode", type=str, default="random",
                   choices=["random", "critical"],
                   help="How to choose success clip windows. "
                        "'random' (default) = sample N random reward-improving "
                        "windows per success episode (base ours behavior). "
                        "'critical' = single window centered on the success-side "
                        "critical_t = first-cross of `success_critical_threshold` "
                        "(Variant 2). Mirrors failure-side critical anchoring.")
    p.add_argument("--success_critical_threshold", type=float, default=0.75,
                   help="When --success_anchor_mode=critical, the smoothed "
                        "reward threshold used to define the success critical "
                        "frame. Defaults to 0.75 to match the rewind eval "
                        "success threshold.")
    p.add_argument("--output_manifest", type=str, required=True)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


CAMERA_KEYS = [
    "observation.images.agentview",
    "observation.images.eye_in_hand",
]


def _pool_last_k_frames(
    h: torch.Tensor, n_streams: int, clip_len: int, k: int
) -> torch.Tensor:
    """Per-stream mean over the LAST `k` frames of the transformer output.

    `h` is the stream-major transformer output of shape
    (B, n_streams * clip_len, D): tokens are laid out
    [stream0_t0..stream0_t(T-1), stream1_t0.., ...]. This regroups them
    correctly back to (B, n_streams, clip_len, D), averages each stream's
    last `k` frames, and concatenates the streams -> (B, n_streams * D).

    NOTE: this is deliberately NOT `h.view(B, T, -1)`. That reshape (used by
    the base model's reward head) reinterprets a stream-major sequence as if
    it were time-major, which scrambles streams across frames. Matching needs
    a clean per-stream/per-frame view, so we reshape with the real layout.
    """
    B, L, D = h.shape
    expected = n_streams * clip_len
    if L != expected:
        raise ValueError(
            f"h has {L} tokens but expected n_streams*clip_len={expected} "
            f"(n_streams={n_streams}, clip_len={clip_len})."
        )
    k = max(1, min(int(k), clip_len))
    h_streams = h.view(B, n_streams, clip_len, D)
    pooled = h_streams[:, :, -k:, :].mean(dim=2)        # (B, n_streams, D)
    return pooled.reshape(B, n_streams * D)


def _weighted_match_embedding(
    h: torch.Tensor,
    state: torch.Tensor,
    n_cam: int,
    clip_len: int,
    k: int,
    state_weight: float,
) -> torch.Tensor:
    """Weighted-block matching embedding: [state_trajectory | cam0 | cam1 | ...].

    - The STATE block is the flattened last-`k` robot-state sequence (a pose
      trajectory) -> clips match on how the robot moves.
    - Each CAMERA block is the last-`k` mean of that camera's transformer
      hidden states -> the cameras carry task stage. The language stream is
      dropped (it is the constant task string, useless for ranking).

    Every block is L2-normalized, then scaled by sqrt(weight) and
    concatenated. Because each normalized block is unit-norm, the plain
    cosine of two such embeddings then equals the weighted sum of per-block
    cosines:
        state_weight*cos(state) + cam_weight * sum_c cos(cam_c)
    with cam_weight = (1 - state_weight) / n_cam. So one scalar
    `state_weight` dials how much the match leans on robot motion.

    `h` is the stream-major transformer output (B, (n_cam+1)*clip_len, D);
    `state` is the (B, clip_len, state_dim) robot-state sequence.
    """
    if not (0.0 < state_weight < 1.0):
        raise ValueError(
            f"state_weight must be in (0, 1) for weighted matching, "
            f"got {state_weight}."
        )
    B, L, D = h.shape
    expected = (n_cam + 1) * clip_len
    if L != expected:
        raise ValueError(
            f"h has {L} tokens but expected (n_cam+1)*clip_len={expected} "
            f"(n_cam={n_cam}, clip_len={clip_len})."
        )
    k = max(1, min(int(k), clip_len))
    cam_weight = (1.0 - state_weight) / n_cam

    h_streams = h.view(B, n_cam + 1, clip_len, D)
    cam_blocks = h_streams[:, :n_cam, -k:, :].mean(dim=2)   # (B, n_cam, D)
    state_block = state[:, -k:, :].reshape(B, -1)           # (B, k*state_dim)

    def _unit(x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

    parts = [_unit(state_block) * (state_weight ** 0.5)]
    for c in range(n_cam):
        parts.append(_unit(cam_blocks[:, c]) * (cam_weight ** 0.5))
    return torch.cat(parts, dim=-1)


def _dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """Dynamic Time Warping distance between two trajectories.

    `seq_a`, `seq_b` are (T, dim) arrays of per-frame state vectors. Local
    cost is the Euclidean distance between frames; the return value is the
    accumulated cost of the optimal warping path. It is 0.0 for identical
    trajectories AND for ones that differ only by time-stretching (repeated
    frames) -- which is exactly why DTW captures motion shape rather than
    rigid frame-to-frame alignment.
    """
    a = np.asarray(seq_a, dtype=np.float64)
    b = np.asarray(seq_b, dtype=np.float64)
    ta, tb = a.shape[0], b.shape[0]
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    acc = np.full((ta + 1, tb + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, ta + 1):
        for j in range(1, tb + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(
                acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1]
            )
    return float(acc[ta, tb])


def _blend_match_scores(
    cos_sims: np.ndarray, dtw_dists: np.ndarray, dtw_weight: float
) -> np.ndarray:
    """Blend cosine similarity and DTW distance into a single match score.

    Both inputs are (n_query, n_bank). Per query row, cosine and DTW are
    each min-max normalized to [0, 1] so they are comparable, then combined
    (DTW inverted, since a smaller distance is a better match):
        score = (1 - w) * cosine_norm + w * (1 - dtw_norm)
    Higher score = better match. `w` is `dtw_weight`.
    """
    def _row_minmax(x: np.ndarray) -> np.ndarray:
        lo = x.min(axis=1, keepdims=True)
        hi = x.max(axis=1, keepdims=True)
        return (x - lo) / (hi - lo + 1e-12)

    w = float(dtw_weight)
    cos_n = _row_minmax(np.asarray(cos_sims, dtype=np.float64))
    dtw_n = _row_minmax(np.asarray(dtw_dists, dtype=np.float64))
    return (1.0 - w) * cos_n + w * (1.0 - dtw_n)


@torch.no_grad()
def _extract_clip_embedding(
    policy: RewindStatePolicy,
    raw_images: dict[str, torch.Tensor],
    state: torch.Tensor,
    lang_emb: torch.Tensor,
    match_window: int,
    match_state_weight: float,
) -> torch.Tensor:
    cfg = policy.config
    D = cfg.d_model

    batch = {**raw_images, "task.minilm": lang_emb}
    img_seq = policy._encode_images(batch)
    B, N_cam, T, _ = img_seq.shape

    s_proj = policy.state_mlp(state)
    vis_proj = torch.stack(
        [policy.visual_mlps[c](img_seq[:, c]) + s_proj for c in range(N_cam)],
        dim=1,
    )
    lang_proj = policy.lang_mlp(lang_emb)
    lang_proj = lang_proj.unsqueeze(1).expand(B, T, D)
    lang_proj = lang_proj + s_proj
    lang_proj = lang_proj.unsqueeze(1)

    x = torch.cat([vis_proj, lang_proj], dim=1)
    x[:, :N_cam, 0, :] += policy.first_pos
    x_tokens = x.view(B, (N_cam + 1) * T, D)
    L = x_tokens.size(1)

    device = x_tokens.device
    lengths = torch.tensor([T], dtype=torch.long, device=device)
    base_mask = torch.arange(T, device=device).expand(B, T) >= lengths.unsqueeze(1)
    mask = base_mask.unsqueeze(1).expand(B, N_cam + 1, T).reshape(B, (N_cam + 1) * T)
    causal_mask = torch.triu(
        torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1
    )
    h = policy.transformer(
        x_tokens, mask=causal_mask, src_key_padding_mask=mask, is_causal=True,
    )
    if match_state_weight <= 0.0:
        # Default: no state weighting -> per-camera (+lang) stage embedding.
        emb = _pool_last_k_frames(h, N_cam + 1, T, match_window)
    else:
        # Weighted blend of robot-state trajectory + per-camera stage blocks.
        emb = _weighted_match_embedding(
            h, state, N_cam, T, match_window, match_state_weight,
        )
    return emb.squeeze(0).detach().cpu()


def _slice_clip_batch(item: dict, start: int, clip_len: int, frame_stride: int) -> dict:
    idx = torch.tensor(strided_indices(start, clip_len, frame_stride), dtype=torch.long)
    out: dict = {}
    for cam in CAMERA_KEYS:
        out[cam] = item[cam][idx].unsqueeze(0)
    out["observation.state"] = item["observation.state"][idx].unsqueeze(0)
    out["task"] = [item["task"]]
    out["episode_index"] = torch.tensor([item["episode_index"]])
    return out


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    span = args.clip_len * args.frame_stride

    # 1. Load base policy + preprocessor.
    print(f"[1/5] Loading base policy from {args.base_ckpt}")
    policy_cfg = PreTrainedConfig.from_pretrained(args.base_ckpt)
    policy_cfg.pretrained_path = args.base_ckpt
    policy_cfg.device = args.device

    success_ds = EpisodeDataset(
        repo_id=args.success_repo_id,
        episodes=list(range(args.num_episodes)),
        camera_keys=CAMERA_KEYS,
        text_embedding_key=None,
        state_key="observation.state",
        use_precomputed_embeddings_only=False,
        pad_to_length=None,
    )
    failure_ds = EpisodeDataset(
        repo_id=args.failure_repo_id,
        episodes=list(range(args.num_episodes)),
        camera_keys=CAMERA_KEYS,
        text_embedding_key=None,
        state_key="observation.state",
        use_precomputed_embeddings_only=False,
        pad_to_length=None,
    )
    ds_meta = success_ds.meta

    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    policy.eval()
    policy.to(args.device)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        dataset_stats=ds_meta.stats if hasattr(ds_meta, "stats") else None,
    )

    # 2. Dense rewards on both sources.
    print(f"[2/5] Computing dense rewards (success={len(success_ds)}, "
          f"failure={len(failure_ds)})")
    rewards_per_source: dict[str, dict[int, np.ndarray]] = {"success": {}, "failure": {}}
    for label, ds in [("success", success_ds), ("failure", failure_ds)]:
        for i in tqdm(range(len(ds)), desc=label):
            item = ds[i]
            ep_idx = int(item["episode_index"])
            batch = {
                **{cam: item[cam].unsqueeze(0).to(args.device) for cam in CAMERA_KEYS},
                "observation.state": item["observation.state"].unsqueeze(0).to(args.device),
                "task": [item["task"]],
                "episode_index": torch.tensor([ep_idx], device=args.device),
            }
            batch = preprocessor(batch)
            raw_images = {k: batch[k] for k in CAMERA_KEYS}
            with torch.no_grad():
                pred = dense_predict_per_frame(
                    reward_model=policy,
                    raw_images=raw_images,
                    lang_emb=batch["task.minilm"],
                    state=batch.get("observation.state"),
                    max_window=MAX_WINDOW,
                    sampling="uniform",
                ).detach().float().cpu().numpy().astype(np.float32)
            rewards_per_source[label][ep_idx] = pred

    # 3. Failure critical clips + 4. success bank.
    print(f"[3/5] Building failure queries and success bank "
          f"(clip_len={args.clip_len}, frame_stride={args.frame_stride}, "
          f"span={span})")
    failure_queries: list[dict] = []   # one per failure ep
    success_bank: list[dict] = []      # many per success ep
    failure_query_embs: list[np.ndarray] = []
    success_bank_embs: list[np.ndarray] = []
    failure_query_states: list[np.ndarray] = []   # raw state seq per clip, for DTW
    success_bank_states: list[np.ndarray] = []

    for i in tqdm(range(len(failure_ds)), desc="failure queries"):
        item = failure_ds[i]
        ep_idx = int(item["episode_index"])
        ep_len = int(item[CAMERA_KEYS[0]].shape[0])
        if ep_len < span:
            continue
        rew = rewards_per_source["failure"][ep_idx]
        crit_t = critical_timestep(rew, args.smooth_window)
        start, end = centered_window(crit_t, span, ep_len)
        batch = preprocessor(_slice_clip_batch(item, start, args.clip_len, args.frame_stride))
        emb = _extract_clip_embedding(
            policy=policy,
            raw_images={k: batch[k] for k in CAMERA_KEYS},
            state=batch["observation.state"],
            lang_emb=batch["task.minilm"],
            match_window=args.match_window,
            match_state_weight=args.match_state_weight,
        ).numpy()
        failure_queries.append({
            "episode_index": ep_idx,
            "start": int(start),
            "end": int(end),
            "critical_t": int(crit_t),
        })
        failure_query_embs.append(emb)
        failure_query_states.append(
            batch["observation.state"][0].detach().cpu().numpy()
        )

    for i in tqdm(range(len(success_ds)), desc="success bank"):
        item = success_ds[i]
        ep_idx = int(item["episode_index"])
        ep_len = int(item[CAMERA_KEYS[0]].shape[0])
        if ep_len < span:
            continue
        rew_raw = rewards_per_source["success"][ep_idx]
        rew_smoothed = smooth(rew_raw, args.smooth_window)

        # Choose where to place clip(s) within this success episode.
        if args.success_anchor_mode == "critical":
            # Variant 2: single window centered on the success critical frame
            # (first crossing of the reward threshold). Mirrors failure-side
            # anchoring so that the last-K frames of failure and success clips
            # cover the same semantic moment (the insertion attempt).
            crit_t = success_critical_timestep(
                rew_raw, args.smooth_window, args.success_critical_threshold,
            )
            s_start, s_end = centered_window(crit_t, span, ep_len)
            starts = [s_start]
        else:
            # Base ours behavior: many random reward-improving windows.
            starts = sample_random_clip_starts(
                ep_len, span, args.success_clips_per_ep, rng
            )

        for s in starts:
            e = s + span
            r_start = float(rew_smoothed[s])
            r_end = float(rew_smoothed[e - 1])
            if r_end - r_start <= args.reward_delta_eps:
                continue
            batch = preprocessor(_slice_clip_batch(item, s, args.clip_len, args.frame_stride))
            emb = _extract_clip_embedding(
                policy=policy,
                raw_images={k: batch[k] for k in CAMERA_KEYS},
                state=batch["observation.state"],
                lang_emb=batch["task.minilm"],
                match_window=args.match_window,
                match_state_weight=args.match_state_weight,
            ).numpy()
            success_bank.append({
                "episode_index": ep_idx,
                "start": int(s),
                "end": int(e),
                "reward_delta": float(r_end - r_start),
            })
            success_bank_embs.append(emb)
            success_bank_states.append(
                batch["observation.state"][0].detach().cpu().numpy()
            )

    if not failure_queries:
        raise RuntimeError("No failure queries produced (every episode too short).")
    if not success_bank:
        raise RuntimeError(
            "Success bank is empty after reward-improvement filter. "
            "Try lowering --reward_delta_eps."
        )

    # 5. Match: top-K success bank entries per failure query.
    dtw_on = args.match_dtw_weight > 0.0
    metric_desc = (f"cosine+DTW blend (w={args.match_dtw_weight})"
                   if dtw_on else "cosine")
    print(f"[4/5] Matching {len(failure_queries)} queries against "
          f"{len(success_bank)} bank entries (top_k={args.top_k}, {metric_desc})")
    Q = np.stack(failure_query_embs).astype(np.float32)
    Bk = np.stack(success_bank_embs).astype(np.float32)
    Q /= (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    Bk /= (np.linalg.norm(Bk, axis=1, keepdims=True) + 1e-8)
    sims = Q @ Bk.T   # (Q, B) cosine similarity

    if dtw_on:
        # DTW distance on the raw robot-state trajectory of each clip.
        dtw = np.zeros_like(sims, dtype=np.float64)
        for qi in tqdm(range(len(failure_query_states)), desc="DTW"):
            sa = failure_query_states[qi]
            for bi in range(len(success_bank_states)):
                dtw[qi, bi] = _dtw_distance(sa, success_bank_states[bi])
        score = _blend_match_scores(sims, dtw, args.match_dtw_weight)
    else:
        dtw = None
        score = sims

    k = max(1, min(args.top_k, score.shape[1]))
    topk_idx = np.argpartition(-score, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(topk_idx.shape[0])[:, None]
    sorted_within = np.argsort(-score[rows, topk_idx], axis=1)
    topk_idx = topk_idx[rows, sorted_within]

    pairs: list[dict] = []
    for r, q in enumerate(failure_queries):
        candidates = []
        for c in range(k):
            bi = int(topk_idx[r, c])
            entry = success_bank[bi]
            cand = {
                "episode_index": entry["episode_index"],
                "start": entry["start"],
                "end": entry["end"],
                "cosine": float(sims[r, bi]),
                "reward_delta": float(entry["reward_delta"]),
            }
            if dtw_on:
                cand["dtw_distance"] = float(dtw[r, bi])
                cand["match_score"] = float(score[r, bi])
            candidates.append(cand)
        pairs.append({"failure": q, "success_candidates": candidates})

    # 6. Write the manifest.
    payload = {
        "base_ckpt": str(Path(args.base_ckpt).resolve()),
        "success_repo_id": args.success_repo_id,
        "failure_repo_id": args.failure_repo_id,
        "clip_len": int(args.clip_len),
        "frame_stride": int(args.frame_stride),
        "smooth_window": int(args.smooth_window),
        "match_window": int(args.match_window),
        "match_state_weight": float(args.match_state_weight),
        "match_dtw_weight": float(args.match_dtw_weight),
        "top_k": int(args.top_k),
        "pairs": pairs,
    }
    print(f"[5/5] Writing {len(pairs)} pairs to {args.output_manifest}")
    write_manifest(args.output_manifest, payload)
    print(f"  mean cosine over top-1: "
          f"{np.mean([p['success_candidates'][0]['cosine'] for p in pairs]):.4f}")
    print(f"  mean reward_delta over top-1: "
          f"{np.mean([p['success_candidates'][0]['reward_delta'] for p in pairs]):.4f}")


if __name__ == "__main__":
    main()
