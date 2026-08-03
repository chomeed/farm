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
  6. compute mean-pooled hidden-state embedding.
For each failure critical clip:
  7. take top-K success matches by cosine.
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
    p.add_argument("--output_manifest", type=str, required=True)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


CAMERA_KEYS = [
    "observation.images.agentview",
    "observation.images.eye_in_hand",
]


@torch.no_grad()
def _extract_clip_embedding(
    policy: RewindStatePolicy,
    raw_images: dict[str, torch.Tensor],
    state: torch.Tensor,
    lang_emb: torch.Tensor,
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
    flattened = h.view(B, T, -1)
    return flattened.mean(dim=1).squeeze(0).detach().cpu()


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
        ).numpy()
        failure_queries.append({
            "episode_index": ep_idx,
            "start": int(start),
            "end": int(end),
            "critical_t": int(crit_t),
        })
        failure_query_embs.append(emb)

    for i in tqdm(range(len(success_ds)), desc="success bank"):
        item = success_ds[i]
        ep_idx = int(item["episode_index"])
        ep_len = int(item[CAMERA_KEYS[0]].shape[0])
        if ep_len < span:
            continue
        rew_smoothed = smooth(rewards_per_source["success"][ep_idx], args.smooth_window)
        starts = sample_random_clip_starts(ep_len, span, args.success_clips_per_ep, rng)
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
            ).numpy()
            success_bank.append({
                "episode_index": ep_idx,
                "start": int(s),
                "end": int(e),
                "reward_delta": float(r_end - r_start),
            })
            success_bank_embs.append(emb)

    if not failure_queries:
        raise RuntimeError("No failure queries produced (every episode too short).")
    if not success_bank:
        raise RuntimeError(
            "Success bank is empty after reward-improvement filter. "
            "Try lowering --reward_delta_eps."
        )

    # 5. Match: top-K success bank entries per failure query, by cosine.
    print(f"[4/5] Matching {len(failure_queries)} queries against "
          f"{len(success_bank)} bank entries (top_k={args.top_k})")
    Q = np.stack(failure_query_embs).astype(np.float32)
    Bk = np.stack(success_bank_embs).astype(np.float32)
    Q /= (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    Bk /= (np.linalg.norm(Bk, axis=1, keepdims=True) + 1e-8)
    sims = Q @ Bk.T   # (Q, B)

    k = max(1, min(args.top_k, sims.shape[1]))
    topk_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(topk_idx.shape[0])[:, None]
    sorted_within = np.argsort(-sims[rows, topk_idx], axis=1)
    topk_idx = topk_idx[rows, sorted_within]
    topk_sims = sims[rows, topk_idx]

    pairs: list[dict] = []
    for r, q in enumerate(failure_queries):
        candidates = []
        for c in range(k):
            bi = int(topk_idx[r, c])
            entry = success_bank[bi]
            candidates.append({
                "episode_index": entry["episode_index"],
                "start": entry["start"],
                "end": entry["end"],
                "cosine": float(topk_sims[r, c]),
                "reward_delta": float(entry["reward_delta"]),
            })
        pairs.append({"failure": q, "success_candidates": candidates})

    # 6. Write the manifest.
    payload = {
        "base_ckpt": str(Path(args.base_ckpt).resolve()),
        "success_repo_id": args.success_repo_id,
        "failure_repo_id": args.failure_repo_id,
        "clip_len": int(args.clip_len),
        "frame_stride": int(args.frame_stride),
        "smooth_window": int(args.smooth_window),
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
