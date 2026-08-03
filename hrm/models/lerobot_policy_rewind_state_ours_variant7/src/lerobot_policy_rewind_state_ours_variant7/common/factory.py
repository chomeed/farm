"""Generic factory for the rewind_state_ours dataset."""
from __future__ import annotations

from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig

from .preference_datasets import PreferenceTaggedRewindDataset


def make_dataset(cfg: TrainPipelineConfig | DatasetConfig) -> PreferenceTaggedRewindDataset:
    if isinstance(cfg, DatasetConfig):
        cfg.dataset = cfg  # type: ignore

    finetune_encoder = bool(getattr(getattr(cfg, "policy", None), "finetune_encoder", False))
    camera_keys = list(cfg.policy.encoder_camera_keys) if finetune_encoder else None

    return PreferenceTaggedRewindDataset(
        repo_id=cfg.dataset.repo_id,
        root=getattr(cfg.dataset, "root", None),
        camera_keys=camera_keys,
        text_embedding_key=None if finetune_encoder else "task.minilm",
        use_precomputed_embeddings_only=not finetune_encoder,
        rewind=True,
        subsample_video=True,
        pad_to_length=16,
        preference_pairs_path=cfg.policy.preference_pairs_path,
        preference_clip_len=int(cfg.policy.preference_clip_len),
        preference_frame_stride=int(cfg.policy.preference_frame_stride),
        preference_topk=int(cfg.policy.preference_topk),
        preference_seed=int(getattr(cfg.policy, "preference_seed", 0)),
        preference_assert_base_ckpt=getattr(cfg.policy, "pretrained_path", None) or None,
        failure_sampling_mode=str(getattr(cfg.policy, "failure_sampling_mode", "manifest")),
        failure_hybrid_n_recent=int(getattr(cfg.policy, "failure_hybrid_n_recent", 8)),
        success_sampling_mode=str(getattr(cfg.policy, "success_sampling_mode", "hybrid")),
        success_hybrid_n_recent=int(getattr(cfg.policy, "success_hybrid_n_recent", 8)),
    )
