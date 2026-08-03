from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig

from .rewind_datasets import RewindDataset, EpisodeDataset


# Camera-specific embedding keys for the robomimic_square dataset variants.
# Only used when finetune_encoder=False (i.e. the dataset must carry precomputed
# DINOv2 features under these keys).
ROBOMIMIC_EMBEDDING_KEYS = [
    "observation.dinov2.agentview",
    "observation.dinov2.eye_in_hand",
]


def make_dataset(cfg: TrainPipelineConfig | DatasetConfig) -> RewindDataset:
    if isinstance(cfg, DatasetConfig):
        cfg.dataset = cfg  # type: ignore

    finetune_encoder = getattr(getattr(cfg, "policy", None), "finetune_encoder", False)
    camera_keys = None
    if finetune_encoder:
        camera_keys = list(cfg.policy.encoder_camera_keys)

    # When finetuning the encoder, the model encodes raw images online and the
    # MiniLMTextEmbeddingProcessorStep encodes the task string at runtime, so
    # neither precomputed DINOv2 features nor `task.minilm` are required.
    embedding_keys = [] if finetune_encoder else ROBOMIMIC_EMBEDDING_KEYS
    text_embedding_key = None if finetune_encoder else "task.minilm"

    dataset = RewindDataset(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        embedding_keys=embedding_keys,
        camera_keys=camera_keys,
        text_embedding_key=text_embedding_key,
        use_precomputed_embeddings_only=not finetune_encoder,
        rewind=True,
        subsample_video=True,
        pad_to_length=16,
    )

    return dataset


def make_eval_dataset(cfg: TrainPipelineConfig | DatasetConfig) -> EpisodeDataset:
    if isinstance(cfg, DatasetConfig):
        cfg.dataset = cfg  # type: ignore

    finetune_encoder = getattr(getattr(cfg, "policy", None), "finetune_encoder", False)
    embedding_keys = [] if finetune_encoder else ROBOMIMIC_EMBEDDING_KEYS
    text_embedding_key = None if finetune_encoder else "task.minilm"

    dataset = EpisodeDataset(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        embedding_keys=embedding_keys,
        text_embedding_key=text_embedding_key,
        use_precomputed_embeddings_only=False,
        pad_to_length=16,
    )

    return dataset


def make_eval_original_dataset(cfg: TrainPipelineConfig | DatasetConfig) -> EpisodeDataset:
    if isinstance(cfg, DatasetConfig):
        cfg.dataset = cfg  # type: ignore

    dataset = EpisodeDataset(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        use_precomputed_embeddings_only=False,
        pad_to_length=None,
    )

    return dataset
