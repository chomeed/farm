from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig

from .rewind_datasets import RewindDataset, EpisodeDataset

def make_dataset(cfg: TrainPipelineConfig | DatasetConfig) -> RewindDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        RewindDataset
    """

    if isinstance(cfg, DatasetConfig):
        cfg.dataset = cfg  # type: ignore

    finetune_encoder = getattr(getattr(cfg, "policy", None), "finetune_encoder", False)
    camera_keys = None
    if finetune_encoder:
        camera_keys = list(cfg.policy.encoder_camera_keys)

    dataset = RewindDataset(
        repo_id=cfg.dataset.repo_id,
        # peg insertion
        embedding_keys=[
            "observation.dinov2.front_rgb",
            "observation.dinov2.wrist_rgb",
        ],
        # anubis
        # embedding_keys=[
        #     "observation.dinov2.front",
        #     "observation.dinov2.wrist_left",
        #     "observation.dinov2.wrist_right",
        # ],
        camera_keys=camera_keys,
        text_embedding_key="task.minilm",
        use_precomputed_embeddings_only=not finetune_encoder,
        rewind=True,
        subsample_video=True,
        pad_to_length=16,
    )

    return dataset



def make_eval_dataset(cfg: TrainPipelineConfig | DatasetConfig) -> EpisodeDataset:
    """Creates a RewindDataset for evaluation. This dataset does not subsample videos and does not pad to a fixed length.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Returns:
        EpisodeDataset
    """

    if isinstance(cfg, DatasetConfig):
        cfg.dataset = cfg  # type: ignore

    dataset = EpisodeDataset(
        repo_id=cfg.dataset.repo_id,
        # peg insertion
        embedding_keys=[
            "observation.dinov2.front_rgb",
            "observation.dinov2.wrist_rgb",
        ],
        # anubis
        # embedding_keys=[
        #     "observation.dinov2.front",
        #     "observation.dinov2.wrist_left",
        #     "observation.dinov2.wrist_right",
        # ],
        text_embedding_key="task.minilm",
        use_precomputed_embeddings_only=False,
        pad_to_length=16,
    )

    return dataset


def make_eval_original_dataset(cfg: TrainPipelineConfig | DatasetConfig) -> EpisodeDataset:
    """
    set pad_to_length=None so EpisodeDataset returns original-length episodes
    """
    if isinstance(cfg, DatasetConfig):
        cfg.dataset = cfg  # type: ignore

    # Raw images are sufficient when the model's preprocessor does its own text
    # embedding (MiniLMTextEmbeddingProcessorStep) and, in finetune_encoder=True
    # mode, its own image encoding. Don't require precomputed features the
    # dataset may not carry.
    dataset = EpisodeDataset(
        repo_id=cfg.dataset.repo_id,
        use_precomputed_embeddings_only=False,
        pad_to_length=None,
    )

    return dataset