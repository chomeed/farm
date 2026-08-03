from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


@PreTrainedConfig.register_subclass("rewind_state")
@dataclass
class RewindStateConfig(PreTrainedConfig):

    d_model: int = 512
    vis_emb_dim: int = 768   # DINOv2 base image embedding dimension
    text_emb_dim: int = 384  # MiniLM text embedding dimension
    state_dim: int = 19      # default for peg insertion; override per task

    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1

    num_cameras: int = 3

    # Encoder fine-tuning
    finetune_encoder: bool = False
    finetune_encoder_blocks: int = 2
    encoder_ckpt: str = "facebook/dinov2-base"
    encoder_lr: float = 1e-5
    encoder_camera_keys: list[str] = field(default_factory=lambda: [
        "observation.images.front_rgb",
        "observation.images.wrist_rgb",
    ])

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6
    scheduler_peak_lr: float = 1e-4

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "LANGUAGE": NormalizationMode.IDENTITY,
            "REWARD": NormalizationMode.IDENTITY,
        }
    )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.scheduler_peak_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    def validate_features(self) -> None:
        pass

    @property
    def observation_delta_indices(self) -> list[int]:
        return [0, 10, 20, 30]

    @property
    def action_delta_indices(self) -> list:
        return [0]

    @property
    def reward_delta_indices(self):
        return None
