from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("rewind_state_ours_variant5")
@dataclass
class RewindStateOursVariant5Config(PreTrainedConfig):
    # --- Architecture (same as RewindStateConfig) ---
    d_model: int = 512
    vis_emb_dim: int = 768
    text_emb_dim: int = 384
    state_dim: int = 19

    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1

    num_cameras: int = 3

    # --- Encoder fine-tuning (same as RewindStateConfig) ---
    finetune_encoder: bool = False
    finetune_encoder_blocks: int = 2
    encoder_ckpt: str = "facebook/dinov2-base"
    encoder_lr: float = 1e-5
    encoder_camera_keys: list[str] = field(default_factory=lambda: [
        "observation.images.front_rgb",
        "observation.images.wrist_rgb",
    ])

    # --- Training presets (same as RewindStateConfig) ---
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

    # --- Preference pair data pipeline (manifest must match these) ---
    preference_pairs_path: str = ""           # required at train time
    preference_clip_len: int = 16             # must match the manifest
    preference_frame_stride: int = 1          # must match the manifest
    preference_topk: int = 3                  # candidates to sample among
    preference_seed: int = 0                  # rng seed for the wrapper

    # --- Variant 5 (== Variant 3 + L1 TV regularizer on per-frame rewards) ---
    # BT objective: identical to Variant 3 — success/failure clips' mean sigmoid
    # reward over the last K frames, with optional stop-grad on success.
    reward_pref_loss_weight: float = 1.0      # peak weight after warmup
    reward_pref_warmup_steps: int = 500       # linear ramp 0 -> peak
    reward_pref_critical_window: int = 5      # K: # of trailing frames to average
    stop_grad_on_success: bool = True         # no_grad() the success forward; False = two-sided grad (ablation)

    # --- Total-Variation penalty on per-frame rewards within pref clips ---
    # L_TV(p) = mean_t |r(s_t) - r(s_{t-1})|^p, averaged across the (optionally
    # weighted) success and failure sides. `tv_order` selects p:
    #
    #   p = 1 (L1): jump-permissive. A 1-frame cliff and a 100-frame ramp of
    #       the same total drop cost the same. Penalizes oscillations and
    #       chatter; PRESERVES sharp cliffs. Use when failure trajectories
    #       have visible mid-clip wobble.
    #
    #   p = 2 (L2 / H1): quadratic. A 1-frame cliff costs ~10x more than the
    #       same drop spread over 10 frames. Strongly INCENTIVIZES spreading
    #       the BT-induced cliff into a ramp — better for PBRS credit
    #       assignment (negative shaping lands on the bad-action sequence
    #       rather than a single transition frame). Also kills chatter, but
    #       less directly than L1.
    #
    # Setting `tv_loss_weight = 0` recovers Variant 3 exactly regardless of order.
    tv_loss_weight: float = 0.05              # peak weight after warmup
    tv_warmup_steps: int = 500                # linear ramp 0 -> peak
    tv_order: int = 2                         # 1 = L1 (chatter), 2 = L2 (cliff)
    tv_on_success: bool = True                # apply TV to success clip too
    tv_on_failure: bool = True                # apply TV to failure clip

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
