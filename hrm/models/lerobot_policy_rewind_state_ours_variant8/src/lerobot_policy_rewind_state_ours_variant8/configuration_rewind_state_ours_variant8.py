from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("rewind_state_ours_variant8")
@dataclass
class RewindStateOursVariant8Config(PreTrainedConfig):
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

    # --- Variant 8: BT on per-step discounted weighted-mean of sigmoid rewards ---
    # The clip-level discriminand fed into the BT loss is a normalized,
    # γ-weighted sum:
    #     R(clip) = (Σ_{t=0}^{T-1} γ^(T-1-t) · r_t) / (Σ_{t=0}^{T-1} γ^(T-1-t))
    # so the last frame has weight 1 and earlier frames decay geometrically.
    # This replaces Variant 3 / Variant 5's hard trailing-K mean — the
    # discount γ supplies the temporal weighting smoothly across all frames.
    #
    # γ → 1: weighting approaches a uniform clip-mean (every frame matters
    #        equally); recovers global-mean-BT.
    # γ → 0: weighting collapses onto the final frame only; recovers a hard
    #        last-frame BT (more aggressive than trailing-K mean).
    # γ = 0.85 (default): effective horizon ~6 frames; late frames dominate
    #        the gradient while early frames retain ~10% weight, keeping the
    #        shared success/failure prefix anchored by the success MSE loss.
    reward_pref_loss_weight: float = 1.0      # peak weight after warmup
    reward_pref_warmup_steps: int = 500       # linear ramp 0 -> peak
    reward_pref_discount: float = 0.85        # γ for the per-step BT weighting
    stop_grad_on_success: bool = True         # no_grad() the success forward; False = two-sided grad (ablation)

    # Margin-bounded BT (in LOGIT space): clamps (succ_logit - fail_logit) at
    # `bt_target_margin` before the BT logsigmoid, so once the model achieves
    # a logit margin of `m` the loss stops decreasing and BT has no incentive
    # to push further. Prevents the unbounded-collapse failure mode where
    # fail saturates at sigmoid ≈ 0 to earn unbounded loss reduction.
    # Reference values: m=2 → σ(2)-σ(-2) ≈ 0.76 probability separation; m=3
    # → ≈ 0.91. Set very large (e.g. 100.0) to recover standard unbounded BT.
    bt_target_margin: float = 2.0

    # --- Monotonicity loss on per-frame sigmoid predictions within pref clips ---
    # Hinge penalty on the discrete time-derivative of r_t in sigmoid space:
    #   failure clips ("leaving" direction, frames start at critical_t):
    #     L_mono_fail = mean_t max(0, r(t+1) - r(t))     # penalize INCREASES
    #     → enforces a monotone decrease over the post-failure window
    #   success clips:
    #     L_mono_succ = mean_t max(0, r(t) - r(t+1))     # penalize DECREASES
    #     → enforces a monotone increase (progress should rise)
    #
    # This directly encodes the desired SHAPE (rise on success, fall on
    # failure post-critical_t) without manufacturing absolute progress
    # labels — it's a pairwise relative constraint on adjacent frames in
    # the same trajectory. Composes cleanly with MSE and BT.
    #
    # CAVEAT (forwardbackward manifests): when failure clips are emitted in
    # the "approaching" direction (clip ENDS at critical_t, pre-failure
    # region), monotone-decrease supervision is WRONG — those frames should
    # be increasing. For forwardbackward runs, set mono_on_failure=False or
    # rebuild the manifest with only the "leaving" direction.
    mono_loss_weight: float = 1.0             # peak weight after warmup
    mono_warmup_steps: int = 500              # linear ramp 0 -> peak
    mono_on_failure: bool = True              # apply monotone-DECREASE on failure clips
    mono_on_success: bool = True              # apply monotone-INCREASE on success clips
    # Minimum per-step descent in LOGIT space, applied to "leaving"-direction
    # failure pairs (post-critical clips). The hinge becomes:
    #   max(0, r(t+1) - r(t) + mono_step_eps)
    # eps=0 only prevents increases (flat curves trivially OK). eps>0 forces
    # actual descent of at least `eps` logit-units per step. With 8 anchor
    # frames and eps=0.3, total descent ≈ 2.1 logits ≈ σ(1)→σ(-1.1) ≈
    # 0.73→0.25 in probability space.
    mono_step_eps: float = 0.0
    # Same idea for "approaching"-direction failure pairs (pre-critical
    # clips). Usually set to 0 (just prevent increases) since natural rewind
    # progress on these frames is RISING — strict descent would conflict
    # too hard with MSE. Only consulted when the dataset emits a per-batch
    # `_is_approaching` tag (i.e. forwardbackward manifests).
    mono_step_eps_approaching: float = 0.0

    # --- Total-variation regularizer (optional smoothness penalty) ---
    # L_TV(p) = mean_t |r(t+1) - r(t)|^p on per-anchor (failure) or per-frame
    # (success) LOGITS. Adds smoothness beyond what MSE on rewind already
    # provides — useful when the supervised window leaves the trunk free to
    # produce jumpy predictions. Set tv_loss_weight=0 (default) to disable.
    #
    #   p = 1 (L1): jump-permissive. Penalizes oscillations / chatter but
    #     lets a single sharp cliff through unchanged.
    #   p = 2 (L2 / H1): quadratic. Penalizes large frame-to-frame jumps
    #     strongly; encourages spreading any drop into a smooth ramp.
    tv_loss_weight: float = 0.0
    tv_warmup_steps: int = 500
    tv_order: int = 2
    tv_on_failure: bool = True
    tv_on_success: bool = True

    # --- Hybrid failure-clip sampling (computed at __getitem__ time) ---
    # "manifest" (default): load the failure clip using the indices stored in
    #   the precomputed manifest (the original ±65-frame window around
    #   critical_t, uniformly subsampled to 16 frames).
    # "hybrid": ignore the manifest's `subsampled_indices` for the failure
    #   side; instead, load 8 uniform frames from [0, raw_end - n_recent) +
    #   8 contiguous frames at [raw_end - n_recent, raw_end). The clip then
    #   spans the entire pre-failure trajectory (rising-progress prefix) plus
    #   a dense post-critical tail. Success-side sampling stays at the
    #   manifest indices to preserve match alignment.
    #
    # When `failure_sampling_mode="hybrid"`, mono on the failure clip becomes
    # SEGMENTED:
    #   - First (clip_len - failure_hybrid_n_recent) frames → mono-INCREASE
    #     (the rising prefix).
    #   - Last failure_hybrid_n_recent frames → mono-DECREASE (the failing
    #     suffix). This encodes the full rise-then-fall shape.
    # Variant 8 default: hybrid (variant 6 defaulted to "manifest"). Required
    # for `mono_skip_pre_critical_anchors` to do anything.
    failure_sampling_mode: str = "hybrid"
    failure_hybrid_n_recent: int = 8

    # --- Variant 8: skip mono-decrease on pre-critical failure anchors ---
    # In `failure_sampling_mode="hybrid"`, the anchor trajectory starts a few
    # frames BEFORE critical_t (e.g. critical_t-5 ... critical_t+60). Those
    # leading "pre-critical" anchors are visually/dynamically indistinguishable
    # from on-task success frames at the same timestep, so the success-side
    # per-frame supervision generalizes to predict HIGH reward on them.
    # Applying the mono-DECREASE hinge uniformly conflicts with that signal:
    # it pushes predicted reward down on states the success loss is keeping
    # high, producing either premature descent or a non-vanishing mono-loss
    # floor.
    #
    # `mono_skip_pre_critical_anchors = k` drops the FIRST `k` adjacent-anchor
    # diffs from the failure mono-hinge (i.e. the diffs spanning anchors
    # 0→1, 1→2, ..., (k-1)→k). The (k-1)→k diff is also dropped so the
    # "knee" at critical_t isn't forced to step down by `mono_step_eps`;
    # decrease then kicks in from anchor k → k+1 onward.
    #
    # 0 = legacy variant-6 behaviour (apply mono uniformly across anchors).
    # Default = 2 matches the precompute schedule's pre-critical anchor count
    # for direction="leaving" under the default precompute args
    # (pre_critical_frames=5, max_post_critical_frames=60, clip_len=16):
    # np.linspace(crit_t-5, crit_t+60, 16) lands anchors 0 and 1 before
    # crit_t (at offsets -5 and -1), then anchor 2 at +4. Skip the first
    # k=2 diffs (0→1, 1→2) so neither pre-critical pair nor the knee at
    # crit_t is forced to descend by mono_step_eps.
    # If you change the precompute window or clip_len, recompute this:
    #   k = sum(1 for i in subsampled_indices if i < critical_t)
    # No effect when `failure_sampling_mode != "hybrid"`.
    mono_skip_pre_critical_anchors: int = 2

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
