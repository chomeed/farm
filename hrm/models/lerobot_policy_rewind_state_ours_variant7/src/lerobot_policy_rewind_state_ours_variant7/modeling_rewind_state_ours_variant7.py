"""
Rewind reward + Variant 7 preference learning (`rewind_state_ours_variant7`).

Variant 7 swaps Variant 3 / Variant 5's hard trailing-K BT for a
**per-step discounted** BT objective. Every frame in the preference clip
contributes to the BT discriminand with weight γ^(T-1-t), so the last
frame has full weight and earlier frames decay geometrically. There is
no hard window cutoff; γ alone controls the temporal emphasis.

Heads:
  * `fusion_net` - per-frame reward head. Trained with:
                     (i) MSE on per-frame reward labels (rewind sample),
                     (ii) BT loss on the discounted weighted-mean of
                          LOGITS over the full preference clip (success
                          vs failure), with the margin clamped at
                          `bt_target_margin`. Optional stop-grad on
                          success.
                     (iii) Monotonicity hinge on the sigmoid-space
                          per-frame predictions within each pref clip:
                          failure clips (leaving direction) penalized for
                          r(t+1) > r(t); success clips penalized for
                          r(t+1) < r(t). Encodes the rise-then-fall SHAPE
                          via pairwise relative constraints on adjacent
                          frames — no absolute progress labels needed.

Why per-step discount (vs trailing-K mean):
  * Mean-of-K is shape-blind: the model can chatter freely as long as the
    K-frame average is correct (motivation for Variant 5's TV).
  * Per-step discount supervises every frame with weight γ^(T-1-t),
    removing the "free chatter" zone while keeping the temporal emphasis
    on late frames. γ → 1 recovers uniform-mean BT; γ → 0 collapses to
    last-frame BT.

Inference unchanged: `predict_reward(batch)` returns the sigmoid'd [0, 1]
progress signal from `fusion_net`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_rewind_state_ours_variant7 import RewindStateOursVariant7Config

_DINO_MEAN = [0.485, 0.456, 0.406]
_DINO_STD = [0.229, 0.224, 0.225]


def _make_mlp(in_dim: int, d_model: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, d_model),
        nn.GELU(),
        nn.Linear(d_model, d_model),
    )


class RewindStateOursVariant7Policy(PreTrainedPolicy):
    config_class = RewindStateOursVariant7Config
    name = "rewind_state_ours_variant7"

    def __init__(
        self,
        config: RewindStateOursVariant7Config,
        dataset_stats: dict | None = None,
        dataset_meta: dict | None = None,
    ):
        super().__init__(config)
        self.d_model = config.d_model
        self.num_cameras = config.num_cameras
        self.dataset_stats = dataset_stats
        self.dataset_meta = dataset_meta

        if config.finetune_encoder:
            from transformers import AutoModel
            self.vision_encoder = AutoModel.from_pretrained(
                config.encoder_ckpt, attn_implementation="eager"
            )
            for p in self.vision_encoder.parameters():
                p.requires_grad_(False)
            # NB: encoder.layer[-0:] == encoder.layer[0:] == ALL layers, so we
            # have to gate the unfreeze on a strictly positive block count.
            if config.finetune_encoder_blocks > 0:
                for block in self.vision_encoder.encoder.layer[-config.finetune_encoder_blocks:]:
                    for p in block.parameters():
                        p.requires_grad_(True)
                for p in self.vision_encoder.layernorm.parameters():
                    p.requires_grad_(True)

            n_trainable = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in self.vision_encoder.parameters())
            print(f"DINOv2 encoder: {n_trainable:,} / {n_total:,} params trainable "
                  f"(last {config.finetune_encoder_blocks} blocks)")

        self.visual_mlps = nn.ModuleList([
            _make_mlp(config.vis_emb_dim, config.d_model)
            for _ in range(config.num_cameras)
        ])
        self.lang_mlp = _make_mlp(config.text_emb_dim, config.d_model)
        self.state_mlp = _make_mlp(config.state_dim, config.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            config.d_model,
            config.n_heads,
            4 * config.d_model,
            config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, config.n_layers)

        self.first_pos = nn.Parameter(torch.zeros(1, config.d_model))

        # Reward head (matches base RewindStatePolicy.fusion_net exactly so that
        # weights from the 010000 checkpoint load by name).
        self.fusion_net = nn.Sequential(
            nn.LayerNorm(config.d_model * (config.num_cameras + 1)),
            nn.Linear(config.d_model * (config.num_cameras + 1), config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
        )

    # --- inference ---

    @torch.no_grad()
    def predict_reward(self, batch: dict[str, Tensor]) -> Tensor:
        return self._predict_rewards(batch)

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        loss, _ = self.forward(batch)
        return loss

    # --- internals ---

    def _encode_images(self, batch: dict[str, Tensor]) -> Tensor:
        """Online DINOv2 encoding using CLS-attention weighted patch pooling.
        Returns (B, N, T, vis_emb_dim).
        """
        imgs = torch.stack(
            [batch[k] for k in self.config.encoder_camera_keys], dim=1
        )
        B, N, T, C, H, W = imgs.shape
        imgs_flat = imgs.view(B * N * T, C, H, W)

        mean = torch.tensor(_DINO_MEAN, device=imgs_flat.device).view(1, 3, 1, 1)
        std = torch.tensor(_DINO_STD, device=imgs_flat.device).view(1, 3, 1, 1)
        imgs_flat = (imgs_flat - mean) / std

        outputs = self.vision_encoder(pixel_values=imgs_flat, output_attentions=True)

        patch_tokens = outputs.last_hidden_state[:, 1:]
        attn = outputs.attentions[-1][:, :, 0, 1:].mean(dim=1)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        embs = (patch_tokens * attn.unsqueeze(-1)).sum(dim=1)
        return embs.view(B, N, T, -1)

    def _trunk(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Run the shared transformer trunk and return (flattened_h, lengths).

        flattened_h has shape (B, T, (N+1)*D) and is the per-timestep concat of
        all (N+1) modality tokens, ready for either head.
        """
        if self.config.finetune_encoder:
            img_seq = self._encode_images(batch)
        else:
            img_seq = batch["img_seq"]
        lang_emb = batch["task.minilm"]
        state = batch["observation.state"]
        lengths = batch.get("lengths")

        B, N, T, _ = img_seq.shape
        D = self.d_model
        device = img_seq.device

        s_proj = self.state_mlp(state)

        vis_proj = torch.stack(
            [self.visual_mlps[c](img_seq[:, c]) + s_proj for c in range(N)],
            dim=1,
        )

        lang_proj = self.lang_mlp(lang_emb)
        lang_proj = lang_proj.unsqueeze(1).expand(B, T, D)
        lang_proj = lang_proj + s_proj
        lang_proj = lang_proj.unsqueeze(1)

        x = torch.cat([vis_proj, lang_proj], dim=1)
        x[:, :N, 0, :] += self.first_pos

        x_tokens = x.view(B, (N + 1) * T, D)
        L = x_tokens.size(1)

        base_mask = torch.arange(T, device=device).expand(B, T) >= lengths.unsqueeze(1)
        mask = base_mask.unsqueeze(1).expand(B, N + 1, T).reshape(B, (N + 1) * T)
        causal_mask = torch.triu(
            torch.ones(L, L, device=device, dtype=torch.bool),
            diagonal=1,
        )

        h = self.transformer(
            x_tokens,
            mask=causal_mask,
            src_key_padding_mask=mask,
            is_causal=True,
        )

        flattened_h = h.view(B, T, -1)
        return flattened_h, lengths

    def _predict_reward_logits(self, batch: dict[str, Tensor]) -> Tensor:
        """Per-frame raw fusion_net output (pre-sigmoid), shape (B, T).

        BT operates on these to give an unbounded margin space where the
        `bt_target_margin` clamp can meaningfully cap collapse. Sigmoid'd
        rewards (the [0, 1] form used for MSE and downstream RL) live in
        [-1, 1] margin space, so a clamp there would only activate at
        full saturation — useless.
        """
        flattened_h, _ = self._trunk(batch)
        return self.fusion_net(flattened_h).squeeze(-1)

    def _predict_rewards(self, batch: dict[str, Tensor]) -> Tensor:
        """Per-frame reward in [0, 1], shape (B, T). Used for MSE + inference."""
        return torch.sigmoid(self._predict_reward_logits(batch))

    def _reward_pref_weight(self, step: int) -> float:
        """Variant 7: warmup schedule for the BT-with-discount loss."""
        w_max = self.config.reward_pref_loss_weight
        warm = self.config.reward_pref_warmup_steps
        if warm <= 0:
            return w_max
        return w_max * min(1.0, step / warm)

    def _mono_weight(self, step: int) -> float:
        """Variant 7: warmup schedule for the monotonicity loss."""
        w_max = self.config.mono_loss_weight
        warm = self.config.mono_warmup_steps
        if warm <= 0:
            return w_max
        return w_max * min(1.0, step / warm)

    def _tv_weight(self, step: int) -> float:
        """Variant 7: warmup schedule for the TV smoothness regularizer."""
        w_max = self.config.tv_loss_weight
        warm = self.config.tv_warmup_steps
        if warm <= 0:
            return w_max
        return w_max * min(1.0, step / warm)

    def _succ_anchor_mse_weight(self, step: int) -> float:
        """Variant 7: warmup schedule for the success-anchor MSE."""
        w_max = self.config.succ_anchor_mse_weight
        warm = self.config.succ_anchor_mse_warmup_steps
        if warm <= 0:
            return float(w_max)
        return float(w_max) * min(1.0, step / float(warm))

    @staticmethod
    def _tv_loss(r_seq: Tensor, order: int) -> Tensor:
        """Total-variation penalty on the per-step time derivative of r_seq.

        order=1: mean |diff|        — chatter-killing, jump-permissive
        order=2: mean |diff|² (H1)  — penalizes large single jumps strongly
        """
        diffs = r_seq[:, 1:] - r_seq[:, :-1]   # (B, T-1)
        if order == 1:
            return diffs.abs().mean()
        if order == 2:
            return diffs.square().mean()
        raise ValueError(f"tv_order must be 1 or 2, got {order!r}")

    @staticmethod
    def _mono_hinge(
        r_seq: Tensor,
        decreasing: bool,
        eps: float | Tensor = 0.0,
    ) -> Tensor:
        """Hinge penalty on the per-frame time derivative of r_seq (B, T).

        decreasing=True:  penalize diff > -eps → enforces diff <= -eps.
        decreasing=False: penalize diff <  +eps → enforces diff >= +eps.

        `eps` may be a scalar (same threshold for all batch elements) or a
        tensor of shape (B,) (per-batch-element threshold, broadcast across
        the T-1 adjacent-frame pairs).
        """
        diffs = r_seq[:, 1:] - r_seq[:, :-1]   # (B, T-1)
        if isinstance(eps, Tensor):
            eps = eps.to(diffs.dtype).unsqueeze(1)   # (B, 1)
        if decreasing:
            return F.relu(diffs + eps).mean()
        return F.relu(eps - diffs).mean()

    def _discount_weights(self, T: int, device, dtype) -> Tensor:
        """γ^(T-1-t) for t=0..T-1, so the LAST frame has weight 1."""
        gamma = float(self.config.reward_pref_discount)
        if not (0.0 < gamma <= 1.0):
            raise ValueError(
                f"reward_pref_discount must be in (0, 1], got {gamma!r}"
            )
        exps = torch.arange(T - 1, -1, -1, device=device, dtype=dtype)
        return gamma ** exps   # (T,)

    def _discounted_clip_score(self, r_seq: Tensor) -> Tensor:
        """Variant 7: γ-weighted normalized mean over per-step values.

        Given `r_seq` of shape (B, T), returns (B,):
            R = Σ_t w_t · r_t  /  Σ_t w_t,    w_t = γ^(T-1-t)
        Normalization keeps the scale comparable to a plain mean so the
        BT logit magnitude doesn't change discontinuously with T.
        Accepts either sigmoid'd rewards or raw logits; BT uses logits.
        """
        T = r_seq.size(1)
        w = self._discount_weights(T, r_seq.device, r_seq.dtype)   # (T,)
        return (r_seq * w).sum(dim=1) / w.sum()

    _PREF_FAIL_PREFIX = "pref.failure."
    _PREF_SUCC_PREFIX = "pref.success."

    def _extract_pref_subbatch(
        self, batch: dict[str, Tensor], prefix: str
    ) -> dict[str, Tensor]:
        """Strip `prefix` and assemble a normal observation-style sub-batch.

        Pref clips are always full `preference_clip_len` long (no padding,
        no rewind), so we synthesize `lengths` rather than threading it
        through the preprocessor pipeline.

        For `task.minilm` we reuse the un-prefixed entry (single-task
        invariant from the spec) when the prefixed key is absent — the
        MiniLM step in the preprocessor only encodes the rewind sample's
        `task` string, not the namespaced `pref.failure.task` /
        `pref.success.task` strings.
        """
        out: dict[str, Tensor] = {}
        for k, v in batch.items():
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
        if "task.minilm" not in out and "task.minilm" in batch:
            out["task.minilm"] = batch["task.minilm"]

        # Synthesize `lengths` from any pref tensor's time dim. Pref clips
        # are always full clip_len, so a constant tensor is correct.
        if "lengths" not in out:
            ref_t = self._infer_pref_clip_len(out)
            if ref_t is None:
                raise KeyError(
                    f"Preference sub-batch (prefix={prefix!r}) has no tensor "
                    f"from which to infer clip length. Available: "
                    f"{sorted(out.keys())}"
                )
            B, T = ref_t.shape[:2]
            out["lengths"] = torch.full(
                (B,), T, dtype=torch.long, device=ref_t.device,
            )

        # Validate required keys depending on mode.
        # finetune_encoder=True: model encodes images itself, needs raw
        #   observation.images.* tensors, no img_seq.
        # finetune_encoder=False: model expects pre-encoded img_seq.
        if self.config.finetune_encoder:
            cams = list(self.config.encoder_camera_keys)
            missing_cams = [c for c in cams if c not in out]
            if missing_cams:
                raise KeyError(
                    f"Preference sub-batch (prefix={prefix!r}) missing "
                    f"camera tensors: {missing_cams}. "
                    f"Available: {sorted(out.keys())}"
                )
            needed = ("task.minilm", "observation.state", "lengths")
        else:
            needed = ("img_seq", "task.minilm", "observation.state", "lengths")

        missing = [k for k in needed if k not in out]
        if missing:
            raise KeyError(
                f"Preference sub-batch (prefix={prefix!r}) is missing keys: "
                f"{missing}. Available: {sorted(out.keys())}"
            )
        return out

    def _infer_pref_clip_len(self, sub_batch: dict[str, Tensor]) -> Tensor | None:
        """Return any tensor in `sub_batch` we can read T from."""
        # Prefer img_seq (frozen-encoder path) since its layout is fixed.
        if "img_seq" in sub_batch:
            return sub_batch["img_seq"]   # (B, N, T, D)
        # Fall back to a camera tensor (finetune_encoder path).
        for cam in self.config.encoder_camera_keys:
            if cam in sub_batch:
                return sub_batch[cam]     # (B, T, C, H, W)
        # Last resort: observation.state.
        if "observation.state" in sub_batch:
            return sub_batch["observation.state"]   # (B, T, state_dim)
        return None

    def _predict_failure_anchor_logits(self, batch: dict[str, Tensor]) -> Tensor:
        """Anchor-stack forward for the failure clip.

        With `failure_sampling_mode="hybrid"` the dataset emits one hybrid
        clip per manifest anchor, stacked along a new dim 0:
            observation.images.<cam>: (B, N_anchors, T_clip, C, H, W)
            observation.state:        (B, N_anchors, T_clip, state_dim)
            img_seq:                  (B, N_anchors, N_cams, T_clip, vis_dim)
            task.minilm:              (B, text_dim)  (same across anchors)

        This method flattens the anchor dim into the batch dim, runs a
        SINGLE batched forward through trunk + fusion_net producing
        (B*N_anchors, T_clip) logits, then extracts the last frame per
        anchor → (B, N_anchors) — i.e. for each manifest anchor the model's
        "logit at that frame given the 16-frame context ending there."
        """
        # 1. Strip prefix and pick up task.minilm (single-task fallback).
        raw: dict[str, Tensor] = {}
        for k, v in batch.items():
            if k.startswith(self._PREF_FAIL_PREFIX):
                stripped = k[len(self._PREF_FAIL_PREFIX):]
                # Skip dataset-internal tags like `_is_approaching` so they
                # don't ride into the trunk batch.
                if stripped.startswith("_"):
                    continue
                raw[stripped] = v
        if "task.minilm" not in raw and "task.minilm" in batch:
            raw["task.minilm"] = batch["task.minilm"]

        # 2. Infer (B, N_anchors, T_clip) from one of the stacked tensors.
        if self.config.finetune_encoder:
            cams = list(self.config.encoder_camera_keys)
            ref_key = next((c for c in cams if c in raw), None)
            if ref_key is None:
                raise KeyError(
                    f"Anchor-stack failure batch missing camera tensors. "
                    f"Available: {sorted(raw.keys())}"
                )
            B, N_anchors, T_clip = raw[ref_key].shape[:3]
        else:
            if "img_seq" in raw:
                B, N_anchors, _, T_clip = raw["img_seq"].shape[:4]
            else:
                raise KeyError(
                    f"Anchor-stack failure batch missing img_seq. "
                    f"Available: {sorted(raw.keys())}"
                )

        # 3. Flatten anchor dim into batch dim for tensors with anchor dim;
        #    expand task.minilm (which has no anchor dim) across anchors.
        flat: dict[str, Tensor] = {}
        for k, v in raw.items():
            if not isinstance(v, torch.Tensor):
                flat[k] = v
                continue
            if k == "task.minilm":
                # (B, D) → (B*N_anchors, D)
                flat[k] = v.unsqueeze(1).expand(-1, N_anchors, *v.shape[1:]).reshape(B * N_anchors, *v.shape[1:])
            elif v.dim() >= 3 and v.shape[0] == B and v.shape[1] == N_anchors:
                # (B, N_anchors, ...) → (B*N_anchors, ...)
                flat[k] = v.reshape(B * N_anchors, *v.shape[2:])
            else:
                flat[k] = v

        # 4. Synthesize lengths for the flattened batch (all clips full length).
        device = next(t.device for t in flat.values() if isinstance(t, torch.Tensor))
        flat["lengths"] = torch.full(
            (B * N_anchors,), T_clip, dtype=torch.long, device=device,
        )

        # 5. Forward through trunk + fusion_net.
        logits_flat = self._predict_reward_logits(flat)   # (B*N_anchors, T_clip)
        logits = logits_flat.view(B, N_anchors, T_clip)

        # 6. Last logit per anchor → model's prediction at the anchor frame.
        return logits[:, :, -1]   # (B, N_anchors)

    def _predict_success_anchor_logits(self, batch: dict[str, Tensor]) -> Tensor:
        """Anchor-stack forward for the success clip.

        Mirrors `_predict_failure_anchor_logits` with the success prefix.
        With `success_sampling_mode="hybrid"` the dataset emits one hybrid
        clip per manifest anchor, stacked along a new dim 0:
            observation.images.<cam>: (B, N_anchors, T_clip, C, H, W)
            observation.state:        (B, N_anchors, T_clip, state_dim)
            img_seq:                  (B, N_anchors, N_cams, T_clip, vis_dim)
            task.minilm:              (B, text_dim)  (same across anchors)

        Returns (B, N_anchors) — the last-frame logit per anchor.
        """
        raw: dict[str, Tensor] = {}
        for k, v in batch.items():
            if k.startswith(self._PREF_SUCC_PREFIX):
                stripped = k[len(self._PREF_SUCC_PREFIX):]
                # Skip dataset-internal tags like `_is_approaching` so they
                # don't ride into the trunk batch.
                if stripped.startswith("_"):
                    continue
                raw[stripped] = v
        if "task.minilm" not in raw and "task.minilm" in batch:
            raw["task.minilm"] = batch["task.minilm"]

        if self.config.finetune_encoder:
            cams = list(self.config.encoder_camera_keys)
            ref_key = next((c for c in cams if c in raw), None)
            if ref_key is None:
                raise KeyError(
                    f"Anchor-stack success batch missing camera tensors. "
                    f"Available: {sorted(raw.keys())}"
                )
            B, N_anchors, T_clip = raw[ref_key].shape[:3]
        else:
            if "img_seq" in raw:
                B, N_anchors, _, T_clip = raw["img_seq"].shape[:4]
            else:
                raise KeyError(
                    f"Anchor-stack success batch missing img_seq. "
                    f"Available: {sorted(raw.keys())}"
                )

        flat: dict[str, Tensor] = {}
        for k, v in raw.items():
            if not isinstance(v, torch.Tensor):
                flat[k] = v
                continue
            if k == "task.minilm":
                flat[k] = v.unsqueeze(1).expand(-1, N_anchors, *v.shape[1:]).reshape(B * N_anchors, *v.shape[1:])
            elif v.dim() >= 3 and v.shape[0] == B and v.shape[1] == N_anchors:
                flat[k] = v.reshape(B * N_anchors, *v.shape[2:])
            else:
                flat[k] = v

        device = next(t.device for t in flat.values() if isinstance(t, torch.Tensor))
        flat["lengths"] = torch.full(
            (B * N_anchors,), T_clip, dtype=torch.long, device=device,
        )

        logits_flat = self._predict_reward_logits(flat)   # (B*N_anchors, T_clip)
        logits = logits_flat.view(B, N_anchors, T_clip)
        return logits[:, :, -1]   # (B, N_anchors)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        # 1. Per-frame reward MSE on the rewind sample (unchanged).
        rewards = self._predict_rewards(batch)
        reward_loss = F.mse_loss(rewards, batch["reward"])

        # 2. Preference pair. Success is always a single clip (manifest
        #    indices). Failure depends on failure_sampling_mode:
        #      - "manifest": single clip, 16 in-context logits.
        #      - "hybrid":   anchor-stack of N_anchors hybrid clips. We run
        #                    a batched forward on all anchors and keep one
        #                    last-logit per anchor — gives a "per-frame
        #                    prediction at each manifest anchor" trajectory.
        if self.config.success_sampling_mode == "hybrid":
            # (B, N_anchors) per-anchor "last logit" trajectory — analogous
            # to the failure-hybrid path. Used by the new succ-anchor MSE,
            # and also fed to mono-on-success (now operates per-anchor).
            # stop_grad_on_success only really targets BT (which separately
            # detaches its succ score). Mono and the succ-anchor MSE both
            # consume the raw succ_logits and need a gradient path — so
            # only fall back to no_grad when NO consumer wants grad.
            succ_needs_grad = (
                (not self.config.stop_grad_on_success)
                or self.config.mono_on_success
                or (self.config.succ_anchor_mse_weight > 0)
            )
            if succ_needs_grad:
                succ_logits = self._predict_success_anchor_logits(batch)
            else:
                with torch.no_grad():
                    succ_logits = self._predict_success_anchor_logits(batch)
        else:
            succ_batch = self._extract_pref_subbatch(batch, self._PREF_SUCC_PREFIX)
            succ_needs_grad = (
                (not self.config.stop_grad_on_success) or self.config.mono_on_success
            )
            if succ_needs_grad:
                succ_logits = self._predict_reward_logits(succ_batch)
            else:
                with torch.no_grad():
                    succ_logits = self._predict_reward_logits(succ_batch)

        if self.config.failure_sampling_mode == "hybrid":
            # (B, N_anchors) per-anchor "last logit" trajectory. Mono on this
            # trajectory directly trains the shape we want at eval (model's
            # predicted reward as the context window slides through the
            # failure trajectory).
            fail_anchor_logits = self._predict_failure_anchor_logits(batch)
            # For BT (currently off; weight=0) we use the discounted-mean over
            # ANCHORS — semantically the same idea as discounted-mean over
            # in-clip frames, just at a different temporal resolution.
            fail_for_bt = fail_anchor_logits
            fail_for_mono = fail_anchor_logits
            fail_log_logit_score = fail_anchor_logits.mean(dim=1)   # for telemetry
        else:
            fail_batch = self._extract_pref_subbatch(batch, self._PREF_FAIL_PREFIX)
            fail_logits = self._predict_reward_logits(fail_batch)   # (B, T_clip)
            fail_for_bt = fail_logits
            fail_for_mono = fail_logits
            fail_log_logit_score = self._discounted_clip_score(fail_logits)

        # 3. BT on the γ-discounted weighted-mean of LOGITS (over frames in
        #    manifest mode, or over anchors in hybrid mode). Margin clamped.
        fail_logit_score = self._discounted_clip_score(fail_for_bt)
        succ_logit_score = self._discounted_clip_score(succ_logits)
        if self.config.stop_grad_on_success:
            succ_logit_score = succ_logit_score.detach()
        margin = succ_logit_score - fail_logit_score
        bounded_margin = torch.clamp(margin, max=self.config.bt_target_margin)
        reward_pref_loss = -F.logsigmoid(bounded_margin).mean()

        # 4. Monotonicity hinge in LOGIT space.
        #    eps applies ONLY to failure (where the shape is a known
        #    structural property). Success-side mono uses eps=0 — "just
        #    don't go down."
        #
        #    Failure shape depends on mode:
        #     - "manifest": single 16-frame clip. Default mono-DECREASE
        #       across all 16 (correct for "leaving" direction).
        #     - "hybrid": (B, N_anchors) anchor trajectory. The k-th anchor
        #       is the k-th manifest index — which sweeps from
        #       ~critical_t-5 to ~critical_t+60 across the failure tail.
        #       So we want mono-DECREASE across anchors (later anchor =
        #       later in failure trajectory = lower predicted reward).
        fail_eps_leaving = float(self.config.mono_step_eps)
        fail_eps_approaching = float(self.config.mono_step_eps_approaching)
        succ_eps = 0.0
        mono_terms = []
        if self.config.mono_on_failure and fail_for_mono.size(1) > 1:
            # If the dataset emits a per-batch `_is_approaching` tag (fb
            # manifests), pick eps per element; otherwise use the leaving eps
            # uniformly (blend manifest is leaving-only).
            is_approaching = batch.get(self._PREF_FAIL_PREFIX + "_is_approaching")
            if is_approaching is not None:
                # is_approaching: (B,) long with 0/1 values.
                eps_per_b = torch.where(
                    is_approaching.bool(),
                    torch.tensor(fail_eps_approaching, device=fail_for_mono.device),
                    torch.tensor(fail_eps_leaving, device=fail_for_mono.device),
                )
                mono_terms.append(self._mono_hinge(fail_for_mono, decreasing=True, eps=eps_per_b))
            else:
                mono_terms.append(self._mono_hinge(fail_for_mono, decreasing=True, eps=fail_eps_leaving))
        if self.config.mono_on_success and succ_logits.size(1) > 1:
            mono_terms.append(self._mono_hinge(succ_logits, decreasing=False, eps=succ_eps))
        if mono_terms:
            mono_loss = torch.stack(mono_terms).mean()
        else:
            mono_loss = torch.zeros((), device=reward_loss.device)

        # 5. Optional TV smoothness regularizer on the failure trajectory
        #    (per-anchor in hybrid mode, per-frame in manifest mode) and
        #    optionally the success in-clip trajectory.
        tv_order = int(self.config.tv_order)
        tv_terms = []
        if self.config.tv_on_failure and fail_for_mono.size(1) > 1:
            tv_terms.append(self._tv_loss(fail_for_mono, order=tv_order))
        if self.config.tv_on_success and succ_logits.size(1) > 1:
            tv_terms.append(self._tv_loss(succ_logits, order=tv_order))
        if tv_terms:
            tv_loss = torch.stack(tv_terms).mean()
        else:
            tv_loss = torch.zeros((), device=reward_loss.device)

        step_t = batch.get("training_step")
        step = int(step_t.item()) if torch.is_tensor(step_t) else int(step_t or 0)
        reward_pref_w = self._reward_pref_weight(step)
        mono_w = self._mono_weight(step)
        tv_w = self._tv_weight(step)
        succ_anchor_mse_w = self._succ_anchor_mse_weight(step)

        # Variant 7: MSE on the success anchor logits against per-anchor
        # frame-progress. Only active in success-hybrid mode (the keys
        # `_anchor_frames` / `_episode_length` are only emitted there).
        if self.config.success_sampling_mode == "hybrid":
            anchor_frames = batch[self._PREF_SUCC_PREFIX + "_anchor_frames"]   # (B, N_anchors) long
            ep_len = batch[self._PREF_SUCC_PREFIX + "_episode_length"]          # (B,) long
            denom = (ep_len.float() - 1.0).clamp_min(1.0).unsqueeze(-1)         # (B, 1)
            targets = (anchor_frames.float() / denom).clamp(0.0, 1.0)           # (B, N_anchors)
            succ_anchor_mse = F.mse_loss(torch.sigmoid(succ_logits), targets)
        else:
            succ_anchor_mse = torch.zeros((), device=reward_loss.device)

        loss = (
            reward_loss
            + reward_pref_w * reward_pref_loss
            + mono_w * mono_loss
            + tv_w * tv_loss
            + succ_anchor_mse_w * succ_anchor_mse
        )

        # Interpretable sigmoid-space metrics for monitoring (not used in loss).
        # In hybrid mode `fail_for_mono` is the (B, N_anchors) per-anchor
        # last-logit trajectory; in manifest mode it's the (B, T_clip) in-clip
        # logit sequence. Either way the sigmoid'd discounted-mean is a
        # reasonable summary in [0, 1].
        fail_sigmoid_score = self._discounted_clip_score(torch.sigmoid(fail_for_mono)).detach()
        succ_sigmoid_score = self._discounted_clip_score(torch.sigmoid(succ_logits)).detach()
        return loss, {
            "rewards": rewards.detach(),
            "reward_loss": reward_loss.detach(),
            "reward_pref_loss": reward_pref_loss.detach(),
            "mono_loss": mono_loss.detach(),
            "tv_loss": tv_loss.detach(),
            # Logit-space (what BT sees / what's clamped):
            "reward_pref_margin": margin.mean().detach(),
            "fusion_succ_logit_mean": succ_logit_score.mean().detach(),
            "fusion_fail_logit_mean": fail_logit_score.mean().detach(),
            # Sigmoid-space (interpretable as reward in [0,1]):
            "fusion_succ_score_mean": succ_sigmoid_score.mean(),
            "fusion_fail_score_mean": fail_sigmoid_score.mean(),
            "reward_pref_weight": torch.tensor(reward_pref_w, device=loss.device),
            "mono_weight": torch.tensor(mono_w, device=loss.device),
            "tv_weight": torch.tensor(tv_w, device=loss.device),
            # Variant 7: success-anchor MSE.
            "succ_anchor_mse": succ_anchor_mse.detach(),
            "succ_anchor_mse_weight": torch.tensor(succ_anchor_mse_w, device=loss.device),
        }

    def get_optim_params(self):
        if not self.config.finetune_encoder:
            return self.parameters()
        encoder_params = [p for p in self.vision_encoder.parameters() if p.requires_grad]
        other_params = [p for n, p in self.named_parameters()
                        if not n.startswith("vision_encoder") and p.requires_grad]
        return [
            {"params": other_params},
            {"params": encoder_params, "lr": self.config.encoder_lr},
        ]

    def reset(self):
        pass

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("RewindStateOursVariant7Policy does not predict action chunks")

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("RewindStateOursVariant7Policy does not select actions")
