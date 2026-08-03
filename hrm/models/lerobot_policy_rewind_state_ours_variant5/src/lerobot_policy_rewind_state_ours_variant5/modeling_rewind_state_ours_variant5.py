"""
Rewind reward + Variant 5 preference learning (`rewind_state_ours_variant5`).

Variant 5 = Variant 3 + Total-Variation regularizer on per-frame rewards
within preference clips. The BT objective is unchanged from Variant 3.
The TV penalty order is configurable (L1 or L2 / H1) — they target
different failure modes of the BT signal.

Heads:
  * `fusion_net` - per-frame reward head. Trained with:
                     (i) MSE on per-frame reward labels (rewind sample),
                     (ii) BT loss on mean sigmoid over the last K frames of
                          success vs failure clip pairs (with optional
                          stop-grad on success — same as Variant 3), and
                     (iii) TV penalty `mean_t |r(s_t) - r(s_{t-1})|^p`
                          on per-frame rewards within pref clips. Order p
                          is selected via `tv_order`; optional on success
                          and failure sides via config flags.

Why TV (and which order):
  * p = 1 (L1): jump-permissive. Penalizes oscillations / chatter but lets
        sharp monotone drops through unchanged. Use when failure clips show
        mid-trajectory wobble (e.g. the blend-15k chatter phase).
  * p = 2 (L2 / H1): quadratic — strongly penalizes large frame-to-frame
        jumps. Incentivizes spreading the BT-induced cliff into a multi-
        frame ramp, which gives PBRS a sustained negative-shaping signal
        over the bad-action window instead of one spike on a single
        transition.

Setting `tv_loss_weight = 0` recovers Variant 3 regardless of order.

Inference unchanged: `predict_reward(batch)` returns the sigmoid'd [0, 1]
progress signal from `fusion_net`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_rewind_state_ours_variant5 import RewindStateOursVariant5Config

_DINO_MEAN = [0.485, 0.456, 0.406]
_DINO_STD = [0.229, 0.224, 0.225]


def _make_mlp(in_dim: int, d_model: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, d_model),
        nn.GELU(),
        nn.Linear(d_model, d_model),
    )


class RewindStateOursVariant5Policy(PreTrainedPolicy):
    config_class = RewindStateOursVariant5Config
    name = "rewind_state_ours_variant5"

    def __init__(
        self,
        config: RewindStateOursVariant5Config,
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

    def _predict_rewards(self, batch: dict[str, Tensor]) -> Tensor:
        """Per-frame reward in [0, 1], shape (B, T). Same as base RewindStatePolicy."""
        flattened_h, _ = self._trunk(batch)
        r = self.fusion_net(flattened_h).squeeze(-1)
        return torch.sigmoid(r)

    def _reward_pref_weight(self, step: int) -> float:
        """Variant 5: warmup schedule for the direct-to-fusion_net BT loss."""
        w_max = self.config.reward_pref_loss_weight
        warm = self.config.reward_pref_warmup_steps
        if warm <= 0:
            return w_max
        return w_max * min(1.0, step / warm)

    def _tv_weight(self, step: int) -> float:
        """Variant 5: warmup schedule for the L1 TV regularizer on pref clips."""
        w_max = self.config.tv_loss_weight
        warm = self.config.tv_warmup_steps
        if warm <= 0:
            return w_max
        return w_max * min(1.0, step / warm)

    def _predict_reward_tail_mean(self, batch: dict[str, Tensor]) -> Tensor:
        """Variant 5: sigmoid(fusion_net) mean over the trailing K frames.

        Returns (B,) — the clip-level "how close to goal at the end" score
        used as the discriminand of the direct BT loss on fusion_net.

        We pull the trunk + fusion_net WITH gradients (in contrast to
        `predict_reward` which is @torch.no_grad).
        """
        flattened_h, _ = self._trunk(batch)
        r = torch.sigmoid(self.fusion_net(flattened_h).squeeze(-1))   # (B, T)
        K = max(1, int(self.config.reward_pref_critical_window))
        K = min(K, r.size(1))
        return r[:, -K:].mean(dim=1)   # (B,)

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

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        # 1. Per-frame reward MSE on the rewind sample (unchanged).
        rewards = self._predict_rewards(batch)
        reward_loss = F.mse_loss(rewards, batch["reward"])

        # 2. Preference pair (already preprocessed by the data pipeline).
        fail_batch = self._extract_pref_subbatch(batch, self._PREF_FAIL_PREFIX)
        succ_batch = self._extract_pref_subbatch(batch, self._PREF_SUCC_PREFIX)

        # 3. Per-frame rewards on each clip — used for BT (tail mean) AND TV.
        #    Failure side always with grad. Success side gets grad only when
        #    needed: BT uses succ_tail.detach() under stop_grad_on_success, and
        #    TV uses succ_r_seq with grad only when tv_on_success.
        fail_r_seq = self._predict_rewards(fail_batch)          # (B, T) grad
        success_needs_grad = (
            (not self.config.stop_grad_on_success) or self.config.tv_on_success
        )
        if success_needs_grad:
            succ_r_seq = self._predict_rewards(succ_batch)
        else:
            with torch.no_grad():
                succ_r_seq = self._predict_rewards(succ_batch)

        # 4. BT on the last-K-frames sigmoid mean (Variant 3 behavior).
        K = max(1, int(self.config.reward_pref_critical_window))
        K = min(K, fail_r_seq.size(1), succ_r_seq.size(1))
        fail_tail = fail_r_seq[:, -K:].mean(dim=1)
        succ_tail = succ_r_seq[:, -K:].mean(dim=1)
        if self.config.stop_grad_on_success:
            succ_tail = succ_tail.detach()
        reward_pref_loss = -F.logsigmoid(succ_tail - fail_tail).mean()

        # 5. TV penalty: mean_t |r(s_t) - r(s_{t-1})|^p within each pref clip.
        #    p selected by `tv_order` (1 = L1 / chatter, 2 = L2 / cliff).
        order = int(self.config.tv_order)
        if order not in (1, 2):
            raise ValueError(
                f"tv_order must be 1 or 2, got {order!r}"
            )

        def _tv(seq: Tensor) -> Tensor:
            diff = seq[:, 1:] - seq[:, :-1]
            return diff.abs().mean() if order == 1 else diff.square().mean()

        tv_terms = []
        if self.config.tv_on_failure and fail_r_seq.size(1) > 1:
            tv_terms.append(_tv(fail_r_seq))
        if self.config.tv_on_success and succ_r_seq.size(1) > 1:
            tv_terms.append(_tv(succ_r_seq))
        if tv_terms:
            tv_loss = torch.stack(tv_terms).mean()
        else:
            tv_loss = torch.zeros((), device=reward_loss.device)

        step_t = batch.get("training_step")
        step = int(step_t.item()) if torch.is_tensor(step_t) else int(step_t or 0)
        reward_pref_w = self._reward_pref_weight(step)
        tv_w = self._tv_weight(step)

        loss = reward_loss + reward_pref_w * reward_pref_loss + tv_w * tv_loss
        return loss, {
            "rewards": rewards.detach(),
            "reward_loss": reward_loss.detach(),
            "reward_pref_loss": reward_pref_loss.detach(),
            "reward_pref_margin": (succ_tail - fail_tail).mean().detach(),
            "reward_pref_weight": torch.tensor(reward_pref_w, device=loss.device),
            "fusion_succ_tail_mean": succ_tail.mean().detach(),
            "fusion_fail_tail_mean": fail_tail.mean().detach(),
            "tv_loss": tv_loss.detach(),
            "tv_weight": torch.tensor(tv_w, device=loss.device),
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
        raise NotImplementedError("RewindStateOursVariant5Policy does not predict action chunks")

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("RewindStateOursVariant5Policy does not select actions")
