"""
Rewind reward model with proprioceptive state.

Modality fusion follows the SARM-style per-modality MLP projection
(x_i = MLP(v_i) + MLP(s_i) + MLP(g)), but preserves the (N+1)*T transformer
token structure used in the base RewindPolicy. State is summed additively
into every per-frame token (cameras and language).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from .configuration_rewind_state import RewindStateConfig

# ImageNet normalisation constants for online DINOv2 encoding
_DINO_MEAN = [0.485, 0.456, 0.406]
_DINO_STD = [0.229, 0.224, 0.225]


def _make_mlp(in_dim: int, d_model: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, d_model),
        nn.GELU(),
        nn.Linear(d_model, d_model),
    )


class RewindStatePolicy(PreTrainedPolicy):
    config_class = RewindStateConfig
    name = "rewind_state"

    def __init__(
        self,
        config: RewindStateConfig,
        dataset_stats: dict | None = None,
        dataset_meta: dict | None = None,
    ):
        super().__init__(config)
        self.d_model = config.d_model
        self.num_cameras = config.num_cameras
        self.dataset_stats = dataset_stats
        self.dataset_meta = dataset_meta

        # Optional online DINOv2 encoder (partial fine-tune)
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

        # Per-modality MLP projections (replaces the previous Linear projections)
        self.visual_mlps = nn.ModuleList([
            _make_mlp(config.vis_emb_dim, config.d_model)
            for _ in range(config.num_cameras)
        ])
        self.lang_mlp = _make_mlp(config.text_emb_dim, config.d_model)
        self.state_mlp = _make_mlp(config.state_dim, config.d_model)

        # Transformer encoder (unchanged)
        enc_layer = nn.TransformerEncoderLayer(
            config.d_model,
            config.n_heads,
            4 * config.d_model,
            config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, config.n_layers)

        # Positional bias only for first visual frame
        self.first_pos = nn.Parameter(torch.zeros(1, config.d_model))

        # Fusion MLP (unchanged shape: (N+1)*d_model -> d_model -> 1)
        self.fusion_net = nn.Sequential(
            nn.LayerNorm(config.d_model * (config.num_cameras + 1)),
            nn.Linear(config.d_model * (config.num_cameras + 1), config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
        )

    @torch.no_grad()
    def predict_reward(self, batch: dict[str, Tensor]) -> Tensor:
        return self._predict_rewards(batch)

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        loss, _ = self.forward(batch)
        return loss

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

    def _predict_rewards(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute per-frame rewards from a batch. Returns (B, T) in [0, 1]."""
        if self.config.finetune_encoder:
            img_seq = self._encode_images(batch)
        else:
            img_seq = batch["img_seq"]
        lang_emb = batch["task.minilm"]
        state = batch["observation.state"]                  # (B, T, state_dim)
        lengths = batch.get("lengths")

        B, N, T, _ = img_seq.shape                          # (batch_size, num_cameras, seq_len, vis_emb_dim)
        D = self.d_model
        device = img_seq.device

        # State projection (shared across all per-frame tokens at the same timestep)
        s_proj = self.state_mlp(state)                      # (B, T, D)

        # Per-camera visual MLP + per-frame state addition
        vis_proj = torch.stack(
            [self.visual_mlps[c](img_seq[:, c]) + s_proj for c in range(N)],
            dim=1,
        )                                                    # (B, N, T, D)

        # Language projection broadcast across time + per-frame state
        lang_proj = self.lang_mlp(lang_emb)                 # (B, D)
        lang_proj = lang_proj.unsqueeze(1).expand(B, T, D)  # (B, T, D)
        lang_proj = lang_proj + s_proj                      # (B, T, D)
        lang_proj = lang_proj.unsqueeze(1)                  # (B, 1, T, D)

        # Concatenate modality dim
        x = torch.cat([vis_proj, lang_proj], dim=1)         # (B, N+1, T, D)

        # First-frame positional bias on cameras only
        x[:, :N, 0, :] += self.first_pos                    # (B, N+1, T, D)

        # Reshape for transformer: (B, (N+1)*T, D)
        x_tokens = x.view(B, (N + 1) * T, D)
        L = x_tokens.size(1)

        # Build transformer mask
        base_mask = torch.arange(T, device=device).expand(B, T) >= lengths.unsqueeze(1)  # (B, T)
        mask = base_mask.unsqueeze(1).expand(B, N + 1, T).reshape(B, (N + 1) * T)         # (B, (N+1)*T)
        causal_mask = torch.triu(
            torch.ones(L, L, device=device, dtype=torch.bool),
            diagonal=1,
        )

        h = self.transformer(
            x_tokens,
            mask=causal_mask,
            src_key_padding_mask=mask,
            is_causal=True,
        )                                                    # (B, (N+1)*T, D)

        flattened_h = h.view(B, T, -1)                       # (B, T, (N+1)*D)
        r = self.fusion_net(flattened_h).squeeze(-1)         # (B, T)
        return torch.sigmoid(r)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[Tensor, dict]:
        """Training forward: returns (loss, output_dict).

        DDP only synchronizes gradients computed inside .forward(), so the loss
        must be produced here for correct multi-GPU behavior. For inference,
        use predict_reward(batch) instead.
        """
        rewards = self._predict_rewards(batch)
        loss = F.mse_loss(rewards, batch["reward"])
        return loss, {"rewards": rewards}

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
        """Required by PreTrainedPolicy but not used for reward models."""
        pass

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("RewindStatePolicy does not predict action chunks")

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("RewindStatePolicy does not select actions")
