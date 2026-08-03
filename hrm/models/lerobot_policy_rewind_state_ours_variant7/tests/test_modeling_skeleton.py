from __future__ import annotations

import torch

from lerobot_policy_rewind_state_ours_variant7.configuration_rewind_state_ours_variant7 import (
    RewindStateOursVariant7Config,
)
from lerobot_policy_rewind_state_ours_variant7.modeling_rewind_state_ours_variant7 import (
    RewindStateOursVariant7Policy,
)


def _tiny_cfg() -> RewindStateOursVariant7Config:
    cfg = RewindStateOursVariant7Config()
    cfg.d_model = 32
    cfg.n_layers = 1
    cfg.n_heads = 2
    cfg.num_cameras = 2
    cfg.state_dim = 9
    cfg.text_emb_dim = 384
    cfg.vis_emb_dim = 768
    cfg.finetune_encoder = False
    # The existing _tiny_combined_batch emits non-hybrid success keys; hybrid
    # mode is exercised by _tiny_combined_batch_succ_hybrid in dedicated tests.
    cfg.success_sampling_mode = "manifest"
    cfg.succ_anchor_mse_warmup_steps = 0
    return cfg


def _tiny_batch(B: int = 2, T: int = 4) -> dict[str, torch.Tensor]:
    return {
        "img_seq": torch.randn(B, 2, T, 768),
        "task.minilm": torch.randn(B, 384),
        "observation.state": torch.randn(B, T, 9),
        "lengths": torch.tensor([T] * B, dtype=torch.long),
        "reward": torch.zeros(B, T),
    }


def test_policy_instantiates_without_pref_net():
    cfg = _tiny_cfg()
    p = RewindStateOursVariant7Policy(cfg)
    assert not hasattr(p, "pref_net")
    assert not hasattr(p, "predict_preference_score")
    assert hasattr(p, "fusion_net")
    assert hasattr(p, "_trunk")
    assert hasattr(p, "_predict_rewards")
    assert hasattr(p, "_discounted_clip_score")
    assert hasattr(p, "_predict_failure_anchor_logits")
    assert hasattr(p, "_predict_success_anchor_logits")


def test_predict_rewards_shape():
    cfg = _tiny_cfg()
    p = RewindStateOursVariant7Policy(cfg).eval()
    batch = _tiny_batch(B=2, T=4)
    with torch.no_grad():
        r = p._predict_rewards(batch)
        # Recompute the pre-sigmoid logit by hand and confirm `r == sigmoid(logit)`.
        flattened_h, _ = p._trunk(batch)
        logit = p.fusion_net(flattened_h).squeeze(-1)
    assert r.shape == (2, 4)
    assert torch.allclose(r, torch.sigmoid(logit), atol=1e-6)


def test_reward_pref_weight_warmup():
    cfg = _tiny_cfg()
    cfg.reward_pref_loss_weight = 2.0
    cfg.reward_pref_warmup_steps = 100
    p = RewindStateOursVariant7Policy(cfg)
    assert p._reward_pref_weight(0) == 0.0
    assert p._reward_pref_weight(50) == 1.0
    assert p._reward_pref_weight(100) == 2.0
    assert p._reward_pref_weight(1000) == 2.0


def test_reward_pref_weight_no_warmup():
    cfg = _tiny_cfg()
    cfg.reward_pref_loss_weight = 0.5
    cfg.reward_pref_warmup_steps = 0
    p = RewindStateOursVariant7Policy(cfg)
    assert p._reward_pref_weight(0) == 0.5
    assert p._reward_pref_weight(123) == 0.5
