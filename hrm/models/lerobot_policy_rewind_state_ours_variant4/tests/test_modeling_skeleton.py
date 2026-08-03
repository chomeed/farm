from __future__ import annotations

import torch

from lerobot_policy_rewind_state_ours_variant4.configuration_rewind_state_ours_variant4 import (
    RewindStateOursVariant4Config,
)
from lerobot_policy_rewind_state_ours_variant4.modeling_rewind_state_ours_variant4 import (
    RewindStateOursVariant4Policy,
)


def _tiny_cfg() -> RewindStateOursVariant4Config:
    cfg = RewindStateOursVariant4Config()
    cfg.d_model = 32
    cfg.n_layers = 1
    cfg.n_heads = 2
    cfg.num_cameras = 2
    cfg.state_dim = 9
    cfg.text_emb_dim = 384
    cfg.vis_emb_dim = 768
    cfg.finetune_encoder = False
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
    p = RewindStateOursVariant4Policy(cfg)
    assert not hasattr(p, "pref_net")
    assert not hasattr(p, "predict_preference_score")
    assert hasattr(p, "fusion_net")
    assert hasattr(p, "_trunk")
    assert hasattr(p, "_predict_rewards")
    assert hasattr(p, "_predict_clip_potential")


def test_predict_rewards_shape():
    cfg = _tiny_cfg()
    p = RewindStateOursVariant4Policy(cfg).eval()
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
    p = RewindStateOursVariant4Policy(cfg)
    assert p._reward_pref_weight(0) == 0.0
    assert p._reward_pref_weight(50) == 1.0
    assert p._reward_pref_weight(100) == 2.0
    assert p._reward_pref_weight(1000) == 2.0


def test_reward_pref_weight_no_warmup():
    cfg = _tiny_cfg()
    cfg.reward_pref_loss_weight = 0.5
    cfg.reward_pref_warmup_steps = 0
    p = RewindStateOursVariant4Policy(cfg)
    assert p._reward_pref_weight(0) == 0.5
    assert p._reward_pref_weight(123) == 0.5


def test_potential_delta_clamps_to_clip_length():
    cfg = _tiny_cfg()
    cfg.reward_pref_potential_delta = 8
    p = RewindStateOursVariant4Policy(cfg)
    assert p._potential_delta(20) == 8   # within range -> unchanged
    assert p._potential_delta(4) == 3    # clamped to T-1
    assert p._potential_delta(2) == 1    # clamped to T-1


def test_sample_potential_anchor_in_valid_range():
    cfg = _tiny_cfg()
    cfg.reward_pref_potential_delta = 8
    p = RewindStateOursVariant4Policy(cfg)
    # T=20, delta=8 -> hi = T-1-delta = 11; anchors must stay in [0, 11].
    anchors = {p._sample_potential_anchor(20) for _ in range(200)}
    assert min(anchors) >= 0
    assert max(anchors) <= 11
    assert len(anchors) > 1, "anchor never varied — random branch not exercised"
    # Short clip (hi <= 0): only t=0 is valid.
    assert p._sample_potential_anchor(4) == 0
