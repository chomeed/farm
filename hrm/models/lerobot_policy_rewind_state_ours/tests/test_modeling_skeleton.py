from __future__ import annotations

import torch

from lerobot_policy_rewind_state_ours.configuration_rewind_state_ours import (
    RewindStateOursConfig,
)
from lerobot_policy_rewind_state_ours.modeling_rewind_state_ours import (
    RewindStateOursPolicy,
)


def _tiny_cfg() -> RewindStateOursConfig:
    cfg = RewindStateOursConfig()
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


def test_policy_instantiates_and_has_pref_net():
    cfg = _tiny_cfg()
    p = RewindStateOursPolicy(cfg)
    assert hasattr(p, "pref_net")
    assert hasattr(p, "fusion_net")
    assert hasattr(p, "_trunk")
    assert hasattr(p, "_predict_rewards")
    assert hasattr(p, "_predict_preference_score")


def test_predict_rewards_shape():
    cfg = _tiny_cfg()
    p = RewindStateOursPolicy(cfg).eval()
    batch = _tiny_batch(B=2, T=4)
    with torch.no_grad():
        r = p._predict_rewards(batch)
        # Recompute the pre-sigmoid logit by hand and confirm `r == sigmoid(logit)`.
        flattened_h, _ = p._trunk(batch)
        logit = p.fusion_net(flattened_h).squeeze(-1)
    assert r.shape == (2, 4)
    assert torch.allclose(r, torch.sigmoid(logit), atol=1e-6)


def test_predict_preference_score_shape():
    cfg = _tiny_cfg()
    p = RewindStateOursPolicy(cfg).eval()
    batch = _tiny_batch(B=3, T=4)
    with torch.no_grad():
        s = p._predict_preference_score(batch)
    assert s.shape == (3,)
    # Pre-sigmoid score; no bound check.
    assert torch.isfinite(s).all()


def test_preference_weight_warmup():
    cfg = _tiny_cfg()
    cfg.preference_loss_weight = 2.0
    cfg.preference_warmup_steps = 100
    p = RewindStateOursPolicy(cfg)
    assert p._preference_weight(0) == 0.0
    assert p._preference_weight(50) == 1.0
    assert p._preference_weight(100) == 2.0
    assert p._preference_weight(1000) == 2.0


def test_preference_weight_no_warmup():
    cfg = _tiny_cfg()
    cfg.preference_loss_weight = 0.5
    cfg.preference_warmup_steps = 0
    p = RewindStateOursPolicy(cfg)
    assert p._preference_weight(0) == 0.5
    assert p._preference_weight(123) == 0.5
