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
    cfg.preference_warmup_steps = 0
    return cfg


def _tiny_combined_batch(B: int = 2, T: int = 4):
    """Rewind sample + namespaced pref pair, mirroring the wrapper's output.

    The wrapper emits raw observation.images.*, observation.state, task
    string under pref.failure.* / pref.success.*. It does NOT emit img_seq
    or lengths — those are synthesized by the model.

    For the rewind side we use the frozen-encoder shortcut (`img_seq`,
    `task.minilm`, `lengths`) because the test config sets
    `finetune_encoder=False`. The pref sub-batches use the same shortcut
    so the model's frozen-encoder code path is exercised.
    """
    rng = torch.Generator().manual_seed(0)
    base = {
        "img_seq": torch.randn(B, 2, T, 768, generator=rng),
        "task.minilm": torch.randn(B, 384, generator=rng),
        "observation.state": torch.randn(B, T, 9, generator=rng),
        "lengths": torch.tensor([T] * B, dtype=torch.long),
        "reward": torch.linspace(0.0, 1.0, T).unsqueeze(0).repeat(B, 1),
    }
    pref_failure = {
        "pref.failure.img_seq": torch.randn(B, 2, T, 768, generator=rng),
        "pref.failure.observation.state": torch.randn(B, T, 9, generator=rng),
        # No pref.failure.task.minilm — model falls back to base["task.minilm"].
        # No pref.failure.lengths — model synthesizes from img_seq's T dim.
    }
    pref_success = {
        "pref.success.img_seq": torch.randn(B, 2, T, 768, generator=rng),
        "pref.success.observation.state": torch.randn(B, T, 9, generator=rng),
    }
    return {**base, **pref_failure, **pref_success}


def test_forward_returns_finite_loss_and_info():
    p = RewindStateOursPolicy(_tiny_cfg()).train()
    batch = _tiny_combined_batch()
    batch["training_step"] = torch.tensor(0)
    loss, info = p(batch)
    assert torch.isfinite(loss)
    assert torch.isfinite(info["reward_loss"])
    assert torch.isfinite(info["preference_loss"])
    assert torch.isfinite(info["preference_margin"])
    assert torch.isfinite(info["preference_weight"])


def test_warmup_zeroes_pref_weight_at_step_zero():
    cfg = _tiny_cfg()
    cfg.preference_warmup_steps = 100
    p = RewindStateOursPolicy(cfg).train()
    batch = _tiny_combined_batch()
    batch["training_step"] = torch.tensor(0)
    loss, info = p(batch)
    # Total loss should equal reward_loss when pref_weight is 0.
    assert info["preference_weight"].item() == 0.0
    assert torch.allclose(loss, info["reward_loss"])


def test_warmup_full_pref_weight_after_warmup():
    cfg = _tiny_cfg()
    cfg.preference_warmup_steps = 100
    cfg.preference_loss_weight = 1.0
    p = RewindStateOursPolicy(cfg).train()
    batch = _tiny_combined_batch()
    batch["training_step"] = torch.tensor(200)
    loss, info = p(batch)
    assert info["preference_weight"].item() == 1.0
    expected = info["reward_loss"] + info["preference_loss"]
    assert torch.allclose(loss, expected)


def test_backprop_writes_gradients_into_pref_net():
    p = RewindStateOursPolicy(_tiny_cfg()).train()
    batch = _tiny_combined_batch()
    batch["training_step"] = torch.tensor(0)
    loss, _ = p(batch)
    loss.backward()
    pref_grads = [param.grad for param in p.pref_net.parameters() if param.requires_grad]
    assert all(g is not None for g in pref_grads)
    assert any(g.abs().sum() > 0 for g in pref_grads)


def test_missing_pref_keys_raises():
    p = RewindStateOursPolicy(_tiny_cfg()).train()
    rewind_only = {
        "img_seq": torch.randn(1, 2, 4, 768),
        "task.minilm": torch.randn(1, 384),
        "observation.state": torch.randn(1, 4, 9),
        "lengths": torch.tensor([4], dtype=torch.long),
        "reward": torch.zeros(1, 4),
    }
    try:
        _ = p(rewind_only)
    except KeyError:
        return
    raise AssertionError("Expected KeyError for missing pref.* keys")


def test_finetune_encoder_pref_keys_match_wrapper_output(monkeypatch):
    """The model accepts what PreferenceTaggedRewindDataset actually emits.

    Wrapper output for finetune_encoder=True:
      pref.failure.observation.images.<cam1>, <cam2>
      pref.failure.observation.state
      pref.failure.task  (string)
    No pref.failure.img_seq, no pref.failure.lengths.
    """
    cfg = _tiny_cfg()
    cfg.finetune_encoder = True
    cfg.encoder_camera_keys = [
        "observation.images.agentview",
        "observation.images.eye_in_hand",
    ]
    cfg.preference_warmup_steps = 0

    p = RewindStateOursPolicy(cfg).train()
    # Monkey-patch _encode_images to skip the real DINOv2 call.
    def fake_encode(self, batch):
        # Return (B, N=2, T, vis_emb_dim).
        cam0 = batch[cfg.encoder_camera_keys[0]]
        B, T = cam0.shape[:2]
        return torch.zeros(B, 2, T, cfg.vis_emb_dim)
    monkeypatch.setattr(RewindStateOursPolicy, "_encode_images", fake_encode)

    B, T = 2, 4
    rng = torch.Generator().manual_seed(0)
    batch = {
        # Rewind side (raw images for finetune_encoder=True).
        "observation.images.agentview": torch.randn(B, T, 3, 84, 84, generator=rng),
        "observation.images.eye_in_hand": torch.randn(B, T, 3, 84, 84, generator=rng),
        "task.minilm": torch.randn(B, 384, generator=rng),
        "observation.state": torch.randn(B, T, 9, generator=rng),
        "lengths": torch.tensor([T] * B, dtype=torch.long),
        "reward": torch.linspace(0.0, 1.0, T).unsqueeze(0).repeat(B, 1),
        # Pref pair (raw images, no img_seq, no lengths, no task.minilm).
        "pref.failure.observation.images.agentview": torch.randn(B, T, 3, 84, 84, generator=rng),
        "pref.failure.observation.images.eye_in_hand": torch.randn(B, T, 3, 84, 84, generator=rng),
        "pref.failure.observation.state": torch.randn(B, T, 9, generator=rng),
        "pref.success.observation.images.agentview": torch.randn(B, T, 3, 84, 84, generator=rng),
        "pref.success.observation.images.eye_in_hand": torch.randn(B, T, 3, 84, 84, generator=rng),
        "pref.success.observation.state": torch.randn(B, T, 9, generator=rng),
        "training_step": torch.tensor(0),
    }
    loss, info = p(batch)
    assert torch.isfinite(loss)
    assert torch.isfinite(info["preference_margin"])


def test_pref_loss_direction(monkeypatch):
    """succ_score >> fail_score should give a smaller BT loss than the reverse.

    This locks the sign of the Bradley-Terry gradient: the model is rewarded
    for ranking success above failure, not the other way around.
    """
    p = RewindStateOursPolicy(_tiny_cfg()).train()
    batch = _tiny_combined_batch()
    batch["training_step"] = torch.tensor(0)

    # Force preference scores via monkey-patch so we test forward()'s direction
    # independently of pref_net's random initialization.
    call_state = {"i": 0}
    high = torch.tensor([10.0, 10.0])
    low = torch.tensor([-10.0, -10.0])

    # First two calls: succ=high, fail=low  -> pref_loss should be small.
    # Next two calls: succ=low,  fail=high -> pref_loss should be large.
    sequence_correct = [low, high]   # forward calls fail first, then succ
    sequence_wrong = [high, low]

    def make_patch(seq):
        def fake_score(self, sub_batch):
            v = seq[call_state["i"] % len(seq)]
            call_state["i"] += 1
            return v.clone()
        return fake_score

    # Run the "correct" direction.
    monkeypatch.setattr(
        RewindStateOursPolicy,
        "_predict_preference_score",
        make_patch(sequence_correct),
    )
    call_state["i"] = 0
    cfg2 = _tiny_cfg()
    cfg2.preference_warmup_steps = 0
    cfg2.preference_loss_weight = 1.0
    p2 = RewindStateOursPolicy(cfg2).train()
    _, info_correct = p2(batch)

    # Run the "wrong" direction.
    monkeypatch.setattr(
        RewindStateOursPolicy,
        "_predict_preference_score",
        make_patch(sequence_wrong),
    )
    call_state["i"] = 0
    p3 = RewindStateOursPolicy(cfg2).train()
    _, info_wrong = p3(batch)

    assert info_correct["preference_loss"].item() < info_wrong["preference_loss"].item()
    # And specifically: when succ >> fail, BT loss should be very small.
    assert info_correct["preference_loss"].item() < 1e-3
    # When succ << fail, BT loss should be roughly |succ - fail| = 20.
    assert info_wrong["preference_loss"].item() > 15.0
