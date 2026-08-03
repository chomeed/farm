from __future__ import annotations

import torch

from lerobot_policy_rewind_state_ours_variant8.configuration_rewind_state_ours_variant8 import (
    RewindStateOursVariant8Config,
)
from lerobot_policy_rewind_state_ours_variant8.modeling_rewind_state_ours_variant8 import (
    RewindStateOursVariant8Policy,
)


def _tiny_cfg() -> RewindStateOursVariant8Config:
    cfg = RewindStateOursVariant8Config()
    cfg.d_model = 32
    cfg.n_layers = 1
    cfg.n_heads = 2
    cfg.num_cameras = 2
    cfg.state_dim = 9
    cfg.text_emb_dim = 384
    cfg.vis_emb_dim = 768
    cfg.finetune_encoder = False
    cfg.reward_pref_warmup_steps = 0
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
    p = RewindStateOursVariant8Policy(_tiny_cfg()).train()
    batch = _tiny_combined_batch()
    batch["training_step"] = torch.tensor(0)
    loss, info = p(batch)
    assert torch.isfinite(loss)
    assert torch.isfinite(info["reward_loss"])
    assert torch.isfinite(info["reward_pref_loss"])
    assert torch.isfinite(info["reward_pref_margin"])
    assert torch.isfinite(info["reward_pref_weight"])


def test_warmup_zeroes_pref_weight_at_step_zero():
    cfg = _tiny_cfg()
    cfg.reward_pref_warmup_steps = 100
    p = RewindStateOursVariant8Policy(cfg).train()
    batch = _tiny_combined_batch()
    batch["training_step"] = torch.tensor(0)
    loss, info = p(batch)
    # Total loss should equal reward_loss when the pref weight is 0.
    assert info["reward_pref_weight"].item() == 0.0
    assert torch.allclose(loss, info["reward_loss"])


def test_warmup_full_pref_weight_after_warmup():
    cfg = _tiny_cfg()
    cfg.reward_pref_warmup_steps = 100
    cfg.reward_pref_loss_weight = 1.0
    p = RewindStateOursVariant8Policy(cfg).train()
    batch = _tiny_combined_batch()
    batch["training_step"] = torch.tensor(200)
    loss, info = p(batch)
    assert info["reward_pref_weight"].item() == 1.0
    expected = info["reward_loss"] + info["reward_pref_loss"]
    assert torch.allclose(loss, expected)


def test_backprop_writes_gradients_into_fusion_net():
    cfg = _tiny_cfg()
    cfg.reward_pref_loss_weight = 1.0
    p = RewindStateOursVariant8Policy(cfg).train()
    batch = _tiny_combined_batch()
    batch["training_step"] = torch.tensor(0)
    loss, _ = p(batch)
    loss.backward()
    grads = [param.grad for param in p.fusion_net.parameters() if param.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_stop_grad_on_success_blocks_success_gradient(monkeypatch):
    """With stop_grad_on_success=True, succ_score must not carry gradient;
    with False, it must. fail_score always carries gradient."""
    succ_leaf = torch.tensor([0.6, 0.6], requires_grad=True)
    fail_leaf = torch.tensor([0.4, 0.4], requires_grad=True)

    def run(stop_grad: bool):
        succ_leaf.grad = None
        fail_leaf.grad = None
        calls = {"n": 0}

        def fake_score(self, r_seq):
            # forward() calls fail first, then succ.
            leaf = fail_leaf if calls["n"] == 0 else succ_leaf
            calls["n"] += 1
            return leaf * 1.0  # fresh op so torch.no_grad() can take effect

        monkeypatch.setattr(
            RewindStateOursVariant8Policy, "_discounted_clip_score", fake_score
        )
        cfg = _tiny_cfg()
        cfg.reward_pref_warmup_steps = 0
        cfg.reward_pref_loss_weight = 1.0
        cfg.stop_grad_on_success = stop_grad
        p = RewindStateOursVariant8Policy(cfg).train()
        batch = _tiny_combined_batch()
        batch["training_step"] = torch.tensor(0)
        loss, _ = p(batch)
        loss.backward()
        return succ_leaf.grad, fail_leaf.grad

    # Only _discounted_clip_score is patched; reward_loss calls _predict_rewards,
    # so fail_leaf's grad comes exclusively from the BT loss path.
    succ_grad, fail_grad = run(stop_grad=True)
    assert succ_grad is None
    assert fail_grad is not None and fail_grad.abs().sum() > 0

    succ_grad, fail_grad = run(stop_grad=False)
    assert succ_grad is not None and succ_grad.abs().sum() > 0
    assert fail_grad is not None and fail_grad.abs().sum() > 0


def test_missing_pref_keys_raises():
    p = RewindStateOursVariant8Policy(_tiny_cfg()).train()
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
    cfg.reward_pref_warmup_steps = 0

    p = RewindStateOursVariant8Policy(cfg).train()
    # Monkey-patch _encode_images to skip the real DINOv2 call.
    def fake_encode(self, batch):
        # Return (B, N=2, T, vis_emb_dim).
        cam0 = batch[cfg.encoder_camera_keys[0]]
        B, T = cam0.shape[:2]
        return torch.zeros(B, 2, T, cfg.vis_emb_dim)
    monkeypatch.setattr(RewindStateOursVariant8Policy, "_encode_images", fake_encode)

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
    assert torch.isfinite(info["reward_pref_margin"])


def test_pref_loss_direction(monkeypatch):
    """succ_score > fail_score should give a smaller BT loss than the reverse.

    Locks the sign of the Bradley-Terry gradient on fusion_net's clip score.
    Scores are sigmoid'd weighted means, so values live in [0, 1].
    """
    batch = _tiny_combined_batch()
    batch["training_step"] = torch.tensor(0)
    high = torch.tensor([0.99, 0.99])
    low = torch.tensor([0.01, 0.01])

    def make_patch(seq):
        state = {"i": 0}

        def fake_score(self, r_seq):
            v = seq[state["i"] % len(seq)]
            state["i"] += 1
            return v.clone()

        return fake_score

    cfg = _tiny_cfg()
    cfg.reward_pref_warmup_steps = 0
    cfg.reward_pref_loss_weight = 1.0

    # forward() calls fail first, then succ.
    monkeypatch.setattr(
        RewindStateOursVariant8Policy, "_discounted_clip_score",
        make_patch([low, high]),  # fail=low, succ=high  -> correct ranking
    )
    _, info_correct = RewindStateOursVariant8Policy(cfg).train()(batch)

    monkeypatch.setattr(
        RewindStateOursVariant8Policy, "_discounted_clip_score",
        make_patch([high, low]),  # fail=high, succ=low  -> wrong ranking
    )
    _, info_wrong = RewindStateOursVariant8Policy(cfg).train()(batch)

    assert info_correct["reward_pref_loss"].item() < info_wrong["reward_pref_loss"].item()
    assert info_correct["reward_pref_loss"].item() < 0.5
    assert info_wrong["reward_pref_loss"].item() > 1.0
