from __future__ import annotations

from lerobot.configs.policies import PreTrainedConfig

from lerobot_policy_rewind_state_ours.configuration_rewind_state_ours import (
    RewindStateOursConfig,
)


def test_registered_as_pretrained_subclass():
    cfg = PreTrainedConfig.get_choice_class("rewind_state_ours")
    assert cfg is RewindStateOursConfig


def test_default_preference_fields():
    cfg = RewindStateOursConfig()
    assert cfg.preference_pairs_path == ""
    assert cfg.preference_loss_weight == 1.0
    assert cfg.preference_warmup_steps == 500
    assert cfg.preference_clip_len == 16
    assert cfg.preference_frame_stride == 1
    assert cfg.preference_topk == 3
    assert cfg.preference_seed == 0


def test_inherits_base_rewind_state_field_defaults():
    """Same numeric defaults as RewindStateConfig, so checkpoint shapes match."""
    cfg = RewindStateOursConfig()
    assert cfg.d_model == 512
    assert cfg.vis_emb_dim == 768
    assert cfg.text_emb_dim == 384
    assert cfg.n_layers == 4
    assert cfg.n_heads == 8
    assert cfg.dropout == 0.1
    assert cfg.num_cameras == 3


def test_optimizer_and_scheduler_presets_present():
    cfg = RewindStateOursConfig()
    opt = cfg.get_optimizer_preset()
    sched = cfg.get_scheduler_preset()
    assert opt.lr == cfg.optimizer_lr
    assert sched.peak_lr == cfg.scheduler_peak_lr
