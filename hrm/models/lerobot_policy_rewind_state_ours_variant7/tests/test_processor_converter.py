from __future__ import annotations

import torch

from lerobot.processor.converters import transition_to_batch
from lerobot.processor.core import TransitionKey

from lerobot_policy_rewind_state_ours_variant7.processor_rewind_state_ours_variant7 import (
    _rso_batch_to_transition,
)


def _make_batch():
    return {
        "observation.state": torch.zeros(2, 4, 9),
        "task": ["pick", "pick"],
        "reward": torch.zeros(2, 4),
        # Namespaced pref keys — should pass through to COMPLEMENTARY_DATA.
        "pref.failure.observation.images.agentview": torch.randn(2, 16, 3, 84, 84),
        "pref.failure.observation.state": torch.randn(2, 16, 9),
        "pref.success.observation.images.agentview": torch.randn(2, 16, 3, 84, 84),
        "pref.success.observation.state": torch.randn(2, 16, 9),
    }


def test_pref_keys_round_trip_through_converter():
    batch = _make_batch()
    transition = _rso_batch_to_transition(batch)
    comp = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
    for k in batch:
        if k.startswith("pref."):
            assert k in comp, f"Missing {k} in COMPLEMENTARY_DATA"
            assert torch.equal(comp[k], batch[k])

    out = transition_to_batch(transition)
    for k in batch:
        if k.startswith("pref."):
            assert k in out, f"Missing {k} after round trip"
            assert torch.equal(out[k], batch[k])


def test_no_pref_keys_no_complementary_change():
    batch = {
        "observation.state": torch.zeros(2, 4, 9),
        "task": ["pick", "pick"],
    }
    transition = _rso_batch_to_transition(batch)
    comp = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
    assert not any(k.startswith("pref.") for k in comp)


def test_make_pre_post_processors_runs_without_dataset_stats():
    """Constructing the pipeline returns a tuple of (pre, post) without errors.

    This is a wiring test; we don't run the full pipeline (it would require
    encoder weights). We just confirm the factory returns and that the
    preprocessor's to_transition is wired to our custom converter.
    """
    from lerobot_policy_rewind_state_ours_variant7.configuration_rewind_state_ours_variant7 import (
        RewindStateOursVariant7Config,
    )
    from lerobot_policy_rewind_state_ours_variant7.processor_rewind_state_ours_variant7 import (
        make_rewind_state_ours_variant7_pre_post_processors,
    )
    cfg = RewindStateOursVariant7Config()
    cfg.finetune_encoder = True   # avoid frozen-DINOv2 path which needs ckpt download
    cfg.input_features = {}        # NormalizerProcessor accepts empty
    cfg.output_features = {}
    pre, post = make_rewind_state_ours_variant7_pre_post_processors(cfg, dataset_stats=None)
    assert pre is not None and post is not None
    # The wiring contract: our converter is the active to_transition.
    assert pre.to_transition is _rso_batch_to_transition


def test_pref_state_normalize_step_uses_same_stats():
    import torch
    from lerobot.processor.core import EnvTransition, TransitionKey
    from lerobot_policy_rewind_state_ours_variant7.processor_rewind_state_ours_variant7 import (
        _PrefStateNormalizeStep,
    )

    mean = torch.tensor([1.0, 2.0, 3.0])
    std = torch.tensor([0.5, 0.5, 0.5])
    step = _PrefStateNormalizeStep(mean=mean, std=std,
                                   keys=("pref.failure.observation.state",))

    raw = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]])  # (1, 2, 3)
    transition: EnvTransition = {
        TransitionKey.OBSERVATION: {},
        TransitionKey.COMPLEMENTARY_DATA: {"pref.failure.observation.state": raw.clone()},
    }
    out = step(transition)
    out_state = out[TransitionKey.COMPLEMENTARY_DATA]["pref.failure.observation.state"]
    expected = (raw - mean) / std
    assert torch.allclose(out_state, expected)


def test_pref_state_normalize_step_no_keys_present_is_noop():
    import torch
    from lerobot.processor.core import EnvTransition, TransitionKey
    from lerobot_policy_rewind_state_ours_variant7.processor_rewind_state_ours_variant7 import (
        _PrefStateNormalizeStep,
    )

    mean = torch.tensor([0.0, 0.0])
    std = torch.tensor([1.0, 1.0])
    step = _PrefStateNormalizeStep(mean=mean, std=std,
                                   keys=("pref.failure.observation.state",))
    transition: EnvTransition = {
        TransitionKey.OBSERVATION: {},
        TransitionKey.COMPLEMENTARY_DATA: {},
    }
    # Must not raise.
    out = step(transition)
    assert out is not None
