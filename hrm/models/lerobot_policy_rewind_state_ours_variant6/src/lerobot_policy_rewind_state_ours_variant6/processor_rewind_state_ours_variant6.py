"""
Processor pipeline for `rewind_state_ours`.

Mirrors `make_rewind_state_pre_post_processors` from the base package, plus a
custom batch_to_transition converter that lets namespaced `pref.*` keys
(failure and success preference clips) survive the OBSERVATION/COMP_DATA split
in `lerobot.processor.converters.batch_to_transition`.
"""
from __future__ import annotations

from typing import Any

import torch

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.core import TransitionKey
from lerobot.processor.pipeline import PipelineFeatureType
from lerobot.configs.types import PolicyFeature
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

# Reuse the existing rewind_state processor steps via direct module import to
# avoid duplicating them here. They are pure ProcessorStep subclasses.
from lerobot_policy_rewind_state.processor_rewind_state import (
    DINOImageEmbeddingProcessorStep,
    MiniLMTextEmbeddingProcessorStep,
    RawImageLengthsProcessorStep,
    StackCameraEmbeddingsProcessorStep,
)

from .configuration_rewind_state_ours_variant6 import RewindStateOursVariant6Config

_RSO_PASSTHROUGH_PREFIXES: tuple[str, ...] = ("pref.failure.", "pref.success.")


class _PrefStateNormalizeStep(ProcessorStep):
    """In-place mean/std normalize the pref.*.observation.state tensors.

    Reads/writes `COMPLEMENTARY_DATA[key]` (where the keys live after our custom
    converter routes them). Shares the `(mean, std)` tensors with the standard
    NormalizerProcessorStep applied to `observation.state`.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        keys: tuple[str, ...],
        eps: float = 1e-8,
    ):
        super().__init__()
        self._mean = mean
        self._std = std
        self._keys = tuple(keys)
        self._eps = eps

    def __call__(self, transition):
        new_t = transition.copy() if hasattr(transition, "copy") else dict(transition)
        comp = new_t.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        comp = dict(comp)
        for k in self._keys:
            if k not in comp:
                continue
            x = comp[k]
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"{k!r} must be torch.Tensor, got {type(x)}")
            mean = self._mean.to(device=x.device, dtype=x.dtype)
            std = self._std.to(device=x.device, dtype=x.dtype)
            comp[k] = (x - mean) / (std + self._eps)
        new_t[TransitionKey.COMPLEMENTARY_DATA] = comp
        return new_t

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def _rso_batch_to_transition(batch: dict[str, Any]):
    """Custom converter that preserves rewind_state_ours preference clips.

    The default `batch_to_transition` only routes keys starting with
    `observation.` to OBSERVATION; everything else is dropped or routed to
    typed slots. We thread the namespaced `pref.failure.*` and
    `pref.success.*` keys through COMPLEMENTARY_DATA so the model's
    `forward` can read them after the pipeline runs.
    """
    transition = batch_to_transition(batch)
    extras = {
        k: v for k, v in batch.items()
        if any(k.startswith(p) for p in _RSO_PASSTHROUGH_PREFIXES)
    }
    if extras:
        comp = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        comp.update(extras)
        transition[TransitionKey.COMPLEMENTARY_DATA] = comp
    return transition


def make_rewind_state_ours_variant6_pre_post_processors(
    config: RewindStateOursVariant6Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    finetune_encoder = bool(getattr(config, "finetune_encoder", False))

    image_keys = [
        key for key in config.input_features
        if key.startswith("observation.images.")
    ]

    norm_features = {
        k: v
        for k, v in {**config.input_features, **config.output_features}.items()
        if not k.startswith("observation.images.")
        and not k.startswith("observation.dinov2.")
    }

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
    ]

    if finetune_encoder:
        input_steps.append(
            RawImageLengthsProcessorStep(
                image_keys=image_keys,
                lengths_output_key="lengths",
            )
        )
    else:
        input_steps.extend([
            DINOImageEmbeddingProcessorStep(
                image_keys=image_keys,
                ckpt="facebook/dinov2-base",
                device=config.device,
            ),
            StackCameraEmbeddingsProcessorStep(
                image_keys=image_keys,
                output_key="img_seq",
                lengths_output_key="lengths",
                drop_input_keys=False,
            ),
        ])

    input_steps.extend([
        MiniLMTextEmbeddingProcessorStep(
            ckpt="sentence-transformers/all-MiniLM-L12-v2",
            device=config.device,
            text_key="task",
            output_key="task.minilm",
            read_from_observation=False,
        ),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features=norm_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ])

    # Pref-side state normalization: reuse the same (mean, std) tensors that
    # the standard NormalizerProcessorStep used for observation.state.
    if dataset_stats is not None and "observation.state" in dataset_stats:
        s_stats = dataset_stats["observation.state"]
        # NormalizerProcessorStep expects keys "mean" and "std" (or pre-loaded
        # tensors); we read them in the same way to stay consistent.
        s_mean = s_stats.get("mean") if isinstance(s_stats, dict) else getattr(s_stats, "mean", None)
        s_std = s_stats.get("std") if isinstance(s_stats, dict) else getattr(s_stats, "std", None)
        if s_mean is not None and s_std is not None:
            input_steps.append(
                _PrefStateNormalizeStep(
                    mean=torch.as_tensor(s_mean),
                    std=torch.as_tensor(s_std),
                    keys=(
                        "pref.failure.observation.state",
                        "pref.success.observation.state",
                    ),
                )
            )

    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            to_transition=_rso_batch_to_transition,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
