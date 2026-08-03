from typing import Any
import torch
from PIL import Image

from .configuration_rewind_state import RewindStateConfig
from .vision_encoder import FrozenDINOv2Encoder
from .text_encoder import FrozenMiniLMTextEncoder

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.processor import ProcessorStep
from lerobot.processor.pipeline import PipelineFeatureType
from lerobot.processor.core import EnvTransition, TransitionKey

from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

class CopyEmbeddingProcessorStep(ProcessorStep):
    """
    Copies a precomputed embedding from one observation key to another.

    Example:
      - input:  observation["task.minilm"]      -> (B, D) or (D,)
      - output: observation["text_features"]    -> (B, D)

    If input is unbatched (D,), it becomes (1, D).
    """

    def __init__(
        self,
        src_key: str,
        output_key: str,
        drop_input_key: bool = False,
    ):
        super().__init__()
        self.src_key = src_key
        self.output_key = output_key
        self.drop_input_key = drop_input_key

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy() if hasattr(transition, "copy") else dict(transition)
        observation = new_transition.get(TransitionKey.OBSERVATION)

        if observation is None:
            raise ValueError("Transition is missing OBSERVATION")

        if self.src_key not in observation:
            raise ValueError(
                f"Missing key {self.src_key!r}. Available keys: {list(observation.keys())}"
            )

        x = observation[self.src_key]

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"{self.src_key} must be a torch.Tensor, got {type(x)}")

        if x.ndim == 1:
            x = x.unsqueeze(0)   # (D,) -> (1, D)
        elif x.ndim == 2:
            pass                 # already (B, D)
        else:
            raise ValueError(
                f"Expected {self.src_key} to have shape (D,) or (B, D), got {tuple(x.shape)}"
            )

        observation[self.output_key] = x

        if self.drop_input_key:
            observation.pop(self.src_key, None)

        new_transition[TransitionKey.OBSERVATION] = observation
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
    
class MiniLMTextEmbeddingProcessorStep(ProcessorStep):
    """
    Encodes task/language text into frozen MiniLM embeddings.

    Reads text from:
      - COMPLEMENTARY_DATA[text_key]
      or optionally OBSERVATION[text_key]

    Writes:
      - OBSERVATION[output_key] with shape (B, D)

    Supported input forms:
      - str
      - list[str]
    """

    def __init__(
        self,
        ckpt: str,
        device: str,
        text_key: str = "task",
        output_key: str = "text_features",
        read_from_observation: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.text_key = text_key
        self.output_key = output_key
        self.read_from_observation = read_from_observation
        self.encoder = FrozenMiniLMTextEncoder(ckpt=ckpt, device=self.device)

    def _infer_batch_size(self, observation: dict[str, Any]) -> int:
        """
        Infer batch size from any tensor in observation.
        """
        for v in observation.values():
            if isinstance(v, torch.Tensor) and v.ndim >= 1:
                return v.shape[0]
        return 1

    def _get_texts(self, transition: EnvTransition, batch_size: int) -> list[str]:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        comp_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

        source = observation if self.read_from_observation else comp_data
        text = source.get(self.text_key)

        if text is None:
            raise ValueError(
                f"Could not find text under "
                f"{'OBSERVATION' if self.read_from_observation else 'COMPLEMENTARY_DATA'}[{self.text_key!r}]"
            )

        if isinstance(text, str):
            return [text] * batch_size

        if isinstance(text, list):
            if len(text) == batch_size:
                return [str(t) for t in text]
            if len(text) == 1:
                return [str(text[0])] * batch_size
            raise ValueError(
                f"Text list length {len(text)} does not match batch size {batch_size}"
            )

        return [str(text)] * batch_size

    @torch.no_grad()
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy() if hasattr(transition, "copy") else dict(transition)
        observation = new_transition.get(TransitionKey.OBSERVATION)

        if observation is None:
            raise ValueError("Transition is missing OBSERVATION")

        batch_size = self._infer_batch_size(observation)
        texts = self._get_texts(new_transition, batch_size)
        observation[self.output_key] = self.encoder.encode_text(texts)

        new_transition[TransitionKey.OBSERVATION] = observation
        return new_transition
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
    

class DINOImageEmbeddingProcessorStep(ProcessorStep):
    """
    Replaces raw image observations with frozen DINOv2 embeddings.

    Expected input for each image key:
      - (B, T, C, H, W) or
      - (B, C, H, W)

    Output:
      - (B, T, D) or
      - (B, D)

    Also writes:
      - observation["lengths"]: (B,)
        derived from a pad mask if available, otherwise full sequence length
    """

    def __init__(
        self,
        image_keys: list[str],
        ckpt: str,
        device: str,
        lengths_pad_mask_key: str = "observation.state_is_pad",
        lengths_output_key: str = "lengths",
    ):
        super().__init__()
        self.image_keys = image_keys
        self.device = torch.device(device)
        self.encoder = FrozenDINOv2Encoder(ckpt=ckpt, device=self.device)
        self.lengths_pad_mask_key = lengths_pad_mask_key
        self.lengths_output_key = lengths_output_key

    def _tensor_to_pil_list(self, x: torch.Tensor) -> list[Image.Image]:
        if x.ndim != 4:
            raise ValueError(f"Expected (N, C, H, W), got {tuple(x.shape)}")

        x = x.detach().cpu()

        if x.dtype.is_floating_point:
            x = x.clamp(0, 1).mul(255).to(torch.uint8)
        else:
            x = x.to(torch.uint8)

        pil_images = []
        for img in x:
            img = img.permute(1, 2, 0).contiguous().numpy()
            if img.shape[-1] == 1:
                img = img.repeat(3, axis=-1)
            pil_images.append(Image.fromarray(img))
        return pil_images

    @torch.no_grad()
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy() if hasattr(transition, "copy") else dict(transition)
        observation = new_transition.get(TransitionKey.OBSERVATION)

        if observation is None:
            raise ValueError("Transition is missing OBSERVATION")

        inferred_lengths = None

        for key in self.image_keys:
            if key not in observation:
                continue

            x = observation[key]

            if not isinstance(x, torch.Tensor):
                raise TypeError(f"{key} must be a torch.Tensor, got {type(x)}")

            if x.ndim == 5:
                B, T, C, H, W = x.shape
                flat = x.reshape(B * T, C, H, W)
                pil_images = self._tensor_to_pil_list(flat)
                emb = self.encoder.encode_image(pil_images)  # (B*T, D)
                observation[key] = emb.reshape(B, T, -1)

                if inferred_lengths is None:
                    if self.lengths_pad_mask_key in observation:
                        pad_mask = observation[self.lengths_pad_mask_key]
                        if not isinstance(pad_mask, torch.Tensor) or pad_mask.ndim != 2:
                            raise ValueError(
                                f"{self.lengths_pad_mask_key} must be a tensor of shape (B, T), "
                                f"got {type(pad_mask)} with shape {getattr(pad_mask, 'shape', None)}"
                            )
                        # assumes True = padded
                        inferred_lengths = (~pad_mask.bool()).sum(dim=1).long()
                    else:
                        inferred_lengths = torch.full(
                            (B,), T, dtype=torch.long, device=observation[key].device
                        )

            elif x.ndim == 4:
                pil_images = self._tensor_to_pil_list(x)
                emb = self.encoder.encode_image(pil_images)  # (B, D)
                observation[key] = emb

                if inferred_lengths is None:
                    B = emb.shape[0]
                    inferred_lengths = torch.ones(B, dtype=torch.long, device=emb.device)

            else:
                raise ValueError(
                    f"{key} must have shape (B,T,C,H,W) or (B,C,H,W), got {tuple(x.shape)}"
                )

        if inferred_lengths is not None:
            observation[self.lengths_output_key] = inferred_lengths

        new_transition[TransitionKey.OBSERVATION] = observation
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

class StackCameraEmbeddingsProcessorStep(ProcessorStep):
    """
    Stacks per-camera embeddings into a single img_seq tensor.

    Input camera embeddings:
      - each key: (B, D) or (B, T, D)

    Output:
      - observation[output_key]: (B, N, T, D)
      - observation[lengths_output_key]: (B,)
    """

    def __init__(
        self,
        image_keys: list[str],
        output_key: str = "img_seq",
        lengths_output_key: str = "lengths",
        drop_input_keys: bool = False,
    ):
        super().__init__()
        self.image_keys = image_keys
        self.output_key = output_key
        self.lengths_output_key = lengths_output_key
        self.drop_input_keys = drop_input_keys

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy() if hasattr(transition, "copy") else dict(transition)
        observation = new_transition.get(TransitionKey.OBSERVATION)

        if observation is None:
            raise ValueError("Transition is missing OBSERVATION")

        present_keys = [k for k in self.image_keys if k in observation]
        if not present_keys:
            raise ValueError(
                f"No image keys found among {self.image_keys}, available keys: {list(observation.keys())}"
            )

        cams = []
        inferred_lengths = None

        for key in present_keys:
            cam = observation[key]

            if not isinstance(cam, torch.Tensor):
                raise TypeError(f"{key} must be a torch.Tensor, got {type(cam)}")

            if cam.ndim == 2:
                # (B, D) -> (B, 1, D)
                B = cam.shape[0]
                cam = cam.unsqueeze(1)
                if inferred_lengths is None:
                    inferred_lengths = torch.ones(B, dtype=torch.long, device=cam.device)

            elif cam.ndim == 3:
                # (B, T, D)
                B, T, _ = cam.shape
                if inferred_lengths is None:
                    inferred_lengths = torch.full((B,), T, dtype=torch.long, device=cam.device)

            else:
                raise ValueError(
                    f"Expected camera embedding {key} to have shape (B, D) or (B, T, D), got {tuple(cam.shape)}"
                )

            cams.append(cam)

        observation[self.output_key] = torch.stack(cams, dim=1)  # (B, N, T, D)

        if inferred_lengths is not None:
            observation[self.lengths_output_key] = inferred_lengths

        if self.drop_input_keys:
            for key in present_keys:
                observation.pop(key, None)

        new_transition[TransitionKey.OBSERVATION] = observation
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


class RawImageLengthsProcessorStep(ProcessorStep):
    """
    Derive `lengths` from a raw image tensor and leave the images untouched.

    Used when the downstream model runs its own (finetuned) image encoder and
    consumes observation.images.* directly, so the preprocessor must not encode
    or stack anything — it only needs to publish a `lengths` tensor for masking.

    Expected input for each image key:
      - (B, T, C, H, W)

    Output:
      - observation[lengths_output_key]: (B,)
        derived from a pad mask if available, otherwise full sequence length.
    """

    def __init__(
        self,
        image_keys: list[str],
        lengths_output_key: str = "lengths",
        lengths_pad_mask_key: str = "observation.state_is_pad",
    ):
        super().__init__()
        self.image_keys = list(image_keys)
        self.lengths_output_key = lengths_output_key
        self.lengths_pad_mask_key = lengths_pad_mask_key

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy() if hasattr(transition, "copy") else dict(transition)
        observation = new_transition.get(TransitionKey.OBSERVATION)

        if observation is None:
            raise ValueError("Transition is missing OBSERVATION")

        chosen = None
        for k in self.image_keys:
            if k in observation:
                chosen = observation[k]
                break
        if chosen is None:
            raise ValueError(
                f"No image key found among {self.image_keys}, "
                f"available keys: {list(observation.keys())}"
            )
        if not isinstance(chosen, torch.Tensor) or chosen.ndim != 5:
            raise ValueError(
                f"Expected raw image tensor of shape (B, T, C, H, W), "
                f"got {type(chosen)} with shape {getattr(chosen, 'shape', None)}"
            )

        B, T = chosen.shape[0], chosen.shape[1]

        if self.lengths_pad_mask_key in observation:
            pad_mask = observation[self.lengths_pad_mask_key]
            if not isinstance(pad_mask, torch.Tensor) or pad_mask.ndim != 2:
                raise ValueError(
                    f"{self.lengths_pad_mask_key} must be a tensor of shape (B, T), "
                    f"got {type(pad_mask)} with shape {getattr(pad_mask, 'shape', None)}"
                )
            lengths = (~pad_mask.bool()).sum(dim=1).long()
        else:
            lengths = torch.full((B,), T, dtype=torch.long, device=chosen.device)

        observation[self.lengths_output_key] = lengths
        new_transition[TransitionKey.OBSERVATION] = observation
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_rewind_state_pre_post_processors(
    config: RewindStateConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    # Two mutually exclusive modes:
    #   - finetune_encoder=True  → model encodes raw images online with its own DINOv2.
    #                              Preprocessor must pass observation.images.* through
    #                              untouched and only compute `lengths` for masking.
    #   - finetune_encoder=False → model consumes stacked frozen-DINOv2 features as
    #                              `img_seq`. Preprocessor encodes observation.images.*
    #                              with a frozen DINOv2 and stacks cameras into img_seq.
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
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )