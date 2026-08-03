import random
from collections.abc import Callable
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class RewindDataset(LeRobotDataset):
    """
    A clip-sampling dataset built on top of LeRobotDataset.

    Supports two modes:

    1. Raw-image mode
       - loads observation.images.* clips
       - optionally also loads precomputed embedding clips

    2. Embedding-only mode
       - skips video/image loading entirely
       - loads only precomputed embeddings such as:
         - observation.dinov2.front
         - observation.dinov2.wrist_left
         - observation.dinov2.wrist_right
         - task.minilm

    Output keys:
      - task
      - reward: Tensor [T]
      - class_label: Tensor [T]
      - episode_index: int
      - camera_keys: list[str]
      - is_rewind: bool
      - is_negative: bool
      - optional raw image keys: observation.images.*
      - optional embedding keys: observation.dinov2.*
      - optional text embedding key: task.minilm
    """

    def __init__(
        self,
        *args,
        camera_keys: str | list[str] | None = None,
        embedding_keys: str | list[str] | None = None,
        text_embedding_key: str | None = "task.minilm",
        state_key: str | None = "observation.state",
        use_precomputed_embeddings_only: bool = False,
        min_clip_len: int = 3,
        max_clip_len: int | None = None,
        rewind: bool = True,
        rewind_ratio: float = 0.5,
        sample_negative: bool = False,
        negative_ratio: float = 0.2,
        subsample_video: bool = False,
        pad_to_length: int | None = None,
        task_transform: Callable[[str], object] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.min_clip_len = min_clip_len
        self.max_clip_len = max_clip_len
        self.rewind = rewind
        self.rewind_ratio = rewind_ratio
        self.sample_negative = sample_negative
        self.negative_ratio = negative_ratio
        self.subsample_video = subsample_video
        self.pad_to_length = pad_to_length
        self.task_transform = task_transform
        self.use_precomputed_embeddings_only = use_precomputed_embeddings_only

        available_features = set(self.meta.features.keys())
        available_cameras = list(self.meta.camera_keys)

        if embedding_keys is None:
            self.embedding_keys = []
        elif isinstance(embedding_keys, str):
            self.embedding_keys = [embedding_keys]
        else:
            self.embedding_keys = list(embedding_keys)

        invalid_embedding_keys = [k for k in self.embedding_keys if k not in available_features]
        if invalid_embedding_keys:
            raise ValueError(
                f"Invalid embedding_keys={invalid_embedding_keys}. "
                f"Available features: {sorted(available_features)}"
            )

        self.text_embedding_key = text_embedding_key
        if self.text_embedding_key is not None and self.text_embedding_key not in available_features:
            raise ValueError(
                f"text_embedding_key={self.text_embedding_key!r} not found in dataset features."
            )

        self.state_key = state_key
        if self.state_key is not None and self.state_key not in available_features:
            raise ValueError(
                f"state_key={self.state_key!r} not found in dataset features."
            )

        if self.use_precomputed_embeddings_only:
            self.camera_keys = []
            if len(self.embedding_keys) == 0:
                raise ValueError(
                    "use_precomputed_embeddings_only=True requires embedding_keys to be provided."
                )
        else:
            if len(available_cameras) == 0:
                raise ValueError("RewindDataset requires at least one camera/image/video feature.")

            if camera_keys is None:
                self.camera_keys = [available_cameras[0]]
            elif isinstance(camera_keys, str):
                self.camera_keys = [camera_keys]
            else:
                self.camera_keys = list(camera_keys)

            invalid_camera_keys = [k for k in self.camera_keys if k not in available_cameras]
            if invalid_camera_keys:
                raise ValueError(
                    f"Invalid camera_keys={invalid_camera_keys}. "
                    f"Available camera keys: {available_cameras}"
                )

        if self.min_clip_len < 2:
            raise ValueError("min_clip_len must be >= 2")
        if self.max_clip_len is not None and self.max_clip_len < self.min_clip_len:
            raise ValueError("max_clip_len must be >= min_clip_len")
        if self.subsample_video and self.pad_to_length is None:
            raise ValueError("When subsample_video=True, pad_to_length must be set.")

    def __len__(self):
        return super().__len__()

    def _get_episode_bounds(self, ep_idx: int) -> tuple[int, int]:
        ep = self.meta.episodes[ep_idx]
        start = int(ep["dataset_from_index"])
        end = int(ep["dataset_to_index"])
        return start, end

    def _abs_to_rel_indices(self, abs_indices: list[int]) -> list[int]:
        if self._absolute_to_relative_idx is None:
            return abs_indices
        return [self._absolute_to_relative_idx[i] for i in abs_indices]

    def _sample_episode_index(self) -> int:
        if self.episodes is not None:
            return random.choice(self.episodes)
        return random.randint(0, self.meta.total_episodes - 1)

    def _get_episode_task(self, ep_idx: int):
        start, _ = self._get_episode_bounds(ep_idx)
        rel_start = self._abs_to_rel_indices([start])[0]
        item = self.hf_dataset[rel_start]
        task_idx = (
            item["task_index"].item()
            if isinstance(item["task_index"], torch.Tensor)
            else int(item["task_index"])
        )
        task = self.meta.tasks.iloc[task_idx].name
        if self.task_transform is not None:
            task = self.task_transform(task)
        return task

    def _sample_negative_task(self, avoid_ep_idx: int):
        candidate_eps = self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        if len(candidate_eps) <= 1:
            raise ValueError("Need at least 2 episodes to sample a negative task.")

        neg_ep_idx = random.choice(candidate_eps)
        while neg_ep_idx == avoid_ep_idx:
            neg_ep_idx = random.choice(candidate_eps)

        return self._get_episode_task(neg_ep_idx)

    def _load_single_camera_frames(
        self, ep_idx: int, abs_indices: list[int], camera_key: str
    ) -> torch.Tensor:
        rel_indices = self._abs_to_rel_indices(abs_indices)

        if camera_key in self.meta.video_keys:
            timestamps = self.hf_dataset[rel_indices]["timestamp"]
            timestamps = [
                ts.item() if isinstance(ts, torch.Tensor) else float(ts)
                for ts in timestamps
            ]
            query_timestamps = {camera_key: timestamps}
            frames = self._query_videos(query_timestamps, ep_idx)[camera_key]
            return frames

        items = self.hf_dataset[rel_indices][camera_key]
        if isinstance(items, torch.Tensor):
            return items
        return torch.stack(items)

    def _load_frames_from_indices(
        self, ep_idx: int, abs_indices: list[int], camera_keys: list[str]
    ) -> dict[str, torch.Tensor]:
        frames_dict = {}
        for camera_key in camera_keys:
            frames_dict[camera_key] = self._load_single_camera_frames(ep_idx, abs_indices, camera_key)
        return frames_dict
    
    def _load_embedding_features_from_indices(
        self, abs_indices: list[int], embedding_keys: list[str]
    ) -> dict[str, torch.Tensor]:
        if len(embedding_keys) == 0:
            return {}

        rel_indices = self._abs_to_rel_indices(abs_indices)
        batch = self.hf_dataset[rel_indices]

        out = {}
        for key in embedding_keys:
            items = batch[key]
            if isinstance(items, torch.Tensor):
                out[key] = items.float()
            elif len(items) > 0 and isinstance(items[0], torch.Tensor):
                out[key] = torch.stack(items).float()
            else:
                out[key] = torch.as_tensor(np.asarray(items), dtype=torch.float32)
        return out


    def _load_feature_from_indices(self, abs_indices: list[int], feature_key: str) -> torch.Tensor:
        rel_indices = self._abs_to_rel_indices(abs_indices)
        items = self.hf_dataset[rel_indices][feature_key]

        if isinstance(items, torch.Tensor):
            return items

        if len(items) == 0:
            raise ValueError(f"No values found for feature_key={feature_key!r}")

        first = items[0]
        if isinstance(first, torch.Tensor):
            return torch.stack(items)

        return torch.as_tensor(np.asarray(items))

    def _load_embedding_features_from_indices(
        self, abs_indices: list[int], embedding_keys: list[str]
    ) -> dict[str, torch.Tensor]:
        features_dict = {}
        for key in embedding_keys:
            features_dict[key] = self._load_feature_from_indices(abs_indices, key).float()
        return features_dict

    def _load_text_embedding(self, ep_idx: int) -> torch.Tensor | None:
        if self.text_embedding_key is None:
            return None

        start, _ = self._get_episode_bounds(ep_idx)
        rel_start = self._abs_to_rel_indices([start])[0]
        item = self.hf_dataset[rel_start]
        value = item[self.text_embedding_key]

        if isinstance(value, torch.Tensor):
            return value.float()

        return torch.as_tensor(np.asarray(value), dtype=torch.float32)

    def _sample_clip_indices(self, ep_idx: int) -> tuple[list[int], int, int, int]:
        ep_start, ep_end = self._get_episode_bounds(ep_idx)
        ep_len = ep_end - ep_start

        if ep_len < self.min_clip_len:
            raise ValueError(
                f"Episode {ep_idx} is too short ({ep_len} frames) for min_clip_len={self.min_clip_len}."
            )

        start_abs = random.randint(ep_start, ep_end - self.min_clip_len)
        min_end_abs = start_abs + self.min_clip_len
        max_end_abs = ep_end
        if self.max_clip_len is not None:
            max_end_abs = min(max_end_abs, start_abs + self.max_clip_len)

        end_abs = random.randint(min_end_abs, max_end_abs)
        abs_indices = list(range(start_abs, end_abs))
        return abs_indices, ep_start, ep_end, start_abs

    def _sample_forward_clip(self, ep_idx: int, camera_keys: list[str]):
        abs_indices, _, ep_end, start_abs = self._sample_clip_indices(ep_idx)

        frames_dict = {}
        if not self.use_precomputed_embeddings_only and len(camera_keys) > 0:
            frames_dict = self._load_frames_from_indices(ep_idx, abs_indices, camera_keys)

        embedding_dict = self._load_embedding_features_from_indices(abs_indices, self.embedding_keys)

        state = None
        if self.state_key is not None:
            state = self._load_feature_from_indices(abs_indices, self.state_key).float()

        full_length = ep_end - start_abs
        progress = np.arange(len(abs_indices), dtype=np.float32) / max(full_length - 1, 1)
        class_label = np.ones(len(abs_indices), dtype=np.float32)

        return frames_dict, embedding_dict, state, progress, class_label

    def _sample_rewind_clip(self, ep_idx: int, camera_keys: list[str]):
        ep_start, ep_end = self._get_episode_bounds(ep_idx)
        ep_len = ep_end - ep_start

        if ep_len < self.min_clip_len:
            raise ValueError(
                f"Episode {ep_idx} is too short ({ep_len} frames) for min_clip_len={self.min_clip_len}."
            )

        first_half_end = ep_start + max(ep_len // 2, 1)
        second_half_start = ep_start + min(ep_len // 2, ep_len - 1)

        start_abs = random.randint(ep_start, first_half_end - 1)
        end_abs = random.randint(second_half_start + 1, ep_end)

        while end_abs - start_abs < self.min_clip_len:
            start_abs = random.randint(ep_start, first_half_end - 1)
            end_abs = random.randint(second_half_start + 1, ep_end)

        if self.max_clip_len is not None:
            end_abs = min(end_abs, start_abs + self.max_clip_len)

        forward_indices = list(range(start_abs, end_abs))

        forward_frames_dict = {}
        if not self.use_precomputed_embeddings_only and len(camera_keys) > 0:
            forward_frames_dict = self._load_frames_from_indices(ep_idx, forward_indices, camera_keys)

        forward_embedding_dict = self._load_embedding_features_from_indices(forward_indices, self.embedding_keys)

        forward_state = None
        if self.state_key is not None:
            forward_state = self._load_feature_from_indices(forward_indices, self.state_key).float()

        full_length = ep_end - start_abs
        forward_progress = np.arange(len(forward_indices), dtype=np.float32) / max(full_length - 1, 1)

        if len(forward_indices) < 3:
            raise RuntimeError("Unexpected short forward clip after rewind sampling.")

        selected_end_point = random.randint(2, len(forward_indices))

        frames_dict = {}
        for camera_key, forward_frames in forward_frames_dict.items():
            reverse_frames = torch.flip(forward_frames, dims=[0])[1:selected_end_point]
            frames_dict[camera_key] = torch.cat([forward_frames, reverse_frames], dim=0)

        embedding_dict = {}
        for key, forward_emb in forward_embedding_dict.items():
            reverse_emb = torch.flip(forward_emb, dims=[0])[1:selected_end_point]
            embedding_dict[key] = torch.cat([forward_emb, reverse_emb], dim=0)

        state = None
        if forward_state is not None:
            reverse_state = torch.flip(forward_state, dims=[0])[1:selected_end_point]
            state = torch.cat([forward_state, reverse_state], dim=0)

        reverse_progress = forward_progress[::-1][1:selected_end_point]
        progress = np.concatenate([forward_progress, reverse_progress], axis=0)
        class_label = np.ones(len(progress), dtype=np.float32)

        return frames_dict, embedding_dict, state, progress, class_label

    def _pad_or_subsample_sequence_dict(
        self,
        seq_dict: dict[str, torch.Tensor],
        target_len: int,
    ) -> dict[str, torch.Tensor]:
        if len(seq_dict) == 0:
            return seq_dict

        first_key = next(iter(seq_dict))
        current_len = len(seq_dict[first_key])

        if current_len < target_len:
            pad_n = target_len - current_len
            for key, x in seq_dict.items():
                last = x[-1:].repeat((pad_n,) + (1,) * (x.ndim - 1))
                seq_dict[key] = torch.cat([x, last], dim=0)
        elif current_len > target_len:
            idx = np.linspace(0, current_len - 1, target_len).astype(int)
            for key, x in seq_dict.items():
                seq_dict[key] = x[idx]

        return seq_dict

    def __getitem__(self, idx):
        self._ensure_hf_dataset_loaded()

        ep_idx = self._sample_episode_index()
        camera_keys = self.camera_keys

        do_rewind = self.rewind and (random.random() < self.rewind_ratio)
        if do_rewind:
            frames_dict, embedding_dict, state, progress, class_label = self._sample_rewind_clip(ep_idx, camera_keys)
        else:
            frames_dict, embedding_dict, state, progress, class_label = self._sample_forward_clip(ep_idx, camera_keys)

        is_negative = False
        if self.sample_negative and random.random() < self.negative_ratio:
            task = self._sample_negative_task(ep_idx)
            progress = np.zeros_like(progress, dtype=np.float32)
            class_label = np.zeros_like(class_label, dtype=np.float32)
            is_negative = True
        else:
            task = self._get_episode_task(ep_idx)

        text_embedding = self._load_text_embedding(ep_idx)

        if self.subsample_video:
            if len(frames_dict) > 0:
                frames_dict = self._pad_or_subsample_sequence_dict(frames_dict, self.pad_to_length)
            if len(embedding_dict) > 0:
                embedding_dict = self._pad_or_subsample_sequence_dict(embedding_dict, self.pad_to_length)
            if state is not None:
                state_dict = {self.state_key: state}
                state_dict = self._pad_or_subsample_sequence_dict(state_dict, self.pad_to_length)
                state = state_dict[self.state_key]

            target_len = self.pad_to_length
            current_len = len(progress)
            if current_len < target_len:
                pad_n = target_len - current_len
                progress = np.concatenate(
                    [progress, np.full((pad_n,), progress[-1], dtype=progress.dtype)], axis=0
                )
                class_label = np.concatenate(
                    [class_label, np.full((pad_n,), class_label[-1], dtype=class_label.dtype)], axis=0
                )
            elif current_len > target_len:
                idx = np.linspace(0, current_len - 1, target_len).astype(int)
                progress = progress[idx]
                class_label = class_label[idx]

        output = {
            "task": task,
            "reward": torch.as_tensor(progress, dtype=torch.float32),
            "class_label": torch.as_tensor(class_label, dtype=torch.float32),
            "episode_index": ep_idx,
            "camera_keys": camera_keys,
            "is_rewind": do_rewind,
            "is_negative": is_negative,
        }

        for camera_key, video_array in frames_dict.items():
            output[camera_key] = video_array.float()

        for key, value in embedding_dict.items():
            output[key] = value.float()

        if state is not None:
            output[self.state_key] = state.float()

        if text_embedding is not None:
            output[self.text_embedding_key] = text_embedding.float()

        return output


import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class EpisodeDataset(LeRobotDataset):
    def __init__(
        self,
        *args,
        camera_keys: str | list[str] | None = None,
        embedding_keys: str | list[str] | None = None,
        text_embedding_key: str | None = None,
        state_key: str | None = "observation.state",
        use_precomputed_embeddings_only: bool = False,
        pad_to_length: int = 64,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.use_precomputed_embeddings_only = use_precomputed_embeddings_only
        print("[EpisodeDataset] use_precomputed_embeddings_only:", self.use_precomputed_embeddings_only)
        self.pad_to_length = pad_to_length

        available_features = set(self.meta.features.keys())
        available_cameras = list(self.meta.camera_keys)

        if embedding_keys is None:
            self.embedding_keys = []
        elif isinstance(embedding_keys, str):
            self.embedding_keys = [embedding_keys]
        else:
            self.embedding_keys = list(embedding_keys)

        invalid_embedding_keys = [k for k in self.embedding_keys if k not in available_features]
        if invalid_embedding_keys:
            raise ValueError(
                f"Invalid embedding_keys={invalid_embedding_keys}. "
                f"Available features: {sorted(available_features)}"
            )

        self.text_embedding_key = text_embedding_key
        if self.text_embedding_key is not None and self.text_embedding_key not in available_features:
            raise ValueError(
                f"text_embedding_key={self.text_embedding_key!r} not found in dataset features."
            )

        self.state_key = state_key
        if self.state_key is not None and self.state_key not in available_features:
            raise ValueError(
                f"state_key={self.state_key!r} not found in dataset features."
            )

        if self.use_precomputed_embeddings_only:
            self.camera_keys = []
            if len(self.embedding_keys) == 0:
                raise ValueError(
                    "use_precomputed_embeddings_only=True requires embedding_keys."
                )
        else:
            if camera_keys is None:
                self.camera_keys = available_cameras
            elif isinstance(camera_keys, str):
                self.camera_keys = [camera_keys]
            else:
                self.camera_keys = list(camera_keys)

            invalid_camera_keys = [k for k in self.camera_keys if k not in available_cameras]
            if invalid_camera_keys:
                raise ValueError(
                    f"Invalid camera_keys={invalid_camera_keys}. "
                    f"Available camera keys: {available_cameras}"
                )
            print("Using camera keys:", self.camera_keys)
        

    def __len__(self):
        if self.episodes is not None:
            return len(self.episodes)
        return self.meta.total_episodes

    def _idx_to_episode_index(self, idx: int) -> int:
        if self.episodes is not None:
            return int(self.episodes[idx])
        return int(idx)

    def _get_episode_bounds(self, ep_idx: int) -> tuple[int, int]:
        ep = self.meta.episodes[ep_idx]
        start = int(ep["dataset_from_index"])
        end = int(ep["dataset_to_index"])
        return start, end

    def _abs_to_rel_indices(self, abs_indices: list[int]) -> list[int]:
        if self._absolute_to_relative_idx is None:
            return abs_indices
        return [self._absolute_to_relative_idx[i] for i in abs_indices]

    def _get_episode_task(self, ep_idx: int):
        start, _ = self._get_episode_bounds(ep_idx)
        rel_start = self._abs_to_rel_indices([start])[0]
        item = self.hf_dataset[rel_start]
        task_idx = (
            item["task_index"].item()
            if isinstance(item["task_index"], torch.Tensor)
            else int(item["task_index"])
        )
        return self.meta.tasks.iloc[task_idx].name

    def _load_single_camera_frames(
        self, ep_idx: int, abs_indices: list[int], camera_key: str
    ) -> torch.Tensor:
        rel_indices = self._abs_to_rel_indices(abs_indices)

        if camera_key in self.meta.video_keys:
            timestamps = self.hf_dataset[rel_indices]["timestamp"]
            timestamps = [
                ts.item() if isinstance(ts, torch.Tensor) else float(ts)
                for ts in timestamps
            ]
            query_timestamps = {camera_key: timestamps}
            return self._query_videos(query_timestamps, ep_idx)[camera_key]

        items = self.hf_dataset[rel_indices][camera_key]
        if isinstance(items, torch.Tensor):
            return items
        return torch.stack(items)

    def _load_frames_from_indices(
        self, ep_idx: int, abs_indices: list[int], camera_keys: list[str]
    ) -> dict[str, torch.Tensor]:
        out = {}
        for key in camera_keys:
            out[key] = self._load_single_camera_frames(ep_idx, abs_indices, key)
        return out

    def _load_feature_from_indices(self, abs_indices: list[int], feature_key: str) -> torch.Tensor:
        rel_indices = self._abs_to_rel_indices(abs_indices)
        items = self.hf_dataset[rel_indices][feature_key]

        if isinstance(items, torch.Tensor):
            return items

        if len(items) == 0:
            raise ValueError(f"No values found for feature_key={feature_key!r}")

        first = items[0]
        if isinstance(first, torch.Tensor):
            return torch.stack(items)

        return torch.as_tensor(np.asarray(items))

    def _load_embedding_features_from_indices(
        self, abs_indices: list[int], embedding_keys: list[str]
    ) -> dict[str, torch.Tensor]:
        out = {}
        for key in embedding_keys:
            out[key] = self._load_feature_from_indices(abs_indices, key).float()
        return out

    def _load_text_embedding(self, ep_idx: int) -> torch.Tensor | None:
        if self.text_embedding_key is None:
            return None

        start, _ = self._get_episode_bounds(ep_idx)
        rel_start = self._abs_to_rel_indices([start])[0]
        item = self.hf_dataset[rel_start]
        value = item[self.text_embedding_key]

        if isinstance(value, torch.Tensor):
            return value.float()
        return torch.as_tensor(np.asarray(value), dtype=torch.float32)

    def _pad_or_subsample_sequence_dict(
        self,
        seq_dict: dict[str, torch.Tensor],
        target_len: int,
    ) -> dict[str, torch.Tensor]:
        if len(seq_dict) == 0:
            return seq_dict

        first_key = next(iter(seq_dict))
        current_len = len(seq_dict[first_key])

        if current_len < target_len:
            pad_n = target_len - current_len
            for key, x in seq_dict.items():
                last = x[-1:].repeat((pad_n,) + (1,) * (x.ndim - 1))
                seq_dict[key] = torch.cat([x, last], dim=0)
        elif current_len > target_len:
            take_idx = np.linspace(0, current_len - 1, target_len).astype(int)
            for key, x in seq_dict.items():
                seq_dict[key] = x[take_idx]

        return seq_dict

    def __getitem__(self, idx):
        self._ensure_hf_dataset_loaded()

        ep_idx = self._idx_to_episode_index(idx)
        ep_start, ep_end = self._get_episode_bounds(ep_idx)
        abs_indices = list(range(ep_start, ep_end))

        frames_dict = {}
        if not self.use_precomputed_embeddings_only and len(self.camera_keys) > 0:
            frames_dict = self._load_frames_from_indices(ep_idx, abs_indices, self.camera_keys)

        embedding_dict = self._load_embedding_features_from_indices(abs_indices, self.embedding_keys)
        state = None
        if self.state_key is not None:
            state = self._load_feature_from_indices(abs_indices, self.state_key).float()
        text_embedding = self._load_text_embedding(ep_idx)
        task = self._get_episode_task(ep_idx)

        ep_len = len(abs_indices)
        reward = np.arange(ep_len, dtype=np.float32) / max(ep_len - 1, 1)
        class_label = np.ones(ep_len, dtype=np.float32)

        if self.pad_to_length is not None:
            if len(frames_dict) > 0:
                frames_dict = self._pad_or_subsample_sequence_dict(frames_dict, self.pad_to_length)
            if len(embedding_dict) > 0:
                embedding_dict = self._pad_or_subsample_sequence_dict(embedding_dict, self.pad_to_length)
            if state is not None:
                state_dict = {self.state_key: state}
                state_dict = self._pad_or_subsample_sequence_dict(state_dict, self.pad_to_length)
                state = state_dict[self.state_key]

            target_len = self.pad_to_length
            if ep_len < target_len:
                pad_n = target_len - ep_len
                reward = np.concatenate(
                    [reward, np.full((pad_n,), reward[-1], dtype=reward.dtype)], axis=0
                )
                class_label = np.concatenate(
                    [class_label, np.full((pad_n,), class_label[-1], dtype=class_label.dtype)], axis=0
                )
            elif ep_len > target_len:
                take_idx = np.linspace(0, ep_len - 1, target_len).astype(int)
                reward = reward[take_idx]
                class_label = class_label[take_idx]

        output = {
            "task": task,
            "reward": torch.as_tensor(reward, dtype=torch.float32),
            "class_label": torch.as_tensor(class_label, dtype=torch.float32),
            "episode_index": ep_idx,
            "camera_keys": self.camera_keys,
        }

        for camera_key, video_array in frames_dict.items():
            output[camera_key] = video_array.float()

        for key, value in embedding_dict.items():
            output[key] = value.float()

        if state is not None:
            output[self.state_key] = state.float()

        if text_embedding is not None:
            output[self.text_embedding_key] = text_embedding.float()

        return output



if __name__ == "__main__":
    repo_id = "chomeed/anubis_pick_carrot_out_of_pot_randomized_train"

    dataset = RewindDataset(
        repo_id=repo_id,
        embedding_keys=[
            "observation.dinov2.front",
            "observation.dinov2.wrist_left",
            "observation.dinov2.wrist_right",
        ],
        text_embedding_key="task.minilm",
        use_precomputed_embeddings_only=True,
        rewind=True,
        subsample_video=True,
        pad_to_length=64,
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))

    print(batch.keys())
    print(batch["observation.dinov2.front"].shape)
    print(batch["observation.dinov2.wrist_left"].shape)
    print(batch["observation.dinov2.wrist_right"].shape)
    print(batch["task.minilm"].shape)
    print(batch["reward"].shape)
    print(batch["class_label"].shape)