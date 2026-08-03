"""
JSON schema and I/O helpers for the preference-pair manifest produced by
precompute_preference_pairs.py and consumed by PreferenceTaggedRewindDataset.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_REQUIRED_TOP_LEVEL = (
    "base_ckpt",
    "success_repo_id",
    "failure_repo_id",
    "clip_len",
    "frame_stride",
    "smooth_window",
    "top_k",
    "pairs",
)


@dataclass(frozen=True)
class PreferenceManifest:
    base_ckpt: str
    success_repo_id: str
    failure_repo_id: str
    clip_len: int
    frame_stride: int
    smooth_window: int
    top_k: int
    pairs: list[dict[str, Any]]

    def assert_compatible(
        self,
        base_ckpt: str,
        clip_len: int,
        frame_stride: int,
    ) -> None:
        # Resolve both sides through Path.resolve() so symlinks and
        # Path-vs-str differences don't cause spurious mismatches.
        def _canonical(p) -> str:
            try:
                return str(Path(p).resolve())
            except (OSError, RuntimeError):
                return str(p)
        manifest_resolved = _canonical(self.base_ckpt)
        caller_resolved = _canonical(base_ckpt)
        if manifest_resolved != caller_resolved:
            raise ValueError(
                f"Manifest base_ckpt mismatch: "
                f"manifest={self.base_ckpt!r} (resolved {manifest_resolved}) vs "
                f"caller={base_ckpt!r} (resolved {caller_resolved}). "
                f"Regenerate the manifest with "
                f"`precompute_preference_pairs.py --base_ckpt {base_ckpt}`."
            )
        if int(self.clip_len) != int(clip_len):
            raise ValueError(
                f"Manifest clip_len={self.clip_len} but caller wants {clip_len}. "
                f"Regenerate the manifest with --clip_len {clip_len}."
            )
        if int(self.frame_stride) != int(frame_stride):
            raise ValueError(
                f"Manifest frame_stride={self.frame_stride} but caller wants "
                f"{frame_stride}. Regenerate the manifest with "
                f"--frame_stride {frame_stride}."
            )


def write_manifest(path: Path | str, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2))


def load_manifest(path: Path | str) -> PreferenceManifest:
    raw = json.loads(Path(path).read_text())
    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in raw]
    if missing:
        raise ValueError(
            f"Manifest at {path} is missing required field(s): {missing}"
        )
    return PreferenceManifest(
        base_ckpt=str(raw["base_ckpt"]),
        success_repo_id=str(raw["success_repo_id"]),
        failure_repo_id=str(raw["failure_repo_id"]),
        clip_len=int(raw["clip_len"]),
        frame_stride=int(raw["frame_stride"]),
        smooth_window=int(raw["smooth_window"]),
        top_k=int(raw["top_k"]),
        pairs=list(raw["pairs"]),
    )
