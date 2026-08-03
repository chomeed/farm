"""
Pure-function helpers for offline preference-pair precompute.

Deliberate copy of the helpers in finding_critical_points/extract_critical_embeddings.py
so this package is self-contained (no cross-folder import dependency).
"""
from __future__ import annotations

import numpy as np


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.astype(np.float32)
    pad = w // 2
    xp = np.pad(x.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float32) / w
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


def critical_timestep(rewards: np.ndarray, smooth_window: int) -> int:
    return int(np.argmax(smooth(rewards, smooth_window)))


def success_critical_timestep(
    rewards: np.ndarray, smooth_window: int, threshold: float = 0.75
) -> int:
    """Find the success-side critical moment: the first frame whose smoothed
    reward crosses `threshold`. Semantically: "the moment of successful insertion".

    Falls back to argmax(smoothed) if no frame crosses the threshold (which
    shouldn't happen for actual success episodes but is a safe default).
    """
    r = smooth(rewards, smooth_window)
    above = np.where(r >= threshold)[0]
    if above.size > 0:
        return int(above[0])
    return int(np.argmax(r))


def centered_window(t: int, span: int, ep_len: int) -> tuple[int, int]:
    """Return [start, end) of length `span` centered on t, clamped to [0, ep_len]."""
    half = span // 2
    start = t - half
    if start < 0:
        start = 0
    end = start + span
    if end > ep_len:
        end = ep_len
        start = max(0, end - span)
    return int(start), int(end)


def sample_random_clip_starts(
    ep_len: int, span: int, n: int, rng: np.random.Generator
) -> list[int]:
    """Random starts so that [start, start+span) fits within ep_len.

    Returns up to `n` starts, sorted ascending. Returns [] if `ep_len < span`.
    """
    if ep_len < span:
        return []
    high = ep_len - span + 1
    if high >= n:
        starts = rng.choice(high, size=n, replace=False)
    else:
        starts = rng.integers(0, high, size=n)
    return sorted(int(s) for s in starts)


def strided_indices(start: int, clip_len: int, frame_stride: int) -> list[int]:
    return [start + i * frame_stride for i in range(clip_len)]
