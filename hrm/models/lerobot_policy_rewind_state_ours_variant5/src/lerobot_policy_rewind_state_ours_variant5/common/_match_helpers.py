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


FAILURE_CLIP_DIRECTIONS = ("leaving", "approaching")


def failure_clip_bounds(
    crit_t: int,
    ep_len: int,
    pre_critical_frames: int,
    max_post_critical_frames: int,
    direction: str = "leaving",
) -> tuple[int, int]:
    """Variable-length failure clip anchored at the critical timestep.

    Two mirror anchorings around `crit_t`:

      direction="leaving" (default, original behaviour):
          [crit_t - pre_critical_frames,  crit_t + 1 + max_post_critical_frames)
          Critical point sits near the START of the clip; the clip captures
          the failure unfolding away from the critical state. Small lead-in
          (`pre_critical_frames`) keeps a 'before/during/after' structure DTW
          can align against.

      direction="approaching":
          [crit_t - max_post_critical_frames,  crit_t + 1 + pre_critical_frames)
          Mirror of "leaving": critical point sits near the END of the clip;
          the clip captures the run-up to the critical state. Same shape,
          just temporally reversed in role.

    Caller is responsible for then subsampling to a fixed length.
    """
    if direction not in FAILURE_CLIP_DIRECTIONS:
        raise ValueError(
            f"direction must be one of {FAILURE_CLIP_DIRECTIONS}, got {direction!r}"
        )
    pre = int(pre_critical_frames)
    post = int(max_post_critical_frames)
    if direction == "leaving":
        start = max(0, int(crit_t) - pre)
        end = min(int(ep_len), int(crit_t) + 1 + post)
    else:  # "approaching"
        start = max(0, int(crit_t) - post)
        end = min(int(ep_len), int(crit_t) + 1 + pre)
    return int(start), int(end)


def sample_random_variable_clip_bounds(
    ep_len: int,
    min_len: int,
    max_len: int,
    n: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Sample `n` random variable-length clip bounds [start, end) within ep.

    Each clip has length L in [min_len, min(max_len, ep_len)] and start in
    [0, ep_len - L]. Returns sorted by start. Returns [] if ep_len < min_len.
    """
    if ep_len < min_len:
        return []
    effective_max = min(int(max_len), int(ep_len))
    if effective_max < int(min_len):
        return []
    out: list[tuple[int, int]] = []
    for _ in range(int(n)):
        L = int(rng.integers(int(min_len), int(effective_max) + 1))
        start = int(rng.integers(0, int(ep_len) - L + 1))
        out.append((start, start + L))
    out.sort(key=lambda se: se[0])
    return out


def uniform_subsample_indices(start: int, end: int, target_len: int) -> list[int]:
    """Return `target_len` frame indices uniformly spanning [start, end).

    Endpoints (start, end-1) are always included. When the raw clip is
    shorter than `target_len`, indices repeat (linspace + rounding).
    """
    start = int(start)
    end = int(end)
    target_len = int(target_len)
    if end <= start:
        raise ValueError(f"empty clip: start={start} end={end}")
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")
    raw = np.linspace(start, end - 1, target_len)
    return [int(round(x)) for x in raw]


def linear_time_penalty(T: int, start: float, end: float) -> np.ndarray:
    """Length-T schedule that decays linearly from `start` to `end`.

    Use as `time_penalty_weights` in `banded_normalized_dtw` to encode
    "align tightly early, allow more drift later" -- right for clips
    anchored at a critical timestep near frame 0.

    For T==1 the schedule is just [start] (no interior to interpolate over).
    """
    T = int(T)
    if T < 1:
        raise ValueError(f"T must be positive, got {T}")
    if T == 1:
        return np.array([float(start)], dtype=np.float64)
    return np.linspace(float(start), float(end), T, dtype=np.float64)


def banded_normalized_dtw(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    band: int,
    time_penalty_weights: np.ndarray | None = None,
) -> float:
    """Sakoe-Chiba banded DTW with path-length normalization.

    `seq_a` and `seq_b` are (T, D) trajectories. Local cost is Euclidean
    distance between frames. The warping path is restricted to cells (i, j)
    with |i - j * Ta / Tb| <= band (a band of half-width `band` around the
    diagonal). The returned scalar is the total accumulated cost divided by
    the number of cells visited by the optimal path, so longer warping paths
    are not penalized for existing -- only for being a bad match per cell.

    When `time_penalty_weights` is given (a length-Ta float array), every
    cell's local cost is augmented by
        time_penalty_weights[i-1] * |i - j * Ta / Tb|.
    A schedule that is large near i=0 and decays toward 0 makes early
    warping expensive while letting the tail drift -- the right shape for
    clips anchored at a critical timestep near frame 0.
    """
    a = np.asarray(seq_a, dtype=np.float64)
    b = np.asarray(seq_b, dtype=np.float64)
    ta, tb = a.shape[0], b.shape[0]
    if ta == 0 or tb == 0:
        raise ValueError(f"empty sequences not allowed: ta={ta} tb={tb}")
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)

    if time_penalty_weights is not None:
        tpw = np.asarray(time_penalty_weights, dtype=np.float64)
        if tpw.shape != (ta,):
            raise ValueError(
                f"time_penalty_weights must have shape ({ta},), got {tpw.shape}"
            )
    else:
        tpw = None

    INF = float("inf")
    acc = np.full((ta + 1, tb + 1), INF, dtype=np.float64)
    length = np.zeros((ta + 1, tb + 1), dtype=np.int64)
    acc[0, 0] = 0.0

    slope = ta / tb
    for i in range(1, ta + 1):
        for j in range(1, tb + 1):
            diag_offset = abs((i - 1) - (j - 1) * slope)
            if diag_offset > band:
                continue
            local = cost[i - 1, j - 1]
            if tpw is not None:
                local = local + tpw[i - 1] * diag_offset
            choices = (
                (acc[i - 1, j - 1], length[i - 1, j - 1]),
                (acc[i - 1, j], length[i - 1, j]),
                (acc[i, j - 1], length[i, j - 1]),
            )
            best_cost, best_len = min(choices, key=lambda cl: cl[0])
            if best_cost == INF:
                continue
            acc[i, j] = best_cost + local
            length[i, j] = best_len + 1

    if acc[ta, tb] == INF or length[ta, tb] == 0:
        raise ValueError(
            f"DTW path infeasible with band={band} for shapes "
            f"({ta},{a.shape[1] if a.ndim > 1 else 1}) "
            f"vs ({tb},{b.shape[1] if b.ndim > 1 else 1})"
        )
    return float(acc[ta, tb]) / float(length[ta, tb])
