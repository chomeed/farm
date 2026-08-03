"""
Lightweight OOD-on-success failure detector for picking the critical timestep
on failure episodes -- a PCA + K-means stand-in for the FAIL-Detect logpZO
flow (arXiv:2503.08558).

Pipeline:

  1. Fit PCA on all success-episode observations (state-only by default).
  2. Fit K-means on the PCA-projected success states.
  3. Per-frame score s_t = min_k || pca(x_t) - c_k ||_2  (distance to the
     nearest success centroid). High s_t => unlike the success manifold.
  4. Threshold = (1 - alpha) quantile of s_t pooled over success frames.
     This is the scalar analogue of the paper's time-varying conformal band:
     mismatched cohort lengths in our datasets make per-frame bands awkward,
     so we settle for "more OOD than (1-alpha) of all success frames".
  5. critical_t on a failure episode = first frame where s_t > threshold,
     with argmax(s_t) as fallback when no frame crosses (shouldn't happen
     for actual failure episodes but keeps the contract total).

The class is small enough to fit and ship with the precompute manifest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


@dataclass
class PCAKMeansDetector:
    """Fitted detector: PCA + K-means centroids + scalar threshold.

    Attributes:
      pca:            sklearn PCA instance fitted on success observations.
      kmeans:         sklearn KMeans instance fitted on `pca.transform(success_obs)`.
      threshold:      scalar score above which a frame is flagged as OOD.
      alpha:          conformal level used to compute the threshold; kept for
                      provenance only.
      n_success_frames: number of success frames used to fit and calibrate.
    """
    pca: PCA
    kmeans: KMeans
    threshold: float
    alpha: float
    n_success_frames: int

    # ── Inference ──

    def score(self, observations: np.ndarray) -> np.ndarray:
        """Per-frame OOD score for an (T, D) observation array.

        Returns a length-T float array. Larger = more out-of-distribution
        relative to the success manifold the detector was fit on.
        """
        obs = np.asarray(observations, dtype=np.float32)
        if obs.ndim != 2:
            raise ValueError(f"expected (T, D) observations, got {obs.shape}")
        if obs.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        projected = self.pca.transform(obs)
        # (T, n_clusters) Euclidean distance to each centroid; per-frame min.
        diffs = projected[:, None, :] - self.kmeans.cluster_centers_[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        return dists.min(axis=1).astype(np.float32)

    def critical_t(
        self, observations: np.ndarray, fallback: str = "argmax"
    ) -> int:
        """Index of the first OOD frame, or argmax(score) if none crosses."""
        scores = self.score(observations)
        if scores.size == 0:
            return 0
        above = np.where(scores > self.threshold)[0]
        if above.size > 0:
            return int(above[0])
        if fallback == "argmax":
            return int(np.argmax(scores))
        if fallback == "last":
            return int(scores.size - 1)
        raise ValueError(f"unknown fallback={fallback!r}")

    def to_dict(self) -> dict:
        """Serialisable summary for the precompute manifest payload."""
        return {
            "method": "pca_kmeans",
            "n_pca_components": int(self.pca.n_components_),
            "n_clusters": int(self.kmeans.n_clusters),
            "alpha": float(self.alpha),
            "threshold": float(self.threshold),
            "n_success_frames": int(self.n_success_frames),
            "pca_explained_variance_ratio": [
                float(v) for v in self.pca.explained_variance_ratio_
            ],
        }


def fit_pca_kmeans_detector(
    success_states_per_episode: list[np.ndarray],
    n_components: int = 4,
    n_clusters: int = 64,
    alpha: float = 0.05,
    random_state: int = 0,
) -> PCAKMeansDetector:
    """Fit PCA + K-means on success observations and calibrate a scalar
    OOD threshold from those same frames (split-conformal-lite).

    Args:
        success_states_per_episode: list of (T_i, D) state arrays, one per
            success episode. Stacked to fit both the PCA and the K-means.
        n_components: PCA dimensionality. Clamped to <= D and <= total
            frames.
        n_clusters: K-means clusters. Clamped to <= total frames.
        alpha: (1-alpha) quantile of success-frame scores is the threshold.
            E.g. alpha=0.05 => threshold is the 95th percentile.

    Returns:
        Fitted ``PCAKMeansDetector``.
    """
    if not success_states_per_episode:
        raise ValueError("success_states_per_episode is empty")
    stacked = np.concatenate(
        [np.asarray(s, dtype=np.float32) for s in success_states_per_episode],
        axis=0,
    )
    if stacked.ndim != 2:
        raise ValueError(f"each state array must be (T, D); stacked shape={stacked.shape}")
    n_frames, dim = stacked.shape
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")

    eff_components = max(1, min(int(n_components), dim, n_frames))
    eff_clusters = max(2, min(int(n_clusters), n_frames))

    pca = PCA(n_components=eff_components, random_state=random_state).fit(stacked)
    projected = pca.transform(stacked)
    kmeans = KMeans(
        n_clusters=eff_clusters,
        n_init=10,
        random_state=random_state,
    ).fit(projected)

    # Score every success frame and read off the conformal threshold.
    diffs = projected[:, None, :] - kmeans.cluster_centers_[None, :, :]
    success_scores = np.linalg.norm(diffs, axis=-1).min(axis=1).astype(np.float32)
    threshold = float(np.quantile(success_scores, 1.0 - alpha))

    return PCAKMeansDetector(
        pca=pca,
        kmeans=kmeans,
        threshold=threshold,
        alpha=alpha,
        n_success_frames=int(n_frames),
    )


def critical_timestep_fail_detect_pca(
    states: np.ndarray, detector: PCAKMeansDetector
) -> int:
    """Convenience wrapper to mirror the signature of `critical_timestep` /
    `success_critical_timestep` in `_match_helpers.py`."""
    return detector.critical_t(np.asarray(states, dtype=np.float32))
