"""Unit tests for the PCA + K-means failure-onset detector
(`lerobot_policy_rewind_state_ours_variant5.common._failure_detector`).

These tests use synthetic data: success trajectories all live near the
origin (one cluster), and failure trajectories drift away after some
"failure onset" frame. We then check that the detector flags the onset
roughly where it should and that its scalar threshold behaves sensibly.
"""
from __future__ import annotations

import numpy as np
import pytest

from lerobot_policy_rewind_state_ours_variant5.common._failure_detector import (
    PCAKMeansDetector,
    critical_timestep_fail_detect_pca,
    fit_pca_kmeans_detector,
)


@pytest.fixture
def success_states_one_cluster() -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    # 20 success episodes, T=50 frames each, 8-dim state, std=0.05.
    return [rng.normal(scale=0.05, size=(50, 8)).astype(np.float32) for _ in range(20)]


def test_fit_returns_calibrated_detector(success_states_one_cluster):
    det = fit_pca_kmeans_detector(
        success_states_one_cluster, n_components=4, n_clusters=8, alpha=0.05,
    )
    assert isinstance(det, PCAKMeansDetector)
    assert det.pca.n_components_ == 4
    assert det.kmeans.n_clusters == 8
    assert det.threshold > 0.0
    assert det.alpha == 0.05
    # 95% of success scores should be below the threshold by construction
    # (with the remaining 5% above => 95th percentile).
    scores = np.concatenate(
        [det.score(s) for s in success_states_one_cluster]
    )
    frac_above = float(np.mean(scores > det.threshold))
    assert 0.02 <= frac_above <= 0.08, f"95th percentile mis-calibrated: {frac_above:.3f}"


def test_critical_t_flags_onset_for_simple_failure(success_states_one_cluster):
    det = fit_pca_kmeans_detector(
        success_states_one_cluster, n_components=4, n_clusters=8, alpha=0.05,
    )
    # Failure episode: first 30 frames inside success cluster, then drift
    # linearly to a far-OOD region for the remaining 20 frames.
    rng = np.random.default_rng(1)
    onset = 30
    early = rng.normal(scale=0.05, size=(onset, 8)).astype(np.float32)
    drift = (
        rng.normal(scale=0.05, size=(50 - onset, 8)).astype(np.float32)
        + np.linspace(0.0, 5.0, 50 - onset)[:, None].astype(np.float32)
    )
    failure = np.concatenate([early, drift], axis=0)

    crit_t = critical_timestep_fail_detect_pca(failure, det)
    # Allow a few frames of slack: the calibrated 95th-percentile threshold
    # will flag at the moment the drift exceeds the success-cluster noise.
    assert onset - 3 <= crit_t <= onset + 5, (
        f"expected critical_t near onset={onset}, got {crit_t}"
    )


def _deep_inside_cluster_observations(
    det: PCAKMeansDetector, n_frames: int = 50, jitter: float = 1e-6
) -> np.ndarray:
    """Build observations guaranteed to score below the threshold: take the
    first PCA centroid (already known low-score region), invert PCA to get
    back to observation space, then add tiny jitter so frames aren't all
    identical. Far more reliable than re-sampling the success distribution.
    """
    cluster_in_pca = det.kmeans.cluster_centers_[0:1]              # (1, n_components)
    base_obs = det.pca.inverse_transform(cluster_in_pca)           # (1, D)
    out = np.repeat(base_obs, n_frames, axis=0).astype(np.float32)
    out += np.random.default_rng(0).normal(scale=jitter, size=out.shape).astype(np.float32)
    return out


def test_critical_t_argmax_fallback_when_no_cross(success_states_one_cluster):
    det = fit_pca_kmeans_detector(
        success_states_one_cluster, n_components=4, n_clusters=8, alpha=0.05,
    )
    obvious = _deep_inside_cluster_observations(det)
    # Sanity: no frame should cross the threshold.
    assert (det.score(obvious) > det.threshold).sum() == 0
    crit_t = det.critical_t(obvious, fallback="argmax")
    assert 0 <= crit_t < obvious.shape[0]
    # No crossing => critical_t must be the argmax of the score series.
    scores = det.score(obvious)
    assert crit_t == int(np.argmax(scores))


def test_critical_t_last_fallback_when_no_cross(success_states_one_cluster):
    det = fit_pca_kmeans_detector(
        success_states_one_cluster, n_components=4, n_clusters=8, alpha=0.05,
    )
    obvious = _deep_inside_cluster_observations(det)
    assert (det.score(obvious) > det.threshold).sum() == 0
    crit_t = det.critical_t(obvious, fallback="last")
    assert crit_t == obvious.shape[0] - 1


def test_score_shape_and_empty(success_states_one_cluster):
    det = fit_pca_kmeans_detector(
        success_states_one_cluster, n_components=4, n_clusters=8, alpha=0.05,
    )
    arr = success_states_one_cluster[0]
    s = det.score(arr)
    assert s.shape == (arr.shape[0],)
    assert s.dtype == np.float32
    assert det.score(np.zeros((0, 8), dtype=np.float32)).shape == (0,)


def test_fit_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        fit_pca_kmeans_detector([], n_components=4, n_clusters=8, alpha=0.05)


def test_fit_rejects_bad_alpha(success_states_one_cluster):
    with pytest.raises(ValueError, match="alpha"):
        fit_pca_kmeans_detector(
            success_states_one_cluster, n_components=4, n_clusters=8, alpha=0.0,
        )
    with pytest.raises(ValueError, match="alpha"):
        fit_pca_kmeans_detector(
            success_states_one_cluster, n_components=4, n_clusters=8, alpha=1.0,
        )


def test_score_rejects_wrong_shape(success_states_one_cluster):
    det = fit_pca_kmeans_detector(
        success_states_one_cluster, n_components=4, n_clusters=8, alpha=0.05,
    )
    with pytest.raises(ValueError, match="T, D"):
        det.score(np.zeros((10,), dtype=np.float32))


def test_to_dict_payload(success_states_one_cluster):
    det = fit_pca_kmeans_detector(
        success_states_one_cluster, n_components=4, n_clusters=8, alpha=0.05,
    )
    d = det.to_dict()
    assert d["method"] == "pca_kmeans"
    assert d["n_pca_components"] == 4
    assert d["n_clusters"] == 8
    assert d["alpha"] == 0.05
    assert isinstance(d["threshold"], float) and d["threshold"] > 0.0
    assert d["n_success_frames"] == 20 * 50
    assert len(d["pca_explained_variance_ratio"]) == 4


def test_components_clamped_when_oversized():
    # Only 6 total frames, 3-D states. n_components must be clamped to <= 3
    # and <= 6 (frames).
    states = [np.eye(3, dtype=np.float32) * 0.01 + np.arange(3) for _ in range(2)]
    det = fit_pca_kmeans_detector(
        states, n_components=64, n_clusters=2, alpha=0.05,
    )
    assert det.pca.n_components_ <= 3
    assert det.kmeans.n_clusters == 2
