from __future__ import annotations

import numpy as np
import pytest

from lerobot_policy_rewind_state_ours_variant4.common.visualize_potential_shaping import (
    potential_curve,
)


def test_potential_curve_values():
    phi = np.array([0.0, 0.2, 0.4, 0.6, 0.8], dtype=float)
    P = potential_curve(phi, gamma=1.0, delta=2)
    # P[t] = 1.0 * phi[t+2] - phi[t]
    np.testing.assert_allclose(P, [0.4, 0.4, 0.4])


def test_potential_curve_length_is_T_minus_delta():
    phi = np.arange(20, dtype=float)
    P = potential_curve(phi, gamma=0.99, delta=8)
    assert P.shape == (12,)
    np.testing.assert_allclose(P, 0.99 * np.arange(8, 20) - np.arange(12))


def test_potential_curve_gamma_scales_future_term():
    phi = np.array([0.5, 0.5, 1.0], dtype=float)
    P = potential_curve(phi, gamma=0.5, delta=2)
    # P[0] = 0.5 * phi[2] - phi[0] = 0.5 * 1.0 - 0.5 = 0.0
    np.testing.assert_allclose(P, [0.0])


def test_potential_curve_length_one_at_boundary():
    phi = np.arange(4, dtype=float)  # T=4, delta=3 -> length 1
    P = potential_curve(phi, gamma=1.0, delta=3)
    assert P.shape == (1,)
    # P[0] = 1.0 * phi[3] - phi[0] = 3.0
    np.testing.assert_allclose(P, [3.0])


def test_potential_curve_empty_when_too_short():
    phi = np.arange(3, dtype=float)  # T=3, delta=3 -> empty
    P = potential_curve(phi, gamma=1.0, delta=3)
    assert P.shape == (0,)


def test_potential_curve_rejects_delta_below_one():
    phi = np.arange(5, dtype=float)
    with pytest.raises(ValueError, match="delta must be >= 1"):
        potential_curve(phi, gamma=1.0, delta=0)


def test_module_exposes_main_and_parse_args():
    from lerobot_policy_rewind_state_ours_variant4.common import (
        visualize_potential_shaping as m,
    )
    assert hasattr(m, "main")
    assert hasattr(m, "parse_args")
    assert hasattr(m, "potential_curve")


def test_compare_rewind_vs_variant4_potential_module():
    from lerobot_policy_rewind_state_ours_variant4.common import (
        compare_rewind_vs_variant4_potential as m,
    )
    assert hasattr(m, "main")
    assert hasattr(m, "parse_args")
