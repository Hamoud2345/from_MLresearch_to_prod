"""Tests du monitoring de drift."""

from __future__ import annotations

import numpy as np

from power_forecaster.monitoring import population_stability_index


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert population_stability_index(x, x) < 1e-6


def test_psi_flags_shifted_distribution():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    shifted = rng.normal(5, 1, 5000)  # gros decalage de moyenne -> PSI eleve
    assert population_stability_index(ref, shifted) > 0.2
