"""Spectrum.smoothed() and MagnetsteinSolver(smooth_sigma=...)."""

import numpy as np
import pytest

from wnet.distances import DistanceMetric
from wnetdeconv import MagnetsteinSolver, Spectrum_1D


def test_smoothed_preserves_mass_and_grid():
    pos = np.linspace(0.0, 10.0, 501)
    ints = np.zeros_like(pos)
    ints[250] = 5.0  # single spike mid-grid, far from the edges
    s = Spectrum_1D(pos, ints).smoothed(0.1)
    np.testing.assert_allclose(s.positions[0], pos)
    assert s.intensities.sum() == pytest.approx(5.0, rel=1e-9)
    # The spike spreads: neighbours gain mass, the centre loses it.
    assert s.intensities[250] < 5.0
    assert s.intensities[248] > 0.0


def test_smoothed_zero_sigma_is_identity():
    pos = np.linspace(0.0, 1.0, 11)
    ints = np.arange(11, dtype=float)
    s = Spectrum_1D(pos, ints).smoothed(0.0)
    np.testing.assert_allclose(s.intensities, ints)


def test_smoothed_rejects_multidim():
    from wnetdeconv import Spectrum

    s = Spectrum(np.zeros((2, 3)), np.ones(3))
    with pytest.raises(ValueError, match="1-D"):
        s.smoothed(0.1)


def test_magnetstein_smooth_sigma_runs():
    pos = np.linspace(0.0, 2.0, 201)
    theo1 = np.exp(-0.5 * ((pos - 0.5) / 0.02) ** 2)
    theo2 = np.exp(-0.5 * ((pos - 1.5) / 0.02) ** 2)
    # Mixture drawn from slightly shifted components (lineshape mismatch).
    emp = 0.7 * np.exp(-0.5 * ((pos - 0.51) / 0.03) ** 2) + 0.3 * np.exp(
        -0.5 * ((pos - 1.49) / 0.03) ** 2
    )
    solver = MagnetsteinSolver(
        Spectrum_1D(pos, emp),
        [Spectrum_1D(pos, theo1), Spectrum_1D(pos, theo2)],
        DistanceMetric.L1,
        MTD=0.3,
        MTD_th=0.3,
        smooth_sigma=0.02,
    )
    w = np.asarray(solver.optimize().x)
    w = w / w.sum()
    assert w[0] == pytest.approx(0.7, abs=0.05)
    assert w[1] == pytest.approx(0.3, abs=0.05)
