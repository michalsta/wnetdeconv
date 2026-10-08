"""Fractional independent-trash prices must cancel for identical 2D spectra."""

import numpy as np
import pytest

from wnet.distances import DistanceMetric
from wnetdeconv import MagnetsteinSolver, Spectrum


@pytest.mark.parametrize("costs", [(0.020, 0.075), (0.030, 0.085)])
def test_identical_2d_spectrum_has_exact_zero_cost(costs):
    spectrum = Spectrum(np.array([[0.0], [0.0]]), np.array([1.0]))
    solver = MagnetsteinSolver(
        spectrum, [spectrum], DistanceMetric.LINF, MTD=costs[0], MTD_th=costs[1]
    )
    solver.set_point([1.0])
    assert solver.total_cost() == 0.0
    result = solver.optimize()
    assert result.success
    assert result.fun == 0.0
    np.testing.assert_allclose(result.x, [1.0], atol=1e-12)
