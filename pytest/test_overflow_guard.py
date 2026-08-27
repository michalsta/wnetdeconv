"""
Regression tests for the int64 cost-accumulator overflow that made
DeconvSolver.optimize() diverge under default auto-scaling.

Before wnet 1.2.1 the auto cost scale was sized for proportions of order one;
the optimizer could then walk the proportions high enough that the accumulated
integer cost wrapped negative, and L-BFGS-B chased the wrapped values to a
garbage result (observed: x ~ [196, 0.44], cost -40800 on the README example
whose optimum is [5, 10] at cost 0).  DeconvSolver now declares a mass-ratio
flow budget so the scale is sized for the reachable region, and wnet's solve()
raises OverflowError beyond it instead of wrapping.
"""

import numpy as np
import pytest

from wnetdeconv import DeconvSolver, Spectrum_1D
from wnet.distances import DistanceMetric


def make_solver():
    emp = Spectrum_1D([1.0, 100.0], [10.0, 30.0])
    t1 = Spectrum_1D([1.0], [2.0])
    t2 = Spectrum_1D([100.0], [3.0])
    return DeconvSolver(
        empirical_spectrum=emp,
        theoretical_spectra=[t1, t2],
        distance=DistanceMetric.LINF,
        max_distance=10.0,
        trash_cost=100.0,
    )


def test_readme_example_optimizes_to_true_proportions():
    solver = make_solver()
    result = solver.optimize()
    # Intensity quantization (sf_intensity == 5 here) makes cost-0 a plateau
    # about 0.1 wide in each proportion, so match to that granularity.
    assert np.allclose(result.x, [5.0, 10.0], atol=0.1), result.x
    solver.set_point(result.x)
    assert solver.total_cost() == pytest.approx(0.0, abs=1e-6)


def test_cost_stays_nonnegative_at_large_in_budget_points():
    solver = make_solver()
    # Mass-ratio caps are 16*max(1, 40/2) = 320 and 16*max(1, 40/3) ~ 213;
    # these points wrapped the accumulator negative before the budget fix.
    for point in ([200.0, 10.0], [196.2474, 0.4375]):
        solver.set_point(point)
        assert solver.total_cost() >= 0.0, point


def test_solve_raises_beyond_budget_instead_of_wrapping():
    solver = make_solver()
    with pytest.raises(OverflowError):
        solver.set_point([1e6, 1e6])
