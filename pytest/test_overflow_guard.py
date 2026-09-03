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


def test_true_proportions_are_exactly_zero_cost():
    solver = make_solver()
    solver.set_point([5.0, 10.0])
    assert solver.total_cost() == pytest.approx(0.0, abs=1e-6)


def test_readme_example_optimizes_to_true_proportions():
    solver = make_solver()
    result = solver.optimize()
    assert np.allclose(result.x, [5.0, 10.0], atol=0.1), result.x
    solver.set_point(result.x)
    # The cost-0 set is not a symmetric plateau around [5, 10].  Intensity
    # quantization floors the supplies to floor(10*w1) and floor(15*w2)
    # (sf_intensity == 5 here), and cost 0 needs exactly 50 and 150, so the
    # zero-cost region is the half-open box [5, 5.1) x [10, 10 + 1/15) --
    # anchored at the true proportions and extending only upwards.  The
    # objective is a staircase, its exact marginal never vanishes (the
    # gradient at a cost-0 point is the price of one more supply unit), and
    # L-BFGS-B therefore stops on a failed line search at a point decided by
    # rounding: within atol of the truth, but not reliably inside that box.
    # Assert what a descent method can actually promise here -- within one
    # quantized supply unit of the optimum, which is the trash price of one
    # unit, trash_cost / sf_intensity.
    one_unit = 100.0 / solver.sf_intensity
    assert solver.total_cost() <= one_unit + 1e-6, result.x


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
