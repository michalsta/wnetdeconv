"""Certified bounds and precision refinement on the mass-balance kink."""
from types import SimpleNamespace

import numpy as np
import pytest
from wnet.distances import DistanceMetric
from wnet.wnet_cpp import NetworkSimplex, WarmMode
from wnetdeconv import DeconvSolver, ConstrainedSolver, MagnetsteinSolver, Spectrum_1D
from wnetdeconv.solver import _cut_model_lower_bound


TRUTH = np.array([0.7, 0.2, 0.1])


def make(cls, mode=WarmMode.Dual, scale=100):
    empirical = Spectrum_1D([0, 1, 2], TRUTH)
    components = [Spectrum_1D([i], [1]) for i in range(3)]
    config = NetworkSimplex()
    config.warm = mode
    options = dict(empirical_spectrum=empirical, theoretical_spectra=components,
                   distance=DistanceMetric.L1, solver=config, force_dense_1d=True,
                   scale_factor=scale, allow_intensity_loss=True)
    if cls is MagnetsteinSolver:
        for key in ("force_dense_1d", "scale_factor", "allow_intensity_loss"):
            options.pop(key)
        options.update(MTD=0.1, MTD_th=0.03)
    else:
        options.update(max_distance=0.1, experimental_trash_cost=0.1,
                       theoretical_trash_cost=0.03)
    return cls(**options)


@pytest.mark.parametrize("cls", [DeconvSolver, ConstrainedSolver, MagnetsteinSolver])
@pytest.mark.parametrize("mode", [WarmMode.NONE, WarmMode.Dual, WarmMode.LinkCut])
def test_common_dual_solves_continuous_kink(cls, mode):
    solver = make(cls, mode)
    result = solver.optimize_cutting_plane(max_iter=100, tol=1e-8, polish=False)
    assert result.bound_certified
    assert result.success, result
    assert result.n_cut_repairs == 0
    assert 0 <= result.gap <= 1e-8
    assert result.lb <= 1e-10
    assert result.fun <= result.upper_bound
    assert np.abs(result.x - TRUTH).sum() < 1e-6
    assert solver.total_cost() == pytest.approx(result.fun, abs=1e-12)


def test_coarse_oracle_refines_instead_of_claiming_false_convergence():
    solver = make(DeconvSolver, scale=13)
    original = solver.sf_intensity
    result = solver.optimize_cutting_plane(tol=1e-7, max_iter=100, polish=False)
    assert result.n_precision_refinements > 0
    assert solver.sf_intensity > original
    assert result.lb <= 1e-10
    assert result.success, result
    assert 0 <= result.gap <= 1e-7
    assert np.abs(result.x - TRUTH).sum() < 1e-5


def test_polish_improvement_is_not_a_convergence_certificate():
    solver = make(DeconvSolver)
    result = solver.optimize_cutting_plane(max_iter=1, tol=1e-12, polish=True)
    assert result.status == "max_iter"
    assert not result.success
    assert result.gap > 1e-12
    assert solver.total_cost() == pytest.approx(result.fun)


def test_master_bound_does_not_trust_an_inaccurate_primal_objective():
    # min max(x, 1-x) on [0,2] is 0.5. The cut weights define a valid
    # Lagrangian bound even with an inaccurate objective/stationarity residual.
    lp = SimpleNamespace(fun=1000, ineqlin=SimpleNamespace(marginals=np.array([-0.7, -0.3])))
    bound = _cut_model_lower_bound(lp, [[1], [-1]], [0, 1], [2])
    assert bound <= 0.5
    assert bound == pytest.approx(0.3)


@pytest.mark.parametrize("factor", [1, 0, -1, np.inf, np.nan])
def test_precision_factor_validation(factor):
    solver = make(DeconvSolver)
    with pytest.raises(ValueError):
        solver.refine_intensity_precision(factor)

