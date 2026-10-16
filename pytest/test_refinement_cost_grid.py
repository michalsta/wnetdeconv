"""Precision refinement must not trade the cost grid away and then certify.

Each x10 intensity refinement divides the cost scale by ~10 (fixed int64
budget).  On a fine profile grid, enough refinements round one-grid-step
transport costs to zero; a certificate built from those quantized costs then
"proves" a degenerate point optimal.  This case used to return x = [0, 0, 0, 1]
as converged and certified against true weights of about [0.37, 0.27, 0.32,
0.03].  The certificate now carries the cost-grid error, and refinement stops
once that error would outgrow the supply error it removes.
"""
import numpy as np
import pytest

from wnet.distances import DistanceMetric
from wnetdeconv import DeconvSolver, Spectrum_1D


def _wnet_reports_cost_rounding() -> bool:
    probe = DeconvSolver(Spectrum_1D([0.0, 1.0], [1.0, 1.0]), [Spectrum_1D([0.0, 1.0], [1.0, 1.0])],
                         DistanceMetric.L1, 0.5, experimental_trash_cost=0.5,
                         theoretical_trash_cost=0.5)
    probe.set_point([1.0])
    return "cost_rounding_error" in probe.graph.dual_cut()


pytestmark = pytest.mark.skipif(not _wnet_reports_cost_rounding(),
                                reason="wnet does not report cost_rounding_error")


def nmr_mixture(n_grid=10000, n_comp=4, peaks_per_comp=25, seed=11):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 10.0, n_grid)
    comps = []
    for _ in range(n_comp):
        y = np.zeros(n_grid)
        for _ in range(peaks_per_comp):
            c = rng.uniform(0.3, 9.7)
            w = rng.uniform(0.002, 0.006)
            mult = rng.integers(1, 4)
            J = rng.uniform(0.005, 0.012)
            for j in range(mult):
                cj = c + (j - (mult - 1) / 2) * J
                y += rng.uniform(0.3, 1.0) * w**2 / ((x - cj) ** 2 + w**2)
        comps.append(y / y.sum())
    true_w = rng.dirichlet(np.ones(n_comp))
    mix = sum(wk * ck for wk, ck in zip(true_w, comps))
    mix = np.interp(x, x + 0.002, mix) + np.abs(rng.normal(0, 2e-7, n_grid))
    return x, mix / mix.sum(), comps, true_w


def test_refinement_does_not_certify_a_degenerate_point():
    x, mix, comps, true_w = nmr_mixture()
    solver = DeconvSolver(Spectrum_1D(x, mix), [Spectrum_1D(x, c) for c in comps],
                          DistanceMetric.L1, 0.05,
                          experimental_trash_cost=0.05, theoretical_trash_cost=0.05)
    result = solver.optimize()
    assert np.abs(result.x - true_w).max() < 0.01, result.x
    # A certified success must mean a real-cost gap within tolerance; here the
    # tolerance is unreachable, so the honest outcome is an uncertified stop.
    if result.success:
        assert 0 <= result.gap <= result.gap_tolerance
    else:
        assert result.status == "precision_limit"
    assert result.cost_rounding_error >= 0


def test_real_cost_certificate_widens_the_quantized_one():
    x, mix, comps, true_w = nmr_mixture(n_grid=2000)
    solver = DeconvSolver(Spectrum_1D(x, mix), [Spectrum_1D(x, c) for c in comps],
                          DistanceMetric.L1, 0.05,
                          experimental_trash_cost=0.05, theoretical_trash_cost=0.05)
    solver.real_cost_certificate = True
    solver.set_point(true_w)
    cert = solver._certificate()
    raw = solver.graph.dual_cut()
    assert cert["rounding_error"] == cert["supply_error"] + cert["cost_error"]
    assert cert["supply_error"] == raw["rounding_error"]
    assert cert["upper_bound"] >= raw["upper_bound"]
    # The widened cut never lies above the quantized one.
    assert cert["intercept"] + cert["gradient"] @ true_w <= raw["intercept"] + raw["gradient"] @ true_w
