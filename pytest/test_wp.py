"""W_p (p != 1) support: the 1-D chain path routes to wnet's ConvexSweep
solver and must agree with the dense factory end to end.

Since wnet 1.5.0 ConvexSweep snaps coordinates onto a shared grid (spacing
1/sqrt(scale) at p=2) so its integer costs stay Monge; the dense factory
rounds each pair cost instead. The two objectives then differ by up to the
grid model's documented per-unit error times the matched mass
(wnet docs/monge_quantization.md), which these comparisons allow."""

import numpy as np
import pytest

from wnet.distances import DistanceMetric
from wnet.wnet_cpp import ConvexSweep
from wnetdeconv import DeconvSolver, MagnetsteinSolver, Spectrum_1D


def _pair(**overrides):
    emp = Spectrum_1D([1.0, 1.4, 8.0], [2.0, 2.0, 1.0])
    theos = [Spectrum_1D([1.0], [3.0]), Spectrum_1D([5.0], [1.0])]
    kw = dict(
        distance=DistanceMetric.LINF,
        max_distance=2.0,
        experimental_trash_cost=0.5,
        theoretical_trash_cost=0.8125,
        p=2.0,
    )
    kw.update(overrides)
    chain = DeconvSolver(emp, theos, solver=ConvexSweep(), **kw)
    dense = DeconvSolver(emp, theos, force_dense_1d=True, **kw)
    return chain, dense


def _unit_error(graph):
    """Per-unit cost difference allowed between the ConvexSweep grid model and
    the dense per-pair model (mirrors wnet's tests/test_convex_sweep.py)."""
    sweep = max(
        (
            getattr(sg, "convex_sweep_cost_quantization_error", lambda: 0.0)()
            for sg in graph.subgraphs()
        ),
        default=0.0,
    )
    return sweep + 0.5 / graph.scale_factor()


@pytest.mark.parametrize("independent", [True, False])
def test_p2_chain_matches_dense(independent):
    chain, dense = _pair(independent_trash=independent)
    assert chain.graph.count_chain_edges() > 0
    assert dense.graph.count_matching_edges() > 0
    unit = _unit_error(chain.graph)
    emp_mass = chain.empirical_spectrum.sum_intensities
    theo_mass = [t.sum_intensities for t in chain.theoretical_spectra]
    assert unit * emp_mass < 1e-6  # the allowance stays far below a wrong matching
    for pt in ([1.0, 1.0], [0.7, 1.3]):
        chain.set_point(pt)
        dense.set_point(pt)
        assert chain.total_cost() == pytest.approx(
            dense.total_cost(), rel=1e-9, abs=unit * emp_mass
        )
        # d/dw_i moves matched mass by at most theo_mass[i] per unit weight;
        # factor 4 for the dual-potential paths, as in wnet's own parity test.
        np.testing.assert_allclose(
            chain.gradient(),
            dense.gradient(),
            rtol=1e-9,
            atol=max(1e-9, 4 * unit * max(theo_mass)),
        )
    rc = chain.optimize_descent()
    rd = dense.optimize_descent()
    assert rc.fun == pytest.approx(rd.fun, rel=1e-6)
    np.testing.assert_allclose(rc.x, rd.x, rtol=1e-4, atol=1e-6)


def test_magnetstein_solver_accepts_p():
    emp = Spectrum_1D([1.0, 1.4, 3.0], [2.0, 2.0, 1.0])
    theos = [Spectrum_1D([1.0, 3.0], [2.0, 1.0]), Spectrum_1D([1.4], [1.0])]
    s = MagnetsteinSolver(emp, theos, DistanceMetric.L1, MTD=0.5, MTD_th=0.5, p=2.0)
    r = s.optimize()
    w = np.clip(np.asarray(r.x, dtype=float), 0.0, None)
    assert w.sum() > 0
