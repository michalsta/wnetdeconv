"""W_p (p != 1) support: the 1-D chain path routes to wnet's ConvexSweep
solver and must agree with the dense factory end to end."""

import numpy as np
import pytest

from wnet.distances import DistanceMetric
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
    chain = DeconvSolver(emp, theos, **kw)
    dense = DeconvSolver(emp, theos, force_dense_1d=True, **kw)
    return chain, dense


@pytest.mark.parametrize("independent", [True, False])
def test_p2_chain_matches_dense(independent):
    chain, dense = _pair(independent_trash=independent)
    assert chain.graph.count_chain_edges() > 0
    assert dense.graph.count_matching_edges() > 0
    for pt in ([1.0, 1.0], [0.7, 1.3]):
        chain.set_point(pt)
        dense.set_point(pt)
        assert chain.total_cost() == pytest.approx(dense.total_cost(), rel=1e-9)
        np.testing.assert_allclose(
            chain.gradient(), dense.gradient(), rtol=1e-9, atol=1e-9
        )
    rc = chain.optimize()
    rd = dense.optimize()
    assert rc.fun == pytest.approx(rd.fun, rel=1e-6)
    np.testing.assert_allclose(rc.x, rd.x, rtol=1e-4, atol=1e-6)


def test_magnetstein_solver_accepts_p():
    emp = Spectrum_1D([1.0, 1.4, 3.0], [2.0, 2.0, 1.0])
    theos = [Spectrum_1D([1.0, 3.0], [2.0, 1.0]), Spectrum_1D([1.4], [1.0])]
    s = MagnetsteinSolver(emp, theos, DistanceMetric.L1, MTD=0.5, MTD_th=0.5, p=2.0)
    r = s.optimize()
    w = np.clip(np.asarray(r.x, dtype=float), 0.0, None)
    assert w.sum() > 0
