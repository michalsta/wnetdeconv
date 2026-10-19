"""Solver-built NetworkSimplex configs gate warm repair on large components.

Dual repair on one giant connected component (e.g. a 2-D grid whose cap
reaches the neighbouring column) can cost tens of cold solves, so configs
DeconvSolver builds itself set ``repair_max_nodes``; a caller's config is
used unchanged.
"""
import numpy as np
import pytest

from wnet.distances import DistanceMetric
from wnet.wnet_cpp import NetworkSimplex, WarmMode
from wnetdeconv import DeconvSolver, MagnetsteinSolver, Spectrum
from wnetdeconv.solver import REPAIR_MAX_NODES

if not hasattr(NetworkSimplex(), "repair_max_nodes"):
    pytest.skip("wnet without repair_max_nodes", allow_module_level=True)


def grid_spectra(n=24, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(n) * 0.2, np.arange(n) * 0.01)
    pos = np.vstack([xs.ravel(), ys.ravel()])
    def spec(s):
        w = np.random.default_rng(s).random(pos.shape[1]) + 0.1
        return Spectrum(positions=pos, intensities=w / w.sum())
    return spec(seed), [spec(seed + 1), spec(seed + 2)]


def make(cls=MagnetsteinSolver, **kw):
    emp, theos = grid_spectra()
    if cls is MagnetsteinSolver:
        return cls(emp, theos, DistanceMetric.LINF, MTD=0.05, MTD_th=0.05, p=2, **kw)
    return cls(emp, theos, DistanceMetric.LINF, 0.1,
               experimental_trash_cost=0.05, theoretical_trash_cost=0.05, p=2, **kw)


def test_default_config_is_gated():
    assert make().graph._solver.repair_max_nodes == REPAIR_MAX_NODES
    assert make(DeconvSolver).graph._solver.repair_max_nodes == REPAIR_MAX_NODES


def test_method_string_config_is_gated():
    solver = make(DeconvSolver, method="network_simplex")
    assert solver.graph._solver.repair_max_nodes == REPAIR_MAX_NODES


def test_caller_config_is_kept():
    cfg = NetworkSimplex()
    assert make(solver=cfg).graph._solver.repair_max_nodes == -1
    cfg.repair_max_nodes = 123
    assert make(solver=cfg).graph._solver.repair_max_nodes == 123


def test_gated_results_match_cold_solves():
    gated = make()
    assert gated.no_subgraphs() == 1
    assert gated.graph.get_subgraph(0).no_nodes() > REPAIR_MAX_NODES
    cold_cfg = NetworkSimplex()
    cold_cfg.warm = WarmMode.NONE
    cold = make(solver=cold_cfg)
    for w in ([0.5, 0.5], [0.7, 0.3], [0.2, 0.8]):
        gated.set_point(w)
        cold.set_point(w)
        assert gated.total_cost() == pytest.approx(cold.total_cost(), rel=1e-12, abs=1e-15)
