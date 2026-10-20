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
    cfg.repair_max_nodes = -1
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


# --------------------------------------------------------------------------
# Violated-mass gate (wnet with HAS_WARM_MASS_GATE)
# --------------------------------------------------------------------------
from wnet import wnet_cpp
from wnetdeconv import Spectrum_1D
from wnetdeconv.solver import REPAIR_MASS_FRACTION

needs_gate = pytest.mark.skipif(not getattr(wnet_cpp, "HAS_WARM_MASS_GATE", False),
                                reason="wnet/pylmcf without the violated-mass gate")


def rows_spectra(n_rows=12, n_cols=40, seed=3):
    # Rows 0.2 apart with LINF cap 0.1: one small component per row.
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(n_cols) * 0.01, np.arange(n_rows) * 0.2)
    pos = np.vstack([ys.ravel(), xs.ravel()])
    def spec(s):
        w = np.random.default_rng(s).random(pos.shape[1]) + 0.05
        return Spectrum(positions=pos, intensities=w / w.sum())
    return spec(seed), [spec(seed + 1), spec(seed + 2)]


@needs_gate
def test_default_2d_config_gets_the_mass_gate():
    for cfg in (make().graph._solver, make(DeconvSolver).graph._solver,
                make(DeconvSolver, method="network_simplex").graph._solver):
        assert cfg.warm_violation_mass_fraction == REPAIR_MASS_FRACTION
        assert cfg.large_repair_mass_fraction == REPAIR_MASS_FRACTION
        assert cfg.repair_max_nodes == REPAIR_MAX_NODES


@needs_gate
def test_1d_and_caller_configs_keep_their_policy():
    emp = Spectrum_1D(np.linspace(0, 1, 30), np.random.default_rng(1).random(30) + 0.1)
    theo = [Spectrum_1D(np.linspace(0.01, 1.01, 30), np.random.default_rng(2).random(30) + 0.1)]
    one_d = DeconvSolver(emp, theo, DistanceMetric.L1, 0.1, experimental_trash_cost=0.05,
                         theoretical_trash_cost=0.05, p=2, force_dense_1d=True)
    assert isinstance(one_d.graph._solver, NetworkSimplex)
    assert one_d.graph._solver.warm_violation_mass_fraction == -1.0
    assert one_d.graph._solver.large_repair_mass_fraction == -1.0
    cfg = NetworkSimplex()
    kept = make(solver=cfg).graph._solver
    assert kept.warm_violation_mass_fraction == -1.0 and kept.large_repair_mass_fraction == -1.0


def _point_sequence():
    # Far jumps, then the small late steps a cutting-plane run makes, where
    # the gate lets repairs through.
    late = [0.55 + 1e-4 * k for k in range(8)]
    return [[0.5, 0.5], [0.9, 0.1], [0.2, 0.8], [0.6, 0.4]] + [[w, 1 - w] for w in late]


@needs_gate
@pytest.mark.parametrize("spectra", ["grid", "rows"])
@pytest.mark.parametrize("cls", [DeconvSolver, MagnetsteinSolver])
def test_mass_gated_costs_match_cold_solves(spectra, cls):
    emp, theos = grid_spectra() if spectra == "grid" else rows_spectra()
    def build(**kw):
        if cls is MagnetsteinSolver:
            return cls(emp, theos, DistanceMetric.LINF, MTD=0.05, MTD_th=0.05, p=2, **kw)
        return cls(emp, theos, DistanceMetric.LINF, 0.1, experimental_trash_cost=0.05,
                   theoretical_trash_cost=0.05, p=2, **kw)
    gated = build()
    cold_cfg = NetworkSimplex()
    cold_cfg.warm = WarmMode.NONE
    cold = build(solver=cold_cfg)
    free_cfg = NetworkSimplex()
    free_cfg.warm_violation_limit = -1
    free = build(solver=free_cfg)
    policy = repairs = 0
    for w in _point_sequence():
        for s in (gated, cold, free):
            s.set_point(w)
        assert gated.total_cost() == cold.total_cost()
        assert free.total_cost() == cold.total_cost()
        stats = dict(gated.graph.ns_stats())
        policy = stats["policy_cold"]
        repairs = gated.graph.wnet.dual_repair_count()
    assert policy + repairs > 0, "the sequence never reached the repair decision"


@needs_gate
def test_mass_gate_reaches_the_same_certificate():
    emp, theos = rows_spectra()
    results = {}
    for name, cfg in [("default", None), ("cold", "cold"), ("free", "free")]:
        kw = {}
        if cfg == "cold":
            c = NetworkSimplex(); c.warm = WarmMode.NONE; kw["solver"] = c
        elif cfg == "free":
            c = NetworkSimplex(); c.warm_violation_limit = -1; kw["solver"] = c
        s = DeconvSolver(emp, theos, DistanceMetric.LINF, 0.1, experimental_trash_cost=0.05,
                         theoretical_trash_cost=0.05, p=2, **kw)
        results[name] = s.optimize_cutting_plane(tol=1e-8)
    for name in ("cold", "free"):
        a, b = results["default"], results[name]
        assert a.bound_certified and b.bound_certified
        # Every policy solves the same quantized problems exactly; the optimum
        # each certifies must agree within both certificates.
        assert a.lb <= b.upper_bound + 1e-12 and b.lb <= a.upper_bound + 1e-12
