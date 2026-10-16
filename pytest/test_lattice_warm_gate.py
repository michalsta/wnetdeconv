"""The 1-D cold-restart policy keys on uniform common-step grids, not identity.

Warm dual repair loses to cold restarts on profile chains because equal gaps
make the chain's costs massively degenerate.  A grid offset (e.g. a
half-step resampling) keeps that degeneracy, so the policy must fire for it
too; irregular (centroided) positions keep repair.
"""
import numpy as np
import pytest

from wnet.distances import DistanceMetric
from wnet.wnet_cpp import NetworkSimplex
from wnetdeconv import DeconvSolver, Spectrum_1D


def profile(x, seed):
    rng = np.random.default_rng(seed)
    y = np.zeros_like(x)
    for c in rng.uniform(x[0] + 0.5, x[-1] - 0.5, 6):
        y += 0.002**2 / ((x - c) ** 2 + 0.002**2)
    return y / y.sum()


def make(theo_offset_steps=0.0, jitter=0.0, solver=None):
    x = np.linspace(0.0, 4.0, 801)
    step = x[1] - x[0]
    xt = x + theo_offset_steps * step
    if jitter:
        xt = np.sort(xt + np.random.default_rng(3).uniform(-jitter, jitter, xt.size) * step)
    emp = Spectrum_1D(x, profile(x, 1))
    theos = [Spectrum_1D(xt, profile(xt, s)) for s in (2, 3)]
    return DeconvSolver(emp, theos, DistanceMetric.L1, 0.05,
                        experimental_trash_cost=0.05, theoretical_trash_cost=0.05,
                        solver=NetworkSimplex() if solver is None else solver)


@pytest.mark.parametrize("offset", [0.0, 0.5, 0.25, 3.0])
def test_uniform_common_step_grids_get_cold_restarts(offset):
    solver = make(offset)
    assert solver._all_spectra_on_common_lattice()
    assert solver.graph._solver.warm_violation_limit == 0


def test_irregular_positions_keep_repair():
    solver = make(0.5, jitter=0.3)
    assert not solver._all_spectra_on_common_lattice()
    assert solver.graph._solver.warm_violation_limit == -2


def test_different_steps_keep_repair():
    x = np.linspace(0.0, 4.0, 801)
    xt = np.linspace(0.0, 4.0, 601)
    solver = DeconvSolver(Spectrum_1D(x, profile(x, 1)), [Spectrum_1D(xt, profile(xt, 2))],
                          DistanceMetric.L1, 0.05, experimental_trash_cost=0.05,
                          theoretical_trash_cost=0.05, solver=NetworkSimplex())
    assert not solver._all_spectra_on_common_lattice()
    assert solver.graph._solver.warm_violation_limit == -2


def test_explicit_policy_and_other_settings_are_kept():
    cfg = NetworkSimplex()
    cfg.warm_violation_limit = -1
    assert make(0.5, solver=cfg).graph._solver.warm_violation_limit == -1
    cfg = NetworkSimplex()
    if not hasattr(cfg, "repair_max_nodes"):
        pytest.skip("wnet without repair_max_nodes")
    cfg.repair_max_nodes = 123
    chosen = make(0.5, solver=cfg).graph._solver
    assert chosen.warm_violation_limit == 0
    assert chosen.repair_max_nodes == 123


def test_offset_grid_results_match_cold_solves():
    from wnet.wnet_cpp import WarmMode
    gated = make(0.5)
    cold_cfg = NetworkSimplex()
    cold_cfg.warm = WarmMode.NONE
    cold = make(0.5, solver=cold_cfg)
    for w in ([0.5, 0.5], [0.7, 0.4], [0.2, 0.9]):
        gated.set_point(w)
        cold.set_point(w)
        assert gated.total_cost() == pytest.approx(cold.total_cost(), rel=1e-12, abs=1e-15)
