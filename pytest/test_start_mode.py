"""NetworkSimplex StartMode.Trash (crash start from the all-trash flow) through DeconvSolver.

The start mode must not change any answer: set_point costs, gradients and the
cutting-plane optimum agree with the artificial start, for every warm mode
the solver may use, and the 1-D shared-lattice policy keeps the caller's mode.
"""
import numpy as np
import pytest

from wnet.distances import DistanceMetric
from wnet.wnet_cpp import NetworkSimplex, WarmMode
from wnetdeconv import DeconvSolver, MagnetsteinSolver, Spectrum, Spectrum_1D

wnet_cpp = pytest.importorskip("wnet.wnet_cpp")
if not hasattr(wnet_cpp, "StartMode"):
    pytest.skip("wnet without StartMode", allow_module_level=True)
StartMode = wnet_cpp.StartMode


def grid_spectra(n=16, seed=0):
    xs, ys = np.meshgrid(np.arange(n) * 0.02, np.arange(n) * 0.017)
    pos = np.vstack([xs.ravel(), ys.ravel()])

    def spec(s):
        w = np.random.default_rng(s).random(pos.shape[1]) + 0.1
        return Spectrum(positions=pos, intensities=w / w.sum())
    return spec(seed), [spec(seed + 1), spec(seed + 2)]


def config(start, warm):
    cfg = NetworkSimplex()
    cfg.start = start
    cfg.warm = warm
    return cfg


@pytest.mark.parametrize("warm", [WarmMode.NONE, WarmMode.Dual, WarmMode.DualRatio])
@pytest.mark.parametrize("cls", [DeconvSolver, MagnetsteinSolver])
def test_trash_start_same_costs_and_gradients(cls, warm):
    emp, theos = grid_spectra()
    kw = (dict(MTD=0.05, MTD_th=0.05) if cls is MagnetsteinSolver
          else dict(max_distance=0.1, experimental_trash_cost=0.05, theoretical_trash_cost=0.05))
    a = cls(emp, theos, DistanceMetric.L1, solver=config(StartMode.Artificial, warm), **kw)
    b = cls(emp, theos, DistanceMetric.L1, solver=config(StartMode.Trash, warm), **kw)
    rng = np.random.default_rng(4)
    for _ in range(6):
        w = rng.uniform(0.1, 1.0, 2)
        a.set_point(w)
        b.set_point(w)
        assert a.total_cost() == b.total_cost()
        np.testing.assert_array_equal(a.gradient(), b.gradient())
    assert b.graph.crash_start_count() >= 1
    assert a.graph.crash_start_count() == 0


def test_trash_start_same_optimum():
    emp, theos = grid_spectra(seed=3)
    kw = dict(MTD=0.05, MTD_th=0.05)
    a = MagnetsteinSolver(emp, theos, DistanceMetric.L1, solver=config(StartMode.Artificial, WarmMode.DualRatio), **kw)
    b = MagnetsteinSolver(emp, theos, DistanceMetric.L1, solver=config(StartMode.Trash, WarmMode.DualRatio), **kw)
    ra, rb = a.optimize(), b.optimize()
    assert rb.fun == pytest.approx(ra.fun, rel=1e-9, abs=1e-12)
    np.testing.assert_allclose(rb.x, ra.x, atol=1e-6)


def test_lattice_policy_keeps_start_mode():
    # 1-D, common lattice, NetworkSimplex: DeconvSolver copies the config for its
    # shared-lattice warm policy and must carry the start mode across.
    x = np.arange(200) * 0.01
    rng = np.random.default_rng(1)
    emp = Spectrum_1D(x, rng.random(200) + 0.1)
    theos = [Spectrum_1D(x, rng.random(200) + 0.1) for _ in range(2)]
    s = DeconvSolver(emp, theos, DistanceMetric.L1, 0.05, experimental_trash_cost=0.05,
                     theoretical_trash_cost=0.05, solver=config(StartMode.Trash, WarmMode.DualRatio))
    assert s.graph._solver.start == StartMode.Trash
    assert s.graph._solver.warm_violation_limit == 0
