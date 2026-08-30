"""optimize_cutting_plane(): Kelley engine on the piecewise-linear objective."""

import numpy as np
import pytest

from wnet.distances import DistanceMetric
from wnetdeconv import DeconvSolver, MagnetsteinSolver, Spectrum_1D


def _magnetstein_pair():
    emp = Spectrum_1D([0.0, 0.5, 1.0, 1.5], [2.0, 1.0, 1.0, 0.5])
    theos = [
        Spectrum_1D([0.0, 1.0], [2.0, 1.0]),
        Spectrum_1D([0.5, 1.5], [1.0, 1.0]),
    ]
    return MagnetsteinSolver(emp, theos, DistanceMetric.L1, MTD=0.4, MTD_th=0.4)


def test_cp_not_worse_than_slsqp():
    s = _magnetstein_pair()
    f_slsqp = s.optimize().fun
    r = s.optimize_cutting_plane()
    assert r.fun <= f_slsqp + 1e-9
    assert r.nit <= 200


def test_cp_satisfies_mass_constraint():
    s = _magnetstein_pair()
    r = s.optimize_cutting_plane()
    total = float(np.dot(r.x, s._theo_totals))
    assert total == pytest.approx(s._emp_total, rel=1e-9)
    assert (r.x >= -1e-12).all()


def test_cp_unconstrained_matches_lbfgsb():
    emp = Spectrum_1D([0.0, 1.0, 2.0], [1.0, 2.0, 1.0])
    theos = [
        Spectrum_1D([0.0, 2.0], [1.0, 1.0]),
        Spectrum_1D([1.0], [1.0]),
    ]
    s = DeconvSolver(
        emp, theos, DistanceMetric.L1, max_distance=0.6, trash_cost=0.5
    )
    f_ref = s.optimize().fun
    r = s.optimize_cutting_plane()
    assert r.fun <= f_ref + 1e-9
    caps = np.array([c if np.isfinite(c) else 1.0 for c in s._w_caps])
    assert (r.x <= caps + 1e-9).all()


def test_cp_result_reporting():
    s = _magnetstein_pair()
    r = s.optimize_cutting_plane()
    assert hasattr(r, "lb") and hasattr(r, "gap") and hasattr(r, "n_cut_repairs")
    assert r.fun == pytest.approx(r.lb + r.gap, rel=1e-9, abs=1e-12)
    assert r.status in ("converged", "stalled", "max_iter") or r.status.startswith("lp_failed")
