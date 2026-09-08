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
    assert hasattr(r, "polish_improved")
    assert r.fun == pytest.approx(r.lb + r.gap, rel=1e-9, abs=1e-12)
    assert r.status in ("converged", "stalled", "max_iter") or r.status.startswith("lp_failed")


def test_polish_never_worse_than_pure_cp():
    s = _magnetstein_pair()
    r_pure = s.optimize_cutting_plane(polish=False)
    r_hybrid = s.optimize_cutting_plane(polish=True)
    assert r_hybrid.fun <= r_pure.fun + 1e-12
    # The hybrid's point still satisfies the mass constraint.
    total = float(np.dot(r_hybrid.x, s._theo_totals))
    assert total == pytest.approx(s._emp_total, rel=1e-6)


# --- Masserstein face: sum(w) <= 1 carried inside the LP --------------------
# deconvolve() dispatches to SLSQP when the total-mass constraint binds, and
# SLSQP can halt at a kink of the piecewise-linear objective and report
# success there.  optimize_cutting_plane() models the kinks instead.

from wnetdeconv import MassersteinSolver4


def _masserstein_binding():
    """A small instance whose optimum sits on the sum(w) = 1 face."""
    emp = Spectrum_1D([0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0])
    theos = [
        Spectrum_1D([0.0, 3.0], [1.0, 1.0]),
        Spectrum_1D([1.0, 2.0], [1.0, 1.0]),
        Spectrum_1D([0.5, 2.5], [1.0, 1.0]),
    ]
    return MassersteinSolver4(emp, theos, MTD=0.6, MTD_th=0.03)


def test_masserstein_face_is_actually_binding():
    # Guards the fixture: if this stops binding the tests below go vacuous.
    assert _masserstein_binding().deconvolve()["on_simplex_face"] is True


def test_masserstein_cp_not_worse_than_deconvolve():
    s = _masserstein_binding()
    f_shipped = s.deconvolve()["fun"]
    r = s.optimize_cutting_plane()
    assert r.fun <= f_shipped + 1e-9


def test_masserstein_cp_stays_on_the_feasible_set():
    s = _masserstein_binding()
    r = s.optimize_cutting_plane()
    assert (r.x >= -1e-12).all()
    assert r.x.sum() <= 1.0 + 1e-9


def test_masserstein_cp_lower_bound_brackets_the_optimum():
    s = _masserstein_binding()
    r = s.optimize_cutting_plane(polish=False)
    assert r.lb <= r.fun + 1e-9
    assert r.fun == pytest.approx(r.lb + r.gap, rel=1e-9, abs=1e-12)


def test_masserstein_polish_never_worse_and_stays_feasible():
    s = _masserstein_binding()
    r_pure = s.optimize_cutting_plane(polish=False)
    r_hybrid = s.optimize_cutting_plane(polish=True)
    assert r_hybrid.fun <= r_pure.fun + 1e-12
    assert r_hybrid.x.sum() <= 1.0 + 1e-9


def test_masserstein_optimize_override_respects_the_face():
    # DeconvSolver.optimize is bounds-only and would walk off this set.
    s = _masserstein_binding()
    r = s.optimize(x0=np.ones(3))          # x0 deliberately infeasible
    assert r.x.sum() <= 1.0 + 1e-9
    assert (r.x >= -1e-12).all()


def test_masserstein_cp_handles_a_slack_optimum():
    # When the constraint does not bind, the face row is simply inactive.
    emp = Spectrum_1D([0.0, 1.0, 2.0], [1.0, 2.0, 1.0])
    theos = [Spectrum_1D([0.0, 2.0], [1.0, 1.0]), Spectrum_1D([1.0], [1.0])]
    s = MassersteinSolver4(emp, theos, MTD=0.5, MTD_th=5.0)
    shipped = s.deconvolve()
    r = s.optimize_cutting_plane()
    assert r.fun <= shipped["fun"] + 1e-9
    assert r.x.sum() <= 1.0 + 1e-9
