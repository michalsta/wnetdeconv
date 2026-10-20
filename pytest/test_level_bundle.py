"""Level-bundle master (``cutting_plane_master = "level_l1"`` / ``"level_linf"``).

The level step changes only which point the oracle evaluates next; the cut
model, its lower bound and the stopping test stay Kelley's.  So under either
level master

* every cut and lower bound must stay valid for the network current at that
  iteration and the final certificate must bracket the exact optimum of an
  independent joint LP (the dense replica of ``test_cut_transfer``);
* every evaluated point must satisfy the master's box and linear constraints
  (Masserstein ``sum(w) <= 1``, ConstrainedSolver's mass equality);
* the level step must actually be taken (not silently fall back to Kelley);
* it must certify the same optimum as Kelley, up to the quantization the
  fidelity gate allows when the two runs stop at different precisions.
"""
import numpy as np
import pytest

import wnetdeconv.solver as solver_module
from test_cut_transfer import (DENSE_CASES, make_case, master_constraints,
                               run_and_check_dense, _wnet_reports_transfer_bounds)

pytestmark = pytest.mark.skipif(not _wnet_reports_transfer_bounds(),
                                reason="wnet does not report cost transfer bounds")

MASTERS = ["level_l1", "level_linf"]
CASE_IDS = lambda c: "-".join(f"{k}={getattr(v, '__name__', v)}" for k, v in c.items())


def record_points(monkeypatch, solver):
    points = []
    original = type(solver).set_point

    def spy(self, point, *args, **kwargs):
        points.append(np.array(point, float))
        return original(self, point, *args, **kwargs)

    monkeypatch.setattr(type(solver), "set_point", spy)
    return points


@pytest.mark.parametrize("master", MASTERS)
@pytest.mark.parametrize("case", DENSE_CASES, ids=CASE_IDS)
@pytest.mark.parametrize("seed", range(3))
def test_level_cuts_and_bounds_are_valid(monkeypatch, master, case, seed):
    solver, spec = make_case(seed, **case)
    solver.cutting_plane_master = master
    points = record_points(monkeypatch, solver)
    result, _ = run_and_check_dense(monkeypatch, solver, spec)
    assert result.bound_certified
    lower, caps = solver._cutting_plane_box(None)
    cons = master_constraints(solver)
    for x in points:
        assert np.all(x >= lower - 1e-9) and np.all(x <= caps + 1e-9), x
        if "a_eq" in cons:
            assert abs(cons["a_eq"] @ x - cons["b_eq"]) <= 1e-8 * max(1.0, cons["b_eq"]), x
        if "a_ub" in cons:
            assert cons["a_ub"] @ x <= cons["b_ub"] + 1e-9, x


@pytest.mark.parametrize("master", MASTERS)
def test_level_step_is_taken(monkeypatch, master):
    taken = 0
    original = solver_module.DeconvSolver._level_point

    def spy(self, *args, **kwargs):
        nonlocal taken
        x = original(self, *args, **kwargs)
        taken += x is not None
        return x

    monkeypatch.setattr(solver_module.DeconvSolver, "_level_point", spy)
    for seed in range(3):
        for case in DENSE_CASES[:4]:
            solver, _ = make_case(seed, **case)
            solver.cutting_plane_master = master
            solver.optimize_cutting_plane(tol=1e-8)
    assert taken > 10


@pytest.mark.parametrize("master", MASTERS)
@pytest.mark.parametrize("case", DENSE_CASES, ids=CASE_IDS)
def test_level_and_kelley_certify_the_same_optimum(master, case):
    results = {}
    for name in ("kelley", master):
        solver, _ = make_case(1, **case)
        solver.cutting_plane_master = name
        results[name] = solver.optimize_cutting_plane(tol=1e-8)
    a, b = results["kelley"], results[master]
    assert a.bound_certified and b.bound_certified
    # The runs may stop at different precision levels, i.e. certify
    # differently quantized networks; the fidelity gate keeps each within
    # max(tol, 1e-6 |f|) of the real costs, on top of the rounding errors.
    slack = (2 * max(1e-8, 1e-6 * max(1.0, abs(a.fun)))
             + a.cost_rounding_error + b.cost_rounding_error)
    assert a.lb <= b.upper_bound + slack and b.lb <= a.upper_bound + slack


def test_invalid_master_settings_raise():
    solver, _ = make_case(0)
    solver.cutting_plane_master = "bundle"
    with pytest.raises(ValueError, match="cutting_plane_master"):
        solver.optimize_cutting_plane()
    solver.cutting_plane_master = "level_l1"
    for bad in (0.0, 1.0, -0.2, 1.5):
        solver.level_lambda = bad
        with pytest.raises(ValueError, match="level_lambda"):
            solver.optimize_cutting_plane()
