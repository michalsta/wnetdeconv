"""Public defaults, explicit descent, and bounded constrained certificates."""
import numpy as np
import pytest

from wnet.wnet_cpp import CostScaling
from wnetdeconv import DeconvSolver, ConstrainedSolver, MagnetsteinSolver
from test_dual_cutting_plane import make, TRUTH


@pytest.mark.parametrize("cls", [DeconvSolver, ConstrainedSolver, MagnetsteinSolver])
@pytest.mark.parametrize("entry", ["optimize", "optimize_cutting_plane"])
def test_default_is_certified_and_does_not_polish(cls, entry, monkeypatch):
    solver = make(cls)

    def unexpected_descent(*args, **kwargs):
        pytest.fail("The default must not run descent")

    monkeypatch.setattr(solver, "optimize_descent", unexpected_descent)
    result = getattr(solver, entry)(tol=1e-8)
    assert result.success and result.bound_certified
    assert not result.polish_improved
    assert 0 <= result.gap <= 1e-8
    np.testing.assert_allclose(result.x, TRUTH, atol=1e-6)


@pytest.mark.parametrize("cls", [DeconvSolver, ConstrainedSolver, MagnetsteinSolver])
def test_opt_in_polish_uses_explicit_descent(cls, monkeypatch):
    solver = make(cls)
    calls = []
    original = solver.optimize_descent

    def descent(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(solver, "optimize_descent", descent)
    result = solver.optimize(tol=1e-8, polish=True)
    assert result.success and result.bound_certified
    assert len(calls) == 1 and calls[0]["maxiter"] == 100
    assert solver.total_cost() == pytest.approx(result.fun)


@pytest.mark.parametrize("cls", [DeconvSolver, ConstrainedSolver])
@pytest.mark.parametrize("polish", [False, True])
def test_custom_box_bounds_keep_valid_certificates_and_mass_constraints(cls, polish):
    solver = make(cls)
    # Exclude the original optimum. Positive lower bounds matter to both the
    # master lower bound and the feasibility of the first evaluated point.
    bounds = [(0.8, 0.9), (0.05, 0.2), (0.0, 0.1)]
    result = solver.optimize(bounds=bounds, maxiter=100, tol=1e-8, polish=polish)
    assert result.success and result.bound_certified, result
    lower, upper = np.asarray(bounds).T
    assert np.all(result.x >= lower - 1e-10)
    assert np.all(result.x <= upper + 1e-10)
    if cls is ConstrainedSolver:
        assert result.x @ solver._theo_totals == pytest.approx(solver._emp_total)
    # The optimum costs 0.003: the extra 0.1 theoretical mass is trashed.
    assert result.lb <= 0.003 + 1e-10
    assert result.upper_bound >= 0.003 - 1e-10
    assert result.upper_bound - result.lb <= 1e-8


def test_infeasible_mass_and_bounds_are_rejected_before_evaluation(monkeypatch):
    solver = make(ConstrainedSolver)
    monkeypatch.setattr(solver, "set_point", lambda point: pytest.fail("Infeasible point evaluated"))
    with pytest.raises(ValueError, match="infeasible"):
        solver.optimize(bounds=[(0, 0.1)] * 3)


def test_iteration_limit_and_progress_reporting(capsys):
    solver = make(DeconvSolver)
    result = solver.optimize(maxiter=1, tol=1e-12, print_steps=True)
    assert not result.success and result.nit == 1
    assert "step" in capsys.readouterr().out


def test_masserstein_keeps_its_positional_iteration_budget():
    from test_cutting_plane import _masserstein_binding

    result = _masserstein_binding().optimize(np.ones(3), 1)
    assert result.nit == 1
    assert result.x.sum() <= 1 + 1e-10


def test_other_backend_is_retained_and_cannot_claim_certified_success():
    solver = make(DeconvSolver)
    solver.graph._solver = CostScaling()
    solver.graph.build()
    result = solver.optimize(maxiter=5)
    assert isinstance(solver.graph._solver, CostScaling)
    assert not result.bound_certified and not result.success
