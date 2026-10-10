"""Public defaults, explicit descent, and bounded constrained certificates."""
import numpy as np
import pytest

from wnet.distances import DistanceMetric
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


@pytest.mark.parametrize("name", ["MassersteinSolver", "MassersteinSolver2", "MassersteinSolver4"])
@pytest.mark.parametrize("mass", [0.75, 1.0])
def test_masserstein_deconvolve_defaults_to_cutting_planes(name, mass, monkeypatch):
    import wnetdeconv
    from wnetdeconv import Spectrum_1D

    cls = getattr(wnetdeconv, name)
    solver = cls(Spectrum_1D([0, 10], [mass, 1 - mass]),
                 [Spectrum_1D([0], [1])], MTD=0.5,
                 **({"MTD_th": 0.5} if name == "MassersteinSolver4" else {}))

    def unexpected_descent(*args, **kwargs):
        pytest.fail("deconvolve must not run descent")

    monkeypatch.setattr(solver, "optimize_descent", unexpected_descent)
    # Also prohibit direct scipy calls, which the old two-pass method used.
    monkeypatch.setattr("wnetdeconv.solver.minimize", unexpected_descent)
    result = solver.deconvolve(x0=np.array([2.0]))
    assert set(result) == {"probs", "fun", "success", "on_simplex_face"}
    assert result["success"]
    total = sum(result["probs"])
    if name == "MassersteinSolver4":
        np.testing.assert_allclose(result["probs"], [mass], atol=1e-7)
        assert result["on_simplex_face"] is (mass == 1.0)
    else:
        # The one-sided approximation has a flat optimum from matched mass
        # through total mass one: unmatched empirical/theoretical mass pairs
        # cost the same as experimental trash. Check objective and feasibility.
        assert mass - 1e-7 <= total <= 1 + 1e-9
        assert result["on_simplex_face"] is bool(abs(total - 1) <= 1e-9)
    assert total <= 1 + 1e-9
    assert result["fun"] == pytest.approx(0.5 * (1 - mass), abs=1e-8)


@pytest.mark.parametrize("cls", [DeconvSolver, ConstrainedSolver, MagnetsteinSolver])
@pytest.mark.parametrize("p", [1.0, 2.0])
def test_1d_defaults_provide_certified_cutting_planes(cls, p):
    from wnet.wnet_cpp import NetworkSimplex
    from wnetdeconv import Spectrum_1D

    options = dict(empirical_spectrum=Spectrum_1D([0, 10], [0.75, 0.25]),
                   theoretical_spectra=[Spectrum_1D([0], [1]), Spectrum_1D([10], [1])],
                   distance=DistanceMetric.L1, p=p)
    if cls is MagnetsteinSolver:
        options.update(MTD=0.5, MTD_th=0.5)
    else:
        options.update(max_distance=0.5, trash_cost=0.5)
    solver = cls(**options)
    assert isinstance(solver.graph._solver, NetworkSimplex)
    result = solver.optimize()
    assert result.success and result.bound_certified
    assert 0 <= result.gap <= 1e-9
    np.testing.assert_allclose(result.x, [0.75, 0.25], atol=1e-7)
