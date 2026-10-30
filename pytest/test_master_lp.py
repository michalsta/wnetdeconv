"""The persistent HiGHS master LP must agree with the scipy.linprog fallback.

Both back ends solve the same epigraph LP; the Kelley loop reads the primal
point, the objective and the row duals (through ``_cut_model_lower_bound``).
Random cut sets with and without standing rows and the mass equality are
solved both ways and compared.
"""
import numpy as np
import pytest

from wnetdeconv import solver as solver_module
from wnetdeconv.solver import _MasterLP, _cut_model_lower_bound


def _random_problem(rng, n, k, with_eq, with_fixed):
    g = rng.normal(size=(k, n))
    b = rng.normal(size=k)
    caps = rng.uniform(0.5, 2.0, size=n)
    lower = np.zeros(n)
    a_eq = np.ones(n) if with_eq else None
    b_eq = 1.0 if with_eq else None
    fixed_a = fixed_b = None
    if with_fixed:
        rows = rng.uniform(0.0, 1.0, size=(2, n))
        fixed_a = np.column_stack([rows, np.zeros(2)])
        fixed_b = np.array([1.5, 1.2])
    return g, b, caps, lower, a_eq, b_eq, fixed_a, fixed_b


@pytest.mark.skipif(solver_module._highspy is None, reason="highspy not installed")
@pytest.mark.parametrize("with_eq", [False, True])
@pytest.mark.parametrize("with_fixed", [False, True])
def test_highs_matches_linprog(with_eq, with_fixed):
    rng = np.random.default_rng(7)
    for trial in range(5):
        n, k = 6, 25
        g, b, caps, lower, a_eq, b_eq, fixed_a, fixed_b = _random_problem(rng, n, k, with_eq, with_fixed)
        fast = _MasterLP(n, lower, caps, a_eq, b_eq, fixed_a, fixed_b)
        slow = _MasterLP(n, lower, caps, a_eq, b_eq, fixed_a, fixed_b)
        slow._h = None  # force the linprog path
        for i in range(k):
            fast.add_cut(g[i], b[i]); slow.add_cut(g[i], b[i])
            if i < 2:
                continue
            rf, rs = fast.solve(), slow.solve()
            assert rf.success and rs.success
            assert rf.fun == pytest.approx(rs.fun, abs=1e-8)
            model = lambda r: float(np.max(g[: i + 1] @ r.x[:n] + b[: i + 1]))
            assert model(rf) == pytest.approx(rf.fun, abs=1e-8)
            lb_f = _cut_model_lower_bound(rf, fast.cut_g, fast.cut_b, caps, fixed_a, fixed_b, a_eq, b_eq, lower)
            lb_s = _cut_model_lower_bound(rs, slow.cut_g, slow.cut_b, caps, fixed_a, fixed_b, a_eq, b_eq, lower)
            # Both bounds are valid lower bounds on the same LP optimum.
            assert lb_f <= rf.fun + 1e-8 and lb_s <= rs.fun + 1e-8
            assert lb_f == pytest.approx(lb_s, abs=1e-6)
        # Replacing and tightening cuts keeps the two in step.
        scaled_b = [0.5 * x - 0.1 for x in fast.cut_b]
        fast.set_cuts(fast.cut_g, scaled_b); slow.set_cuts(slow.cut_g, scaled_b)
        assert fast.solve().fun == pytest.approx(slow.solve().fun, abs=1e-8)
        lowered = [x - 0.05 for x in fast.cut_b]
        fast.set_intercepts(lowered); slow.set_intercepts(lowered)
        assert fast.solve().fun == pytest.approx(slow.solve().fun, abs=1e-8)


def test_master_lp_reports_failure_without_crash():
    n = 2
    lp = _MasterLP(n, np.zeros(n), np.ones(n), np.ones(n), 5.0)  # sum x = 5 infeasible in [0, 1]^2
    lp.add_cut(np.zeros(n), 0.0)
    r = lp.solve()
    assert not r.success and isinstance(r.message, str)


@pytest.mark.skipif(solver_module._highspy is None, reason="highspy not installed")
def test_failed_highs_run_falls_back(monkeypatch):
    """A non-optimal HiGHS status is retried cold, then solved by linprog."""
    rng = np.random.default_rng(3)
    n, k = 5, 12
    g, b, caps, lower, a_eq, b_eq, fixed_a, fixed_b = _random_problem(rng, n, k, True, False)
    lp = _MasterLP(n, lower, caps, a_eq, b_eq, fixed_a, fixed_b)
    ref = _MasterLP(n, lower, caps, a_eq, b_eq, fixed_a, fixed_b)
    ref._h = None
    for i in range(k):
        lp.add_cut(g[i], b[i]); ref.add_cut(g[i], b[i])
    unknown = solver_module._highspy.HighsModelStatus.kUnknown
    monkeypatch.setattr(lp._h, "getModelStatus", lambda: unknown)
    r = lp.solve()
    assert r.success
    assert r.fun == pytest.approx(ref.solve().fun, abs=1e-8)
