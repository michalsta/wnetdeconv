"""Cuts carried across precision refinements, checked against independent LPs.

A refinement rebuilds the network at a finer supply precision and usually a
different cost scale, so every cut collected so far bounds the *old*
quantized objective.  The cutting-plane loop carries them over with wnet's
transfer bounds instead of discarding them.  These tests replicate the
network's quantized objective exactly (dense factory: every arc and trash
cost rounded to the network's cost grid) and check that

* every cut in the master model, at every iteration, lies below the
  objective of the network current at that iteration;
* every reported lower bound lies below that network's true optimum, which an
  independent joint LP over proportions and flows computes;
* the final certificate brackets the true optimum of the final network;
* dropping the transfer correction is detected (sensitivity of the above);
* carried cuts save oracle calls and lose nothing when wnet lacks support.

Chain (SlopeDP) and ConvexSweep networks use coordinate cost models that are
not replicated here; for them the cuts are checked against the real-cost LP
widened by the network's own transfer bounds.
"""
import numpy as np
import pytest
from scipy.optimize import linprog

import wnetdeconv.solver as solver_module
from wnet.distances import DistanceMetric
from wnet.wnet_cpp import NetworkSimplex, SlopeDP, ConvexSweep
from wnetdeconv import (DeconvSolver, ConstrainedSolver, MagnetsteinSolver,
                        MassersteinSolver4, Spectrum, Spectrum_1D)


def _wnet_reports_transfer_bounds() -> bool:
    probe = DeconvSolver(Spectrum_1D([0.0, 1.0], [1.0, 1.0]), [Spectrum_1D([0.0, 1.0], [1.0, 1.0])],
                         DistanceMetric.L1, 0.5, experimental_trash_cost=0.5,
                         theoretical_trash_cost=0.5)
    probe.set_point([1.0])
    return "transfer_round_down_ratio" in probe.graph.dual_cut()


pytestmark = pytest.mark.skipif(not _wnet_reports_transfer_bounds(),
                                reason="wnet does not report cost transfer bounds")

METRICS = {"L1": DistanceMetric.L1, "LINF": DistanceMetric.LINF}


def llround(x):
    """std::llround for nonnegative costs: halves round away from zero, as the
    network does (np.rint would round them to even)."""
    return np.floor(np.asarray(x, float) + 0.5)


# --------------------------------------------------------------------------
# Instances
# --------------------------------------------------------------------------

def make_case(seed, cls=DeconvSolver, model="both", dim=2, p=1.0, metric="L1",
              n_emp=7, n_theo=4, n_comp=3, md=None, scale=13.0, backend=None,
              ce=0.4, ct=0.25):
    """A small random deconvolution problem and the solver built on it.

    ``md=None`` connects every pair (one component, which the annihilating
    replica below requires).  A small explicit ``scale`` starts on a coarse
    supply and cost grid, so the solve goes through several refinements.
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 3, (dim, n_emp))
    emp_int = rng.uniform(0.2, 1.5, n_emp)
    comps = [(rng.uniform(0, 3, (dim, n_theo)), rng.uniform(0.2, 1.5, n_theo))
             for _ in range(n_comp)]
    empirical = Spectrum(pos, emp_int) if dim > 1 else Spectrum_1D(pos[0], emp_int)
    theoretical = [Spectrum(q, i) if dim > 1 else Spectrum_1D(q[0], i) for q, i in comps]
    if md is None:
        md = 100.0
    kwargs = dict(distance=METRICS[metric], p=p)
    if backend is not None:
        kwargs["solver"] = backend
    elif dim == 1:
        kwargs["solver"] = NetworkSimplex()
        kwargs["force_dense_1d"] = True
    cap = md
    if cls is MagnetsteinSolver:
        kwargs.update(MTD=ce, MTD_th=ct)
        model = "independent"
        cap = (ce + ct) ** (1.0 / p)
    elif cls is MassersteinSolver4:
        kwargs = dict(MTD=ce, MTD_th=ct)
        model, metric, p = "independent", "LINF", 1.0
        cap = ce + ct
    else:
        kwargs.update(max_distance=md, scale_factor=scale, allow_intensity_loss=True)
        if model == "simple":
            kwargs.update(trash_cost=ce)
        elif model == "independent":
            kwargs.update(experimental_trash_cost=ce, theoretical_trash_cost=ct,
                          independent_trash=True)
        else:
            kwargs.update(experimental_trash_cost=ce, theoretical_trash_cost=ct)
    solver = cls(empirical, theoretical, **kwargs)
    if model == "simple":
        ct = ce
    return solver, dict(model=model, ce=ce, ct=ct, p=p, metric=metric, cap=cap)


# --------------------------------------------------------------------------
# Independent replica of the network's objective
# --------------------------------------------------------------------------

class Replica:
    """The continuous-supply objective of ``solver``'s dense network.

    ``scale=None`` gives the real-cost objective R; a cost scale S gives the
    quantized objective Q_S with every arc and trash cost rounded to S.
    """

    def __init__(self, solver, spec, cap):
        emp = solver.empirical_spectrum
        self.e = np.asarray(emp.intensities, float)
        self.E = float(self.e.sum())
        self.t = [np.asarray(t.intensities, float) for t in solver.theoretical_spectra]
        self.T = np.array([ti.sum() for ti in self.t])
        epos = np.asarray(emp.positions, float)
        tpos = np.column_stack([np.asarray(t.positions, float) for t in solver.theoretical_spectra])
        diff = np.abs(epos[:, :, None] - tpos[:, None, :])
        ground = diff.sum(axis=0) if spec["metric"] == "L1" else diff.max(axis=0)
        self.pairs = np.argwhere(ground <= cap)
        self.dist = ground[tuple(self.pairs.T)] ** spec["p"]
        self.owner = np.concatenate([[k] * len(ti) for k, ti in enumerate(self.t)])
        self.tflat = np.concatenate(self.t)
        self.model, self.ce, self.ct = spec["model"], spec["ce"], spec["ct"]
        self.n = len(self.t)

    def _costs(self, scale):
        if scale is None:
            return self.dist, self.ce, self.ct
        q = lambda c: llround(np.asarray(c, float) * scale) / scale  # noqa: E731
        return q(self.dist), float(q(self.ce)), float(q(self.ct))

    def _tau(self, ce, ct):
        return ce + ct if self.model == "independent" else min(ce, ct)

    def value(self, w, scale=None):
        d, ce, ct = self._costs(scale)
        supply = self.tflat * np.asarray(w, float)[self.owner]
        T = supply.sum()
        if self.model == "independent":
            trash = ce * self.E + ct * T
        elif ce <= ct:
            trash = ce * self.E + ct * max(T - self.E, 0)
        else:
            trash = ct * T + ce * max(self.E - T, 0)
        if not len(self.pairs):
            return trash
        ne, nt = len(self.e), len(self.tflat)
        A = np.zeros((ne + nt, len(self.pairs)))
        A[self.pairs[:, 0], np.arange(len(self.pairs))] = 1
        A[ne + self.pairs[:, 1], np.arange(len(self.pairs))] = 1
        res = linprog(d - self._tau(ce, ct), A_ub=A, b_ub=np.concatenate([self.e, supply]),
                      method="highs")
        assert res.success, res.message
        return trash + res.fun

    def optimum(self, lower, caps, scale=None, a_eq=None, b_eq=None, a_ub=None, b_ub=None):
        """min over the master box (and its linear constraints) of the objective."""
        d, ce, ct = self._costs(scale)
        n, P = self.n, len(self.pairs)
        ne, nt = len(self.e), len(self.tflat)
        # Variables: w (n), flows (P), s (1: positive part of T - E or E - T).
        c = np.zeros(n + P + 1)
        c[n:n + P] = d - self._tau(ce, ct)
        const = 0.0
        rows, rhs = [], []
        # Empirical capacity: sum_j f_ij <= e_i.
        for i in range(ne):
            r = np.zeros(n + P + 1)
            r[n + np.flatnonzero(self.pairs[:, 0] == i)] = 1
            rows.append(r); rhs.append(self.e[i])
        # Theoretical capacity: sum_i f_ij <= w_k t_j.
        for j in range(nt):
            r = np.zeros(n + P + 1)
            r[n + np.flatnonzero(self.pairs[:, 1] == j)] = 1
            r[self.owner[j]] = -self.tflat[j]
            rows.append(r); rhs.append(0.0)
        if self.model == "independent":
            const = ce * self.E
            c[:n] += ct * self.T
        elif ce <= ct:
            const = ce * self.E
            c[-1] = ct                                  # s >= T(w) - E
            r = np.zeros(n + P + 1); r[:n] = self.T; r[-1] = -1
            rows.append(r); rhs.append(self.E)
        else:
            c[:n] += ct * self.T
            c[-1] = ce                                  # s >= E - T(w)
            r = np.zeros(n + P + 1); r[:n] = -self.T; r[-1] = -1
            rows.append(r); rhs.append(-self.E)
        if a_ub is not None:
            r = np.zeros(n + P + 1); r[:n] = a_ub
            rows.append(r); rhs.append(b_ub)
        A_eq = b = None
        if a_eq is not None:
            A_eq = np.zeros((1, n + P + 1)); A_eq[0, :n] = a_eq
            b = [b_eq]
        bounds = list(zip(lower, caps)) + [(0, None)] * (P + 1)
        res = linprog(c, A_ub=np.array(rows), b_ub=np.array(rhs), A_eq=A_eq, b_eq=b,
                      bounds=bounds, method="highs",
                      options={"primal_feasibility_tolerance": 1e-10,
                               "dual_feasibility_tolerance": 1e-10})
        assert res.success, res.message
        return res.fun + const


def master_constraints(solver):
    n = len(solver.theoretical_spectra)
    if isinstance(solver, ConstrainedSolver):
        return dict(a_eq=np.array([t.sum_intensities for t in solver.theoretical_spectra]),
                    b_eq=float(solver.empirical_spectrum.sum_intensities))
    if isinstance(solver, solver_module._MassersteinBase):
        return dict(a_ub=np.ones(n), b_ub=1.0)
    return {}


def matching_cap(solver, spec):
    """The per-pair threshold of the solver's dense network."""
    return float(spec["cap"])


# --------------------------------------------------------------------------
# Spying on the master LP
# --------------------------------------------------------------------------

class MasterSpy:
    """Records, at every master LP, the cut model, the returned bound and the
    network it belongs to (cost scale, transfer bounds)."""

    def __init__(self, monkeypatch, solver):
        self.records = []
        original = solver_module._cut_model_lower_bound

        def spy(lp, gradients, intercepts, caps, *args, **kwargs):
            bound = original(lp, gradients, intercepts, caps, *args, **kwargs)
            cut = solver.graph.dual_cut()
            self.records.append(dict(
                scale=float(solver.graph.scale_factor()),
                gradients=[np.array(g, float) for g in gradients],
                intercepts=[float(b) for b in intercepts],
                bound=float(bound), lp_x=np.array(lp.x[:-1], float),
                transfer=solver_module._transfer_bounds(cut)))
            return bound

        monkeypatch.setattr(solver_module, "_cut_model_lower_bound", spy)


def probes_in_box(lower, caps, extra=(), seed=0, count=10):
    rng = np.random.default_rng(seed)
    pts = [lower.copy(), caps.copy(), 0.5 * (lower + caps)]
    pts += list(rng.uniform(lower, caps, (count, len(caps))))
    pts += [np.clip(np.asarray(x, float), lower, caps) for x in extra]
    return pts


def run_and_check_dense(monkeypatch, solver, spec, tol=1e-8, max_iter=200):
    lower, caps = solver._cutting_plane_box(None)
    spy = MasterSpy(monkeypatch, solver)
    result = solver.optimize_cutting_plane(tol=tol, max_iter=max_iter)
    replica = Replica(solver, spec, matching_cap(solver, spec))
    constraints = master_constraints(solver)
    optima = {}
    for k, rec in enumerate(spy.records):
        S = rec["scale"]
        if S not in optima:
            optima[S] = replica.optimum(lower, caps, scale=S, **constraints)
        f_star = optima[S]
        slack = 1e-7 * max(1.0, abs(f_star))
        assert rec["bound"] <= f_star + slack, (k, rec["bound"], f_star)
        probes = probes_in_box(lower, caps, extra=[rec["lp_x"], result.x], seed=k, count=4)
        values = [replica.value(y, scale=S) for y in probes]
        for g, b in zip(rec["gradients"], rec["intercepts"]):
            for y, v in zip(probes, values):
                assert g @ y + b <= v + 1e-7 * max(1.0, abs(v)), (k, y, g @ y + b, v)
    final = optima.get(float(solver.graph.scale_factor()))
    if final is None:
        final = replica.optimum(lower, caps, scale=float(solver.graph.scale_factor()),
                                **constraints)
    slack = 1e-7 * max(1.0, abs(final))
    assert result.lb <= final + slack
    assert result.upper_bound >= final - slack
    if result.success:
        assert result.upper_bound - result.lb <= result.gap_tolerance
    return result, spy


# --------------------------------------------------------------------------
# Dense networks: exact replica
# --------------------------------------------------------------------------

DENSE_CASES = [
    dict(cls=DeconvSolver, model="both", dim=2, p=1.0),
    dict(cls=DeconvSolver, model="both", dim=2, p=2.0, metric="LINF"),
    dict(cls=DeconvSolver, model="simple", dim=2, p=1.0),
    dict(cls=DeconvSolver, model="independent", dim=2, p=2.0, md=1.2),
    dict(cls=DeconvSolver, model="independent", dim=1, p=1.0, md=0.6),
    dict(cls=DeconvSolver, model="both", dim=1, p=2.0),
    dict(cls=DeconvSolver, model="both", dim=2, p=1.0, ce=0.2, ct=0.5),
    dict(cls=ConstrainedSolver, model="both", dim=2, p=1.0),
    dict(cls=ConstrainedSolver, model="independent", dim=2, p=2.0, md=1.5),
    dict(cls=MagnetsteinSolver, dim=2, p=2.0, metric="LINF"),
    dict(cls=MagnetsteinSolver, dim=2, p=1.0),
    dict(cls=MassersteinSolver4, dim=2, p=1.0),
]


@pytest.mark.parametrize("case", DENSE_CASES,
                         ids=lambda c: "-".join(f"{k}={getattr(v, '__name__', v)}" for k, v in c.items()))
@pytest.mark.parametrize("seed", range(4))
def test_every_cut_and_bound_is_valid_for_its_network(monkeypatch, case, seed):
    solver, spec = make_case(seed, **case)
    result, spy = run_and_check_dense(monkeypatch, solver, spec)
    assert result.bound_certified
    assert len({rec["scale"] for rec in spy.records}) >= 1


def test_cases_exercise_refinements_with_carried_cuts(monkeypatch):
    # Guard against the suite above passing vacuously: across the cases,
    # refinements must happen, change the cost scale and carry cuts.
    refinements = carried = scale_changes = 0
    for seed in range(4):
        for case in DENSE_CASES:
            solver, _ = make_case(seed, **case)
            spy = MasterSpy(monkeypatch, solver)
            result = solver.optimize_cutting_plane(tol=1e-8)
            monkeypatch.undo()
            refinements += result.n_precision_refinements
            carried += result.n_cuts_transferred
            scales = [rec["scale"] for rec in spy.records]
            scale_changes += sum(a != b for a, b in zip(scales, scales[1:]))
    assert refinements >= 20
    assert carried >= 50
    assert scale_changes >= 20


def test_naive_carry_over_is_detected(monkeypatch):
    # Keep cuts across refinements WITHOUT the transfer correction: some cut
    # then lies above the refined network's objective, which the replica
    # check must catch.  Shows the validity test above has teeth.
    monkeypatch.setattr(solver_module, "_transfer_bounds", lambda cut: (0.0, 0.0, 0.0))
    caught = 0
    for seed in range(6):
        for case in DENSE_CASES[:7]:
            solver, spec = make_case(seed, **case)
            try:
                run_and_check_dense(monkeypatch, solver, spec)
            except AssertionError:
                caught += 1
            monkeypatch.undo()
            monkeypatch.setattr(solver_module, "_transfer_bounds",
                                lambda cut: (0.0, 0.0, 0.0))
    assert caught > 0


@pytest.mark.parametrize("seed", range(3))
def test_without_wnet_support_cuts_are_discarded(monkeypatch, seed):
    monkeypatch.setattr(solver_module, "_transfer_bounds", lambda cut: None)
    solver, spec = make_case(seed)
    result, _ = run_and_check_dense(monkeypatch, solver, spec)
    assert result.n_cuts_transferred == 0
    assert result.n_precision_refinements > 0


def test_carried_cuts_save_oracle_calls_without_losing_certificates(monkeypatch):
    # Per instance the two trajectories differ, and either may stop just short
    # of the tolerance at the refinement cap; in aggregate carrying cuts must
    # certify about as often, agree on the optimum where both certify, and
    # need fewer oracle evaluations.
    calls = {"carried": 0, "discarded": 0}
    successes = {"carried": 0, "discarded": 0}
    for seed in range(5):
        for case in DENSE_CASES:
            results = {}
            for mode in calls:
                if mode == "discarded":
                    monkeypatch.setattr(solver_module, "_transfer_bounds", lambda cut: None)
                solver, _ = make_case(seed, **case)
                count = [0]
                original = solver.set_point

                def counting(x, **kw):
                    count[0] += 1
                    return original(x, **kw)

                solver.set_point = counting
                results[mode] = solver.optimize_cutting_plane(tol=1e-8)
                calls[mode] += count[0]
                successes[mode] += bool(results[mode].success)
                monkeypatch.undo()
            a, b = results["carried"], results["discarded"]
            if a.success and b.success:
                # Both certify gaps <= tol on networks of equal precision.
                if a.n_precision_refinements == b.n_precision_refinements:
                    assert abs(a.fun - b.fun) <= 2e-8 * max(1.0, abs(b.fun))
    # Measured over 120 instances (10 seeds, tol 1e-8): 94 certified either
    # way; individual instances flip in both directions near the cap.
    assert successes["carried"] >= successes["discarded"] - 2, successes
    assert calls["carried"] < calls["discarded"], calls


def test_real_cost_certificate_keeps_cuts_unchanged(monkeypatch):
    # Real-cost cuts do not depend on the quantization, so they are kept as
    # they are and must bound the real-cost optimum throughout.
    for seed in range(4):
        solver, spec = make_case(seed)
        solver.real_cost_certificate = True
        lower, caps = solver._cutting_plane_box(None)
        spy = MasterSpy(monkeypatch, solver)
        result = solver.optimize_cutting_plane(tol=1e-8)
        monkeypatch.undo()
        replica = Replica(solver, spec, matching_cap(solver, spec))
        r_star = replica.optimum(lower, caps)
        slack = 1e-7 * max(1.0, abs(r_star))
        for rec in spy.records:
            assert rec["bound"] <= r_star + slack
        assert result.lb <= r_star + slack <= result.upper_bound + 2 * slack
        if result.n_precision_refinements:
            assert result.n_cuts_transferred > 0
        # Cuts are never rescaled in real-cost mode: each refinement keeps
        # every earlier cut verbatim.
        for before, after in zip(spy.records, spy.records[1:]):
            m = len(before["gradients"])
            if len(after["gradients"]) > m:
                for g0, b0, g1, b1 in zip(before["gradients"], before["intercepts"],
                                          after["gradients"][:m], after["intercepts"][:m]):
                    assert np.array_equal(g0, g1) and b0 == b1


def test_failed_fidelity_gate_leaves_cuts_untouched(monkeypatch):
    # Force the gate to reject the first refinement: the solve stops with
    # precision_limit on the original network, and the bound it reports still
    # comes from cuts valid for that network.
    solver, spec = make_case(1)
    original = solver_module.DeconvSolver.refine_intensity_precision
    state = {"refined": 0}

    def refine(self, factor=10.0):
        original(self, factor)
        state["refined"] += 1
        real = self.graph.dual_cut

        def poisoned():
            c = dict(real())
            c["cost_rounding_error"] = 1e9
            return c

        self.graph.dual_cut = poisoned

    monkeypatch.setattr(solver_module.DeconvSolver, "refine_intensity_precision", refine)
    lower, caps = solver._cutting_plane_box(None)
    scale0 = float(solver.graph.scale_factor())
    result = solver.optimize_cutting_plane(tol=1e-12)
    assert state["refined"] == 1
    assert result.status == "precision_limit"
    assert result.n_precision_refinements == 0 and result.n_cuts_transferred == 0
    assert float(solver.graph.scale_factor()) == scale0
    replica = Replica(solver, spec, matching_cap(solver, spec))
    f_star = replica.optimum(lower, caps, scale=scale0)
    assert result.lb <= f_star + 1e-7 * max(1.0, abs(f_star))


# --------------------------------------------------------------------------
# Coordinate cost models: real-cost LP widened by the transfer bounds
# --------------------------------------------------------------------------

@pytest.mark.parametrize("backend, p", [(SlopeDP, 1.0), (ConvexSweep, 2.0)])
@pytest.mark.parametrize("seed", range(4))
def test_chain_and_sweep_cuts_bound_the_widened_real_objective(monkeypatch, backend, p, seed):
    solver, spec = make_case(seed, dim=1, p=p, model="independent", backend=backend(),
                             scale=13.0)
    lower, caps = solver._cutting_plane_box(None)
    spy = MasterSpy(monkeypatch, solver)
    result = solver.optimize_cutting_plane(tol=1e-8)
    replica = Replica(solver, spec, np.inf)
    r_star = replica.optimum(lower, caps)
    for k, rec in enumerate(spy.records):
        assert rec["transfer"] is not None
        rho, _, a = rec["transfer"]
        ceiling = (1 + rho) * r_star + a
        assert rec["bound"] <= ceiling + 1e-7 * max(1.0, abs(ceiling)), (k, rec["bound"], ceiling)
        probes = probes_in_box(lower, caps, extra=[rec["lp_x"]], seed=k, count=3)
        for y in probes:
            v = (1 + rho) * replica.value(y) + a
            for g, b in zip(rec["gradients"], rec["intercepts"]):
                assert g @ y + b <= v + 1e-7 * max(1.0, abs(v)), (k, y)
    assert result.bound_certified
