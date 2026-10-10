from collections import namedtuple
from collections.abc import Sequence
from typing import Callable, Optional, Union, List, Tuple
import numpy as np
from scipy.optimize import minimize, OptimizeResult

from wnet import Distribution, WassersteinNetwork
from wnet.scaling import WNetDeconvScaler

_Flow = namedtuple("Flow", ["empirical_peak_idx", "theoretical_peak_idx", "flow"])
from wnet.distances import DistanceMetric

_INDEPENDENT_TRASH_METHOD = "add_independent_asymmetric_trash"


def _cut_model_lower_bound(lp, gradients, intercepts, caps, fixed_a=None,
                           fixed_b=None, a_eq=None, b_eq=None, lower=None):
    """Feasible Lagrangian bound, even when the master LP has residuals.

    Normalize nonnegative cut weights to cancel the unbounded epigraph
    variable, then minimize the resulting affine function over the finite
    box. Standing constraint multipliers need no stationarity assumption.
    """
    dtype = np.longdouble
    weights = np.maximum(-np.asarray(lp.ineqlin.marginals, dtype=dtype), 0)
    count = len(intercepts)
    mass = weights[:count].sum()
    if not mass > 0:
        weights = np.zeros_like(weights)
        weights[0] = mass = 1
    weights /= mass
    coefficient = weights[:count] @ np.asarray(gradients, dtype=dtype)
    constant = weights[:count] @ np.asarray(intercepts, dtype=dtype)
    if fixed_a is not None:
        coefficient += weights[count:] @ np.asarray(fixed_a[:, :-1], dtype=dtype)
        constant -= weights[count:] @ np.asarray(fixed_b, dtype=dtype)
    if a_eq is not None:
        multiplier = -dtype(lp.eqlin.marginals[0]) / mass
        coefficient += multiplier * np.asarray(a_eq, dtype=dtype)
        constant -= multiplier * dtype(b_eq)
    box = np.asarray(caps, dtype=dtype)
    floor = np.zeros_like(box) if lower is None else np.asarray(lower, dtype=dtype)
    bound = constant + np.minimum(coefficient, 0) @ box + np.maximum(coefficient, 0) @ floor
    cushion = 64 * np.finfo(float).eps * (1 + abs(constant) + np.abs(coefficient) @ box)
    return float(bound - cushion)


def _check_rounding_loss(empirical, theoreticals, sf, max_dropped_fraction):
    """Python mirror of wnet's ``ScalerBase::check_rounding_loss``: intensities
    quantize to integer supplies as round-toward-zero(intensity * sf), and this
    computes the fraction of each spectrum's total intensity lost to that
    truncation, raising ValueError past ``max_dropped_fraction``.

    Needed here because the C++ guard runs only inside ``WNetDeconvScaler``,
    which the explicit ``scale_factor`` path bypasses — without this, a
    too-small explicit scale silently deletes mass (up to 100%) and the solver
    returns cost 0 with a nonzero gradient.
    """
    worst_frac = 0.0
    worst_name = ""
    named = [(empirical, "empirical_spectrum")] + [
        (t, f"theoretical_spectra[{i}]") for i, t in enumerate(theoreticals)
    ]
    for d, name in named:
        total = float(d.sum_intensities)
        if not total > 0.0:
            continue
        kept = float(np.sum(np.trunc(np.asarray(d.intensities) * sf))) / sf
        frac = (total - kept) / total
        if frac > worst_frac:
            worst_frac = frac
            worst_name = name
    if worst_frac > max_dropped_fraction:
        raise ValueError(
            f"Integer intensity quantization at sf_intensity={sf} would lose "
            f"{worst_frac * 100.0:.6f}% of {worst_name}'s total intensity to "
            f"rounding (limit {max_dropped_fraction * 100.0:.6f}%): the "
            f"intensity scale is too coarse for this spectrum.  Pass a larger "
            f"scale_factor or allow the loss (allow_intensity_loss=True)."
        )


def _wnet_supports_independent_trash() -> bool:
    """Empirically check whether the installed wnet exposes the independent-abyss
    trash model that :class:`MassersteinSolver4` requires.

    The C++ method ``add_independent_asymmetric_trash`` only exists on builds from
    the ``dual_trash_2`` branch; PyPI / ``main`` wnet does not bind it.  We probe
    the underlying nanobind classes (not the Python wrapper, which forwards
    dynamically) so this works before any network is constructed.
    """
    try:
        from wnet import wnet_cpp
    except Exception:
        return False
    for cls_name in ("CWassersteinNetwork", "CWassersteinNetworkFloat"):
        cls = getattr(wnet_cpp, cls_name, None)
        if cls is not None and hasattr(cls, _INDEPENDENT_TRASH_METHOD):
            return True
    return False


def _require_independent_trash_support() -> None:
    """Raise a directive error if the installed wnet lacks independent trash."""
    if _wnet_supports_independent_trash():
        return
    raise RuntimeError(
        "MassersteinSolver4 (dualdeconv4) requires wnet's independent-trash "
        "model, but the installed wnet does not expose "
        f"'{_INDEPENDENT_TRASH_METHOD}'. It is part of wnet since version "
        "1.3.0; upgrade with:\n"
        "    pip install --upgrade 'wnet>=1.3.0'\n"
        "Use MassersteinSolver2 (dualdeconv2) if you do not need the "
        "two-sided independent denoising penalty."
    )


class DeconvSolver:
    """
    Aligns an empirical spectrum to one or more theoretical spectra using a Wasserstein network approach.
    Alignment of two empirical spectra E1, E2 can be performed by setting E1 as the empirical_spectrum
    and E2 as the only element of theoretical_spectra.

    Parameters
    ----------
    empirical_spectrum : Distribution
        The empirical spectrum to be aligned.
    theoretical_spectra : Sequence[Distribution]
        A sequence of theoretical spectra to align against.
    distance_function : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Function to compute the distance between empirical and theoretical peaks.
    max_distance : int or float
        Maximum allowed distance for matching peaks.  Semantics depend on the
        factory in use:

        * **1-D chain mode** (the default for 1-D data): ``max_distance``
          acts as a *chain-splitting radius* — the merged peak sequence is
          split into independent components wherever the gap between
          *consecutive* peaks exceeds it.  Within a component, mass may
          legally travel along the chain through intermediate peaks to a
          destination *farther* than ``max_distance`` from its origin (each
          hop pays its gap cost); it is not a per-pair cap.  This is
          forwarded to wnet as ``split_distance``.
        * **Dense mode** (dims > 1, ``force_dense_1d=True``, a
          chain-incapable solver such as CostScaling/CapacityScaling, or
          ``independent_trash=True`` with a non-analytic solver, or
          ``p != 1`` without ConvexSweep): a strict *per-pair* matching
          threshold — mass is never transported between peaks farther apart than ``max_distance``.
    trash_cost : int or float, optional
        Cost for assigning unmatched peaks to trash (symmetric). Used as fallback for
        experimental_trash_cost / theoretical_trash_cost when only one is set.
    scale_factor : None, int, or float, optional
        Scaling factor for intensities and costs. If None, it is computed from ``precision``.
    precision : float, optional
        Deprecated, no effect.  Historically drove the auto scale_factor
        (``sf = 1/sqrt(precision * cost_bound)``); the p95-quantile
        ``WNetDeconvScaler`` policy replaced that, and ``ftol`` is now derived
        from the actual scale factors.  Accepted for backward compatibility;
        a non-default value triggers a DeprecationWarning.  Use
        ``scale_factor`` to override quantization explicitly.
    experimental_trash_cost : int or float, optional
        Cost for discarding unmatched empirical peaks. Enables asymmetric trash mode.
    theoretical_trash_cost : int or float, optional
        Cost for discarding unmatched theoretical peaks. Enables asymmetric trash mode.
    method : str, optional
        Min-cost flow algorithm: ``"network_simplex"`` (default), ``"cycle_canceling"``, ``"cost_scaling"``, or ``"capacity_scaling"``.
        Ignored when ``solver`` is provided.
    solver : NetworkSimplex | CostScaling | CycleCanceling | CapacityScaling, optional
        Solver configuration object.  Takes precedence over ``method``.
        Defaults to ``NetworkSimplex()`` (warm restarts, BLOCK_SEARCH pivot).
    force_dense_1d : bool, optional
        In 1D, force the O(m*n) dense factory instead of the O(m+n) chain
        factory (default False = chain in 1D).  Also switches
        ``max_distance`` from the chain-splitting-radius semantics to the
        strict per-pair cap (see ``max_distance`` above).  Forwarded to
        :class:`WassersteinNetwork`.
    allow_intensity_loss : bool, optional
        Intensities are quantized to integer supplies as
        ``int(intensity * sf_intensity)``; peaks below one integer unit floor to
        zero and vanish from the transport network.  Some loss is expected, but
        a too-small scale can silently drop almost everything.  By default, if
        quantization would discard more than 20% of any spectrum's total
        intensity, construction raises ``ValueError``.  Set ``True`` to skip the
        check and proceed anyway.  Default False.
    p : float, optional
        Wasserstein transport order (default 1.0).  Matching a unit across
        distance ``d`` costs ``d**p``, so ``total_cost()`` and gradients are
        in W_p**p units.  For ``p != 1`` on 1-D data the chain-native
        ConvexSweep solver (wnet >= 1.3.0) is the default; any explicitly
        requested other solver forces the dense factory.

    Attributes
    ----------
    sf_intensity : float
        Intensity quantization factor (integer supply units per intensity unit).
    scale_factor : float
        Cost quantization factor chosen by the network (``graph.scale_factor()``).
    empirical_spectrum : Distribution
        The empirical spectrum, as passed in (never rescaled).
    theoretical_spectra : list[Distribution]
        The theoretical spectra, as passed in (never rescaled).
    graph : WassersteinNetwork
        The underlying Wasserstein network graph.
    point : Sequence[float] or np.ndarray or None
        The current point for solving the alignment.

    Methods
    -------
    set_point(point)
        Sets the point for solving the alignment and runs the solver.
    total_cost()
        Returns the total cost of the alignment, rescaled to original units.
    print()
        Prints a string representation of the underlying graph.
    flows()
        Returns a list of flows (alignments) between empirical and theoretical peaks.
    no_subgraphs()
        Returns the number of subgraphs in the alignment network.
    print_diagnostics(subgraphs_too=False)
        Prints diagnostic information about the alignment and optionally about each subgraph.
    """

    def __init__(
        self,
        empirical_spectrum: Distribution,
        theoretical_spectra: Sequence[Distribution],
        distance: DistanceMetric,
        max_distance: Union[int, float],
        trash_cost: Optional[Union[int, float]] = None,
        scale_factor: Optional[Union[int, float]] = None,
        experimental_trash_cost: Optional[Union[int, float]] = None,
        theoretical_trash_cost: Optional[Union[int, float]] = None,
        method: str = None,
        solver=None,
        force_dense_1d: bool = False,
        precision: float = 1e-3,
        independent_trash: bool = False,
        allow_intensity_loss: bool = False,
        p: float = 1.0,
    ) -> None:

        if (
            trash_cost is None
            and experimental_trash_cost is None
            and theoretical_trash_cost is None
        ):
            raise ValueError(
                "At least one of trash_cost, experimental_trash_cost, or theoretical_trash_cost must be provided."
            )

        if not isinstance(empirical_spectrum, Distribution):
            raise TypeError("empirical_spectrum must be a Distribution")
        if not isinstance(theoretical_spectra, Sequence):
            raise TypeError("theoretical_spectra must be a Sequence")
        if not all(isinstance(t, Distribution) for t in theoretical_spectra):
            raise TypeError("all theoretical_spectra elements must be Distribution")
        if not isinstance(max_distance, (int, float)):
            raise TypeError("max_distance must be a number")
        p = float(p)
        if not (p >= 1.0) or not np.isfinite(p):
            raise ValueError(f"Wasserstein order p must be a real >= 1, got {p!r}")
        for name, val in [
            ("trash_cost", trash_cost),
            ("experimental_trash_cost", experimental_trash_cost),
            ("theoretical_trash_cost", theoretical_trash_cost),
        ]:
            if val is not None and not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be a number")
        if scale_factor is not None and not isinstance(scale_factor, (int, float)):
            raise TypeError("scale_factor must be a number")
        if precision != 1e-3:
            import warnings

            warnings.warn(
                "DeconvSolver's `precision` parameter is deprecated and has no "
                "effect (the WNetDeconvScaler p95 policy replaced it); pass "
                "scale_factor to override quantization instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        asymmetric = (
            experimental_trash_cost is not None or theoretical_trash_cost is not None
        )
        # Effective per-side trash costs (fall back to the symmetric trash_cost).
        # Computed unconditionally so the independent_trash branch below can
        # reference them even when neither asymmetric cost was given — that case
        # raises a clean ValueError there rather than an UnboundLocalError here.
        eff_exp = (
            experimental_trash_cost
            if experimental_trash_cost is not None
            else trash_cost
        )
        eff_theo = (
            theoretical_trash_cost if theoretical_trash_cost is not None else trash_cost
        )
        if asymmetric:
            active_costs = [c for c in (eff_exp, eff_theo) if c is not None]
        else:
            active_costs = [trash_cost]

        # Scaling lives in two clean places:
        #   * intensity quantization — computed below; network applies it via
        #     intensity_scale;
        #   * cost/distance quantization — the network itself via set_cost_scaling
        #     on real distances, so no position pre-scaling needed.
        if scale_factor:
            # Explicit override: intensities are quantized with exactly
            # scale_factor; the integer cost scale mirrors it as closely as the
            # int64 solver allows (nearest integer >= 1).  For
            # 0 < scale_factor < 1 no integer cost scale exists, so cost
            # quantization falls back to auto — with a warning, not silently —
            # while the intensity scale is still honored.
            if scale_factor < 0:
                raise ValueError(
                    f"scale_factor must be positive, got {scale_factor!r}"
                )
            sf_intensity = float(scale_factor)
            cost_scale = int(round(sf_intensity))
            if cost_scale < 1:
                import warnings

                warnings.warn(
                    f"scale_factor={sf_intensity!r} < 1 cannot be honored by "
                    f"the integer cost scale (which must be >= 1); cost "
                    f"quantization falls back to auto while intensities are "
                    f"still scaled by {sf_intensity!r}.",
                    stacklevel=2,
                )
                cost_scale = 0
            # The advertised rounding-loss guard applies on this path too (the
            # WNetDeconvScaler below runs it in C++; here we mirror it): a
            # too-coarse explicit scale would otherwise silently delete mass.
            if not allow_intensity_loss:
                _check_rounding_loss(
                    empirical_spectrum, theoretical_spectra, sf_intensity, 0.20
                )
        else:
            scaler = WNetDeconvScaler(
                empirical_spectrum,
                theoretical_spectra,
                distance,
                # For p != 1 matching costs reach max_distance**p; feed the
                # scaler the larger of the two so its int64 budget is safe in
                # both regimes.
                max(max_distance, max_distance**p),
                active_costs,
                max_dropped_fraction=(1.0 if allow_intensity_loss else 0.20),
            )
            sf_intensity = scaler.sf_intensity()
            cost_scale = 0  # auto — network picks cost scale against int64 budget
        self.sf_intensity = sf_intensity

        # Real positions and real costs throughout — the network quantizes costs
        # to integers internally; nothing is pre-scaled here.
        self.empirical_spectrum = empirical_spectrum
        self.theoretical_spectra = list(theoretical_spectra)

        # Certified cutting planes require NetworkSimplex dual bounds.
        # Keep analytic 1-D backends available through explicit solver configs.
        # Shared-grid NetworkSimplex warm repair can be slow; retain the cold
        # restart policy unless the caller explicitly configured repair.
        from wnet.wnet_cpp import (
            NetworkSimplex as _NSConfig,
            SlopeDP as _SlopeDP,
            ConvexSweep as _ConvexSweep,
        )
        # Resolve a method string to its solver config object up front, so the
        # policy below treats method="network_simplex" and NetworkSimplex()
        # identically (the shared-grid warm-repair override used to be bypassed
        # by the string spelling).
        if solver is None and method is not None:
            if method not in WassersteinNetwork._SOLVER_METHODS:
                raise ValueError(
                    f"Unknown method {method!r}. Choose from: "
                    f"{list(WassersteinNetwork._SOLVER_METHODS)}"
                )
            solver = WassersteinNetwork._SOLVER_METHODS[method]()
            method = None
        elif solver is not None and method is not None:
            # Documented behaviour: ``method`` is ignored when ``solver`` is
            # provided.  wnet >= 1.3.0's wrapper raises on solver+method
            # together, so drop the ignored one here.
            method = None
        if solver is None:
            solver = _NSConfig()
        if (
            empirical_spectrum.dimension == 1
            and not force_dense_1d
            and isinstance(solver, _NSConfig)
            and solver.warm_violation_limit == -2
            and self._all_spectra_share_grid()
        ):
            # Copy the config so this policy does not mutate caller state.
            cfg = _NSConfig()
            cfg.pivot = solver.pivot
            cfg.warm = solver.warm
            cfg.warm_violation_limit = 0
            solver = cfg

        # wnet >= 1.3.0 names the two distance-cap semantics separately:
        # ``max_distance`` is a guaranteed per-pair matching threshold (dense
        # semantics), ``split_distance`` is the 1-D chain component-splitting
        # radius.  DeconvSolver's 1-D path has always used the chain factory
        # with the user's cap as the splitting radius, so route it to
        # split_distance there — this preserves the historical behaviour
        # exactly (e.g. one-sided-trash 1-D setups keep the chain instead of
        # being re-gated to dense).  Dims > 1, force_dense_1d=True, and
        # chain-incapable solvers (CostScaling / CapacityScaling, which always
        # fell back to the dense factory) keep per-pair dense semantics.
        from wnet.wnet_cpp import (
            CostScaling as _CSConfig,
            CapacityScaling as _CPSConfig,
        )
        chain_semantics = (
            empirical_spectrum.dimension == 1
            and not force_dense_1d
            and not isinstance(solver, (_CSConfig, _CPSConfig))
            # Independent trash is chain-capable only under the analytic
            # backends (SlopeDP for p == 1, ConvexSweep for any p); any
            # other solver needs the dense factory and therefore per-pair
            # max_distance semantics.
            and (
                not independent_trash
                or isinstance(solver, (_SlopeDP, _ConvexSweep))
            )
            # p != 1 rides the chain only under ConvexSweep.
            and (p == 1.0 or isinstance(solver, _ConvexSweep))
        )
        cap_kwarg = (
            {"split_distance": max_distance}
            if chain_semantics
            else {"max_distance": max_distance}
        )
        self.graph = WassersteinNetwork(
            empirical_spectrum,
            theoretical_spectra,
            distance,
            force_dense_1d=force_dense_1d,
            method=method,
            solver=solver,
            intensity_scale=sf_intensity,
            round_max_distance=False,
            p=p,
            **cap_kwarg,
        )
        # Enable cost scaling so p == 1 carries real fractional distances.
        self.graph.set_cost_scaling(cost_scale)
        if independent_trash:
            # dualdeconv4: independent abysses (no annihilation discount).  An
            # unmatched empirical unit costs C_exp and an unfilled theoretical
            # unit C_theo, charged separately; the match-vs-dump threshold is
            # C_exp + C_theo (caller must pass max_distance >= MTD + MTD_th so
            # the matchable arcs exist).
            if experimental_trash_cost is None or theoretical_trash_cost is None:
                raise ValueError(
                    "independent_trash requires both experimental_trash_cost "
                    "and theoretical_trash_cost."
                )
            self.graph.add_independent_asymmetric_trash(eff_exp, eff_theo)
        elif asymmetric:
            if eff_exp is not None:
                self.graph.add_experimental_trash(eff_exp)
            if eff_theo is not None:
                self.graph.add_theoretical_trash(eff_theo)
        else:
            self.graph.add_simple_trash(trash_cost)
        # Budget the integer cost scale for the proportions the outer loop will
        # actually visit: the optimum for component i can hardly exceed the
        # mass ratio emp_total/theo_total_i (the proportion at which that
        # component alone already carries the whole empirical signal), so
        # reserve a 16x multiple of it per component.  Within this budget the
        # int64 cost accumulator cannot overflow; beyond it the network's
        # solve() raises OverflowError instead of risking a silent wrap that
        # would return a negative (or worse, plausible-looking) total cost.
        emp_total = float(empirical_spectrum.sum_intensities)
        theo_totals = [float(t.sum_intensities) for t in theoretical_spectra]
        # Per-component proportion caps implied by the budget; optimizers use
        # these as L-BFGS-B/SLSQP bounds so the search cannot leave the
        # representable region (solve() raises OverflowError past it).
        self._w_caps = np.array([
            16.0 * max(1.0, emp_total / tt) if tt > 0.0 else np.inf
            for tt in theo_totals
        ])
        flow_budget = emp_total + sum(
            cap * tt for cap, tt in zip(self._w_caps, theo_totals) if tt > 0.0
        )
        self.graph.set_flow_budget(flow_budget)
        self.graph.build()
        # Reported factors (the cost scale is chosen at build()).
        self.scale_factor = self.graph.scale_factor()
        self._ftol = 1.0 / (self.graph.scale_factor() * sf_intensity)
        self.point = None

    def _budget_bounds(self) -> list:
        """Per-component (0, cap) bounds matching the declared flow budget."""
        return [(0.0, c if np.isfinite(c) else None) for c in self._w_caps]

    def _cutting_plane_box(self, bounds):
        """Intersect requested bounds with the finite, nonnegative flow budget."""
        caps = np.array([c if np.isfinite(c) else 1.0 for c in self._w_caps])
        lower = np.zeros_like(caps)
        if bounds is not None:
            if len(bounds) != len(caps):
                raise ValueError("Bounds must contain one pair per component")
            for i, (lo, hi) in enumerate(bounds):
                lo = 0.0 if lo is None else float(lo)
                hi = caps[i] if hi is None else float(hi)
                if np.isnan(lo) or np.isnan(hi):
                    raise ValueError("Bounds must not contain NaN")
                lower[i], caps[i] = max(0.0, lo), min(caps[i], hi)
        if not np.all(np.isfinite(lower)) or np.any(lower > caps):
            raise ValueError("Bounds have no feasible intersection with the flow budget")
        return lower, caps

    def _warn_if_caps_binding(self, x) -> None:
        """Warn when an optimum sits against the budget bound — the true
        optimum may lie beyond the representable region."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        binding = [
            i for i, (xi, cap) in enumerate(zip(x, self._w_caps))
            if np.isfinite(cap) and xi > 0.99 * cap
        ]
        if binding:
            import warnings

            warnings.warn(
                f"Proportions at indices {binding} ended within 1% of the "
                f"overflow-budget bound and may be clipped; pass an explicit "
                f"scale_factor or rescale theoretical intensities to widen "
                f"the representable region.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _all_spectra_share_grid(self) -> bool:
        """True if every theoretical spectrum sits on the empirical spectrum's
        exact position grid (profile-mode data)."""
        emp_pos = np.asarray(self.empirical_spectrum.positions)
        return all(
            np.array_equal(emp_pos, np.asarray(t.positions))
            for t in self.theoretical_spectra
        )

    def set_point(self, point: Union[Sequence[float], np.ndarray]) -> None:
        """
        Set proportions of theoretical spectra and solve the graph at the given point.

        Parameters
        ----------
        point : Sequence[float] or np.ndarray
            Proportions for each theoretical spectrum.

        Returns
        -------
        None
        """
        self.point = point
        self.graph.solve(point)

    def total_cost(self) -> float:
        """
        Calculates the total cost of the graph. Can only be called after set_point().

        Returns:
            float: The normalized total cost.
        """
        # The network returns the real (unscaled) cost: it divides the scaled
        # integer cost by scale_factor() * intensity_scale_factor().
        return self.graph.total_cost()

    def print(self) -> None:
        """
        Prints a string representation of the graph associated with this aligner instance.

        Returns:
            None
        """
        print(str(self.graph))

    def flows(self) -> list[_Flow]:
        """
        Computes and returns a list of flow information for each theoretical spectrum.

        Each flow is represented as a namedtuple containing the empirical peak index,
        theoretical peak index, and the scaled flow value (divided by self.scale_factor).

        Returns:
            list[namedtuple]: A list of Flow namedtuples, one for each theoretical
            spectrum, each containing:
                - empirical_peak_idx (int): Index of the empirical peak.
                - theoretical_peak_idx (int): Index of the theoretical peak.
                - flow (float): Scaled flow value between the peaks.
        """
        result = []
        for i in range(len(self.theoretical_spectra)):
            empirical_peak_idx, theoretical_peak_idx, flow = (
                self.graph.flows_for_target(i)
            )
            result.append(
                _Flow(
                    empirical_peak_idx, theoretical_peak_idx, flow / self.sf_intensity
                )
            )
        return result

    def gradient(self) -> np.ndarray:
        """
        Returns the gradient of total_cost with respect to the point
        (spectrum proportions). Can only be called after set_point().

        Returns
        -------
        np.ndarray
            Array of partial derivatives, one per theoretical spectrum.
        """
        # The network wrapper already returns derivatives in real units
        # (un-scaled by its cost scale_factor()).
        return self.graph.spectrum_proportion_derivatives().astype(float)

    def gradient_fast_approx(self) -> np.ndarray:
        """Fast, APPROXIMATE gradient (dual-potential difference instead of the
        residual shortest-path marginal).

        Much cheaper (skips the per-subgraph Dijkstra) but returns a
        different, basis-dependent gradient: a lower bound on the true
        marginal, exact only on the optimal flow support.  Opt-in; do not use
        as a drop-in replacement for gradient() without validating convergence.
        """
        return self.graph.spectrum_proportion_derivatives_fast_approx().astype(float)

    def optimize(
        self,
        x0: Optional[np.ndarray] = None,
        bounds: Optional[np.array] = None,
        maxiter: Optional[int] = None,
        print_steps: bool = False,
        *,
        tol: Optional[float] = None,
        polish: bool = False,
    ) -> OptimizeResult:
        """Minimize transport cost with cutting planes, without polishing.

        ``maxiter`` limits oracle evaluations (default 200); ``tol`` is the
        absolute certified objective-gap tolerance (default 1e-9). Optional
        bounds intersect the nonnegative integer-flow budget box. The class's
        mass constraints remain active. ``print_steps`` reports model progress.
        Set ``polish=True`` to finish with descent, or call
        :meth:`optimize_descent` explicitly for the previous optimizer.
        Certificates require a NetworkSimplex backend; other backends retain
        their uncertified cutting-plane search without replacing the backend.
        """
        return self.optimize_cutting_plane(
            x0=x0, max_iter=200 if maxiter is None else maxiter,
            tol=tol, polish=polish, bounds=bounds, print_steps=print_steps,
        )

    def optimize_descent(
        self,
        x0: Optional[np.ndarray] = None,
        bounds: Optional[np.array] = None,
        maxiter: Optional[int] = None,
        print_steps: bool = False,
    ) -> OptimizeResult:
        """
        Minimize total transport cost over non-negative spectrum proportions.

        Parameters
        ----------
        x0 : np.ndarray, optional
            Initial proportions. Defaults to a vector of ones.
        maxiter : int, optional
            Iteration cap for the inner optimizer (default: the method's
            own default).
        print_steps : bool
            Prints gradient descent steps for diagnostics purposes.

        Returns
        -------
        scipy.optimize.OptimizeResult
            Standard scipy result; .x holds the optimal proportions.
        """
        n = len(self.theoretical_spectra)
        if x0 is None:
            x0 = np.ones(n)
        if bounds is None:
            bounds = self._budget_bounds()

        # def cost_and_grad(w):
        #     self.set_point(w)
        #     return self.total_cost(), self.gradient()

        step = [0]
        def cost_and_grad(w):
            self.set_point(w)
            c, g = self.total_cost(), self.gradient()
            if print_steps: print(f"step {step[0]:3d}  point=[{', '.join(f"{x:8.4f}" for x in w)}]  cost={c:8.4f}  grad=[{', '.join(f"{x:8.4f}" for x in g)}]")
            step[0] += 1
            return c, g

        options = {"ftol": self._ftol}
        if maxiter is not None:
            options["maxiter"] = maxiter

        result = minimize(
            cost_and_grad,
            x0=x0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options=options,
        )
        self._warn_if_caps_binding(result.x)
        return result

    def optimize_cutting_plane(
        self,
        x0: Optional[np.ndarray] = None,
        max_iter: int = 200,
        tol: Optional[float] = None,
        polish: bool = False,
        *,
        bounds: Optional[np.array] = None,
        print_steps: bool = False,
    ) -> OptimizeResult:
        """Minimize transport cost with Kelley cutting planes.

        NetworkSimplex variants construct common feasible dual cuts for the
        continuous-supply objective with fixed quantized costs. Integer solves
        provide upper bounds including explicit supply-rounding uncertainty.
        Precision is refined adaptively; rebuilding clears all cuts and bounds
        because the cost scale may change. No generic LP transport backend is
        used. The small master LP optimizes mixture weights only.

        Other backends retain the legacy marginal-cut search and return
        ``bound_certified=False`` and ``success=False``; their model gap is
        diagnostic, not an optimality certificate.

        ``tol`` is a positive absolute objective-gap tolerance (default 1e-9).
        ``max_iter`` bounds oracle evaluations (default 200). ``polish`` runs
        descent from the selected point and keeps an improvement without
        treating improvement alone as convergence.

        The result includes ``fun`` (rounded network cost), ``upper_bound``
        (continuous objective upper bound), ``lb``, ``gap=upper_bound-lb``,
        ``rounding_error``, ``bound_certified``, ``n_precision_refinements``,
        ``n_cut_repairs`` and ``polish_improved``. A certified success requires
        a nonnegative gap no larger than ``tol``. Bounds use fixed quantized
        edge costs, rather than claiming exact original real cost coefficients.
        """
        n = len(self.theoretical_spectra)
        if x0 is None:
            x0 = np.ones(n)
        result = self._kelley(
            x0=np.asarray(x0, dtype=float),
            a_eq=None,
            b_eq=None,
            max_iter=max_iter,
            tol=tol,
            bounds=bounds, print_steps=print_steps,
        )
        if polish:
            result = self._polish(result, bounds=bounds)
        self._warn_if_caps_binding(result.x)
        return result

    def refine_intensity_precision(self, factor: float = 10.0) -> None:
        """Rebuild the same network at finer supply precision with safe cost scaling.

        Topology/trash semantics and solver variant stay unchanged. A rebuild
        invalidates the old basis and cuts because quantized costs may change.
        """
        if not np.isfinite(factor) or factor <= 1:
            raise ValueError("Precision factor must be finite and greater than one")
        new_scale = self.sf_intensity * factor
        if not np.isfinite(new_scale):
            raise ValueError("Refined intensity scale must be finite")
        self.graph.refine_intensity_precision(factor)
        self.sf_intensity = new_scale
        self.scale_factor = self.graph.scale_factor()
        self._ftol = 1.0 / (self.scale_factor * new_scale)
        self.point = None

    def _polish(self, cp_result: OptimizeResult, bounds=None) -> OptimizeResult:
        """Warm-start the class's descent optimizer from the cutting-plane
        answer; keep the cheaper of the two evaluated points."""
        import warnings

        if bounds is not None:
            lower, caps = self._cutting_plane_box(bounds)
            bounds = list(zip(lower, caps))

        with warnings.catch_warnings():
            # The caps-binding warning is re-issued by the caller on the
            # final point; suppress the inner optimizer's copy.  The polish
            # only has to finish the current linear piece: cap its
            # iterations, or a seam-corner warm start can send SLSQP on a
            # 2000-iteration crawl (observed: 47 min on a 54k-point
            # spectrum).  The keep-the-cheaper comparison below preserves
            # the monotone guarantee regardless of where the cap lands.
            warnings.simplefilter("ignore", RuntimeWarning)
            descent = self.optimize_descent(
                x0=np.asarray(cp_result.x, dtype=float), maxiter=100, bounds=bounds
            )
        self.set_point(descent.x)
        f_descent = float(self.total_cost())
        self.set_point(cp_result.x)
        f_cp = float(self.total_cost())
        improved = f_descent < f_cp
        if improved:
            cp_result.x = np.asarray(descent.x, dtype=float)
            cp_result.fun = f_descent
            if cp_result.get("bound_certified", False):
                self.set_point(descent.x)
                certificate = self.graph.dual_cut()
                cp_result.upper_bound = certificate["upper_bound"]
                cp_result.rounding_error = certificate["rounding_error"]
                cp_result.gap = cp_result.upper_bound - cp_result.lb
                cp_result.success = 0 <= cp_result.gap <= cp_result.gap_tolerance
                if cp_result.success:
                    cp_result.status = "converged"
            else:
                cp_result.gap = f_descent - cp_result.lb
        self.set_point(cp_result.x)
        cp_result.polish_improved = improved
        cp_result.message += (
            f"; polish {'improved to ' + format(f_descent, '.6g') if improved else 'kept the cutting-plane point'}"
        )
        return cp_result

    def _kelley(self, x0, a_eq, b_eq, max_iter, tol,
                a_ub=None, b_ub=None, bounds=None, print_steps=False) -> OptimizeResult:
        """Shared cutting-plane engine (see :meth:`optimize_cutting_plane`).

        ``a_eq`` / ``b_eq`` optionally impose a linear equality (used by
        :class:`ConstrainedSolver` for the total-mass constraint).
        ``a_ub`` / ``b_ub`` optionally impose linear inequalities
        ``a_ub @ w <= b_ub`` (used by :class:`_MassersteinBase` for the
        ``sum(w) <= 1`` face).  Both are carried inside the per-iteration LP,
        so the model minimizer is always feasible and the returned lower
        bound is a bound on the *constrained* optimum."""
        from scipy.optimize import linprog

        n = len(self.theoretical_spectra)
        # Finite LP box from the overflow-budget caps (a component with no
        # intensity is inert; bound it by 1 to keep the LP bounded).
        lower, caps = self._cutting_plane_box(bounds)
        bounds = list(zip(lower, caps)) + [(None, None)]
        c_lp = np.zeros(n + 1)
        c_lp[-1] = 1.0
        lp_a_eq = None
        lp_b_eq = None
        if a_eq is not None:
            lp_a_eq = np.concatenate([np.asarray(a_eq, float), [0.0]])[None, :]
            lp_b_eq = np.asarray([b_eq], dtype=float)
        # Standing inequalities, padded with a zero in the epigraph column.
        fixed_a_ub = None
        fixed_b_ub = None
        if a_ub is not None:
            rows = np.atleast_2d(np.asarray(a_ub, dtype=float))
            fixed_a_ub = np.column_stack([rows, np.zeros(len(rows))])
            fixed_b_ub = np.atleast_1d(np.asarray(b_ub, dtype=float))

        from wnet.wnet_cpp import NetworkSimplex
        certified = isinstance(self.graph._solver, NetworkSimplex)
        evals_x: list[np.ndarray] = []
        evals_f: list[float] = []
        cut_g: list[np.ndarray] = []
        cut_b: list[float] = []
        n_repairs = 0
        n_refinements = 0
        best_f = np.inf
        best_upper = np.inf
        best_error = np.inf
        best_x = np.asarray(x0, dtype=float)
        lb = -np.inf
        x = np.clip(np.asarray(x0, dtype=float), lower, caps)
        status = "max_iter"
        nit = 0
        gap_tol = tol if tol is not None else 1e-9
        if not np.isfinite(gap_tol) or gap_tol <= 0:
            raise ValueError("Gap tolerance must be finite and positive")
        if max_iter < 1:
            raise ValueError("max_iter must be positive")

        # A clipped user start can violate the mass constraint. Upper bounds
        # must come from points feasible for the same master problem.
        violates_eq = lp_a_eq is not None and not np.allclose(
            lp_a_eq[:, :n] @ x, lp_b_eq, rtol=0, atol=1e-10)
        violates_ub = fixed_a_ub is not None and np.any(
            fixed_a_ub[:, :n] @ x > fixed_b_ub + 1e-10)
        if violates_eq or violates_ub:
            feasible = linprog(np.zeros(n + 1), A_ub=fixed_a_ub, b_ub=fixed_b_ub,
                               A_eq=lp_a_eq, b_eq=lp_b_eq, bounds=bounds, method="highs",
                               options={"primal_feasibility_tolerance": 1e-10,
                                        "dual_feasibility_tolerance": 1e-10})
            if not feasible.success:
                raise ValueError(f"Bounds and mass constraints are infeasible: {feasible.message}")
            x = np.clip(feasible.x[:n], lower, caps)

        for nit in range(1, max_iter + 1):
            self.set_point(x)
            if certified:
                certificate = self.graph.dual_cut()
                f = float(certificate["rounded_cost"])
                g = np.asarray(certificate["gradient"], dtype=float)
                intercept = float(certificate["intercept"])
                upper = float(certificate["upper_bound"])
                error = float(certificate["rounding_error"])
            else:
                f = float(self.total_cost())
                # Legacy backends retain their heuristic search, but its cuts
                # and success flag are explicitly not an optimality certificate.
                g = np.asarray(self.gradient(), dtype=float)
                intercept = f - float(g @ x)
                upper, error = f, np.nan
            evals_x.append(x.copy())
            evals_f.append(f)
            cut_g.append(g)
            cut_b.append(intercept)
            if upper < best_upper:
                best_f, best_upper, best_error, best_x = f, upper, error, x.copy()

            if not certified:
                fx, xs = np.asarray(evals_f), np.stack(evals_x)
                slack = 1e-12 * max(1.0, abs(best_f))
                for i in range(len(cut_g)):
                    overshoot = float((xs @ cut_g[i] + cut_b[i] - fx).max())
                    if overshoot > slack:
                        cut_b[i] -= overshoot
                        n_repairs += 1

            lp_a_ub = np.column_stack([np.stack(cut_g), -np.ones(len(cut_g))])
            lp_b_ub = -np.asarray(cut_b)
            if fixed_a_ub is not None:
                lp_a_ub = np.vstack([lp_a_ub, fixed_a_ub])
                lp_b_ub = np.concatenate([lp_b_ub, fixed_b_ub])
            lp = linprog(
                c_lp, A_ub=lp_a_ub, b_ub=lp_b_ub, A_eq=lp_a_eq, b_eq=lp_b_eq,
                bounds=bounds, method="highs",
                options={"dual_feasibility_tolerance": 1e-10, "primal_feasibility_tolerance": 1e-10},
            )
            if not lp.success:
                status = f"lp_failed: {lp.message}"
                break
            # Modified/repaired models invalidate historical maxima. For the
            # certified path cuts only accumulate between precision rebuilds.
            candidate_lb = _cut_model_lower_bound(
                lp, cut_g, cut_b, caps, fixed_a_ub, fixed_b_ub, a_eq, b_eq, lower
            )
            lb = max(lb, candidate_lb) if certified else candidate_lb
            gap = best_upper - lb
            if print_steps:
                print(f"step {nit:3d} cost={f:.8g} upper={best_upper:.8g} "
                      f"lb={lb:.8g} gap={gap:.3g} refinements={n_refinements}")
            if gap < -gap_tol:
                status = "invalid_bound"
                break
            if 0 <= gap <= gap_tol:
                status = "converged"
                break
            x_next = np.clip(lp.x[:n], lower, caps)
            repeated = any(np.allclose(x_next, xe, rtol=1e-12, atol=1e-14) for xe in evals_x)
            if certified and best_error > gap_tol / 4 and (repeated or gap <= 4 * best_error + gap_tol):
                if n_refinements >= 8:
                    status = "precision_limit"
                    break
                try:
                    self.refine_intensity_precision(10)
                except OverflowError:
                    status = "precision_limit"
                    break
                n_refinements += 1
                # Cost scale can change: discard all previous cuts and bounds.
                evals_x.clear(); evals_f.clear(); cut_g.clear(); cut_b.clear()
                best_f = best_upper = best_error = np.inf
                lb = -np.inf
                x = best_x.copy()
                continue
            if repeated:
                status = "stalled"
                break
            x = x_next

        # A rebuild on the last iteration still needs a fresh returned value.
        if not np.isfinite(best_f):
            self.set_point(best_x)
            best_f = float(self.total_cost())
            certificate = self.graph.dual_cut()
            best_upper, best_error = certificate["upper_bound"], certificate["rounding_error"]
        self.set_point(best_x)
        gap = best_upper - lb
        return OptimizeResult(
            x=best_x, fun=best_f, upper_bound=best_upper, lb=lb, gap=gap,
            rounding_error=best_error, gap_tolerance=gap_tol,
            bound_certified=certified, n_precision_refinements=n_refinements,
            nit=nit, n_cut_repairs=n_repairs,
            polish_improved=False,
            success=(certified and status == "converged" and 0 <= gap <= gap_tol),
            status=status,
            message=f"cutting-plane: {status}, gap={gap:.3g}, certified={certified}, {n_refinements} precision refinement(s)",
        )

    def no_subgraphs(self) -> int:
        """
        Returns the number of subgraphs in the underlying Wasserstein network.

        Returns:
            int: The number of subgraphs present in the graph.
        """
        return self.graph.no_subgraphs()

    def print_diagnostics(self, subgraphs_too=False):
        """
        Prints diagnostic information about the current state of the alignment.

        Parameters
        ----------
        subgraphs_too : bool, optional
            If True, prints diagnostics for each subgraph in addition to the overall graph.

        Diagnostics Printed
        ------------------
        - Number of subgraphs
        - Number of empirical nodes
        - Number of theoretical nodes
        - Number of matching edges (dense factory)
        - Number of chain edges (1D chain factory)
        - Number of src-to-empirical edges
        - Number of theoretical-to-sink edges
        - Number of simple trash edges
        - Matching density
        - Scale factor (and its log10 value)
        - Total cost

        If `subgraphs_too` is True, for each subgraph:
        - Number of empirical nodes
        - Number of theoretical nodes
        - Cost
        - Matching density
        - Theoretical spectra involved
        """
        print("Diagnostics:")
        print("No subgraphs:", self.graph.no_subgraphs())
        print("No empirical nodes:", self.graph.count_empirical_nodes())
        print("No theoretical nodes:", self.graph.count_theoretical_nodes())
        print("No matching edges:", self.graph.count_matching_edges())
        print("No chain edges:", self.graph.count_chain_edges())
        print("No src-to-empirical edges:", self.graph.count_src_to_empirical_edges())
        print(
            "No theoretical-to-sink edges:",
            self.graph.count_theoretical_to_sink_edges(),
        )
        print("No simple trash edges:", self.graph.count_simple_trash_edges())
        print("Matching density:", self.graph.matching_density())
        print(
            "Scale factor:", self.scale_factor, f" log10: {np.log10(self.scale_factor)}"
        )
        print("Total cost:", self.graph.total_cost())
        if not subgraphs_too:
            return
        for ii in range(self.graph.no_subgraphs()):
            s = self.graph.get_subgraph(ii)
            print("Subgraph", ii, ":")
            print("  No. empirical nodes:", s.count_empirical_nodes())
            print("  No. theoretical nodes:", s.count_theoretical_nodes())
            print("  No. matching edges:", s.count_matching_edges())
            print("  No. chain edges:", s.count_chain_edges())
            print("  No. src-to-empirical edges:", s.count_src_to_empirical_edges())
            print(
                "  No. theoretical-to-sink edges:", s.count_theoretical_to_sink_edges()
            )
            print("  No. simple trash edges:", s.count_simple_trash_edges())
            print("  Cost:", s.total_cost())
            print("  Matching density:", s.matching_density())
            print("  Theoretical spectra involved:", s.theoretical_spectra_involved())


class ConstrainedSolver(DeconvSolver):
    """
    DeconvSolver with a total-mass equality constraint:

        sum_s(w_s * total_intensity_s) = total_empirical_intensity

    This couples the proportions so that components with extra unmatched peaks
    (diluted libraries) are naturally down-weighted without tuning
    theo_trash_cost.  The constraint is enforced during the call to
    optimize(), which carries it in the cutting-plane master LP.
    optimize_descent() retains the previous SLSQP implementation.

    All DeconvSolver methods (set_point, total_cost, gradient, flows, …)
    are inherited unchanged and work identically.

    Parameters
    ----------
    Same as DeconvSolver.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._emp_total = self.empirical_spectrum.sum_intensities
        self._theo_totals = np.array(
            [t.sum_intensities for t in self.theoretical_spectra]
        )

    def optimize_descent(
        self,
        x0: Optional[np.ndarray] = None,
        bounds: Optional[np.array] = None,
        maxiter: Optional[int] = None,
    ) -> OptimizeResult:
        """
        Minimize total transport cost subject to the total-mass constraint.

        Parameters
        ----------
        x0 : np.ndarray, optional
            Initial proportions.  Must satisfy the constraint.  Defaults to
            equal weights scaled to satisfy sum_s(w_s * I_s) = I_emp.
        maxiter : int, optional
            SLSQP iteration cap (default 2000).

        Returns
        -------
        scipy.optimize.OptimizeResult
            Standard scipy result; .x holds the optimal proportions.
        """
        n = len(self.theoretical_spectra)
        if x0 is None:
            w0 = self._emp_total / self._theo_totals.sum() # if spectra are normalized it's the same as x0 = np.ones(n)
            x0 = np.full(n, w0)
        if bounds is None:
            bounds = self._budget_bounds()

        # def cost_and_grad(w):
        #     self.set_point(w)
        #     return self.total_cost(), self.gradient()

        step = [0]
        def cost_and_grad(w):
            self.set_point(w)
            c = self.total_cost()
            g = self.gradient()
            print(f"step {step[0]:3d}  point=[{', '.join(f"{x:8.4f}" for x in w)}]  cost={c:8.4f}  grad=[{', '.join(f"{x:8.4f}" for x in g)}]")
            step[0] += 1
            return c, g        

        constraint = {
            "type": "eq",
            "fun": lambda w: np.dot(w, self._theo_totals) - self._emp_total,
            "jac": lambda w: self._theo_totals,
        }

        result = minimize(
            cost_and_grad,
            x0=x0,
            jac=True,
            method="SLSQP", # L-BFGS-B does not allow constraints
            bounds=bounds,
            constraints=constraint,
            options={
                "maxiter": 2000 if maxiter is None else maxiter,
                "ftol": self._ftol,
            },
        )
        self._warn_if_caps_binding(result.x)
        return result

    def optimize_cutting_plane(
        self,
        x0: Optional[np.ndarray] = None,
        max_iter: int = 200,
        tol: Optional[float] = None,
        polish: bool = False,
        *,
        bounds: Optional[np.array] = None,
        print_steps: bool = False,
    ) -> OptimizeResult:
        """Cutting-plane minimization under the total-mass equality.

        Same engine as :meth:`DeconvSolver.optimize_cutting_plane`; the
        constraint ``sum_s(w_s * I_s) = I_emp`` is carried natively by the
        per-iteration LP instead of SLSQP's iterative enforcement.  With
        ``polish=True`` the answer is refined by a warm-started
        SLSQP run and the cheaper evaluated point is returned.  See the base
        method for semantics of the returned result.
        """
        n = len(self.theoretical_spectra)
        if x0 is None:
            w0 = self._emp_total / self._theo_totals.sum()
            x0 = np.full(n, w0)
        result = self._kelley(
            x0=np.asarray(x0, dtype=float),
            a_eq=self._theo_totals,
            b_eq=self._emp_total,
            max_iter=max_iter,
            tol=tol,
            bounds=bounds, print_steps=print_steps,
        )
        if polish:
            result = self._polish(result, bounds=bounds)
        self._warn_if_caps_binding(result.x)
        return result


class MagnetsteinSolver(ConstrainedSolver):
    """
    ConstrainedSolver that normalizes all spectra to sum to 1 internally,
    reproducing magnetstein's dual-LP problem formulation.

    With unit-norm spectra the total-mass equality constraint reduces to
    sum(w) = 1, matching the LP's implicit mass-balance condition.
    experimental_trash_cost = MTD and theoretical_trash_cost = MTD_th
    correspond directly to magnetstein's penalty and penalty_th parameters.

    When ``MTD_th`` is given, the trash model defaults to *independent*
    two-sided trash (``independent_trash=True``): an unmatched empirical
    unit costs MTD and an unfilled theoretical unit costs MTD_th, charged
    separately.  This matches magnetstein's own LP — dualdeconv3/4
    explicitly forbid transport between the two auxiliary trash points.
    ``independent_trash=False`` restores wnet's annihilating asymmetric
    model, where an (unmatched-empirical, unfilled-theoretical) pair can
    cancel at min(MTD, MTD_th); that discount is not part of magnetstein's
    formulation and can zero out small components (e.g. alpha-pinene in the
    magnetstein perfumes mixture), but is kept as an option for continuity
    with earlier releases.

    Parameters
    ----------
    empirical_spectrum : Distribution
        The empirical spectrum (normalized internally to sum to 1).
    theoretical_spectra : Sequence[Distribution]
        A sequence of theoretical spectra (each normalized internally).
    distance : DistanceMetric
        Distance metric. Use DistanceMetric.L1 for 1D NMR spectra.
    MTD : float
        Maximum Transport Distance for the mix (experimental trash cost).
    MTD_th : float, optional
        Maximum Transport Distance for components (theoretical trash cost).
        If None, uses symmetric trash with cost MTD (and ``independent_trash``
        is ignored — the independent model needs both per-side penalties).
    method : str, optional
        Min-cost flow algorithm (default: ``"network_simplex"``). Ignored when ``solver`` is provided.
    solver : NetworkSimplex | CostScaling | CycleCanceling | CapacityScaling, optional
        Solver configuration object.  Takes precedence over ``method``.
    independent_trash : bool, optional
        Trash model when ``MTD_th`` is given (default True = independent,
        matching dualdeconv3/4; False = annihilating asymmetric).  See above.
    smooth_sigma : float, optional
        Gaussian pre-smoothing width in position units, applied to the
        empirical and all theoretical spectra before normalization (default
        0.0 = off).  For evenly gridded 1-D profile data only (see
        :meth:`Spectrum.smoothed`).  On profile 1H NMR, ~0.02 ppm (a typical
        linewidth) consistently helps when reference and mixture lineshapes
        differ — most strongly under field-inhomogeneity distortion — while
        widths much beyond the linewidth start blurring small components
        away; leave at 0 for centroided or non-uniformly gridded data.
    """

    def __init__(
        self,
        empirical_spectrum: Distribution,
        theoretical_spectra: Sequence[Distribution],
        distance: DistanceMetric,
        MTD: float,
        MTD_th: Optional[float] = None,
        method: str = None,
        solver=None,
        precision: float = 1e-3,
        p: float = 1.0,
        independent_trash: bool = True,
        smooth_sigma: float = 0.0,
    ) -> None:
        if smooth_sigma > 0:
            empirical_spectrum = empirical_spectrum.smoothed(smooth_sigma)
            theoretical_spectra = [
                t.smoothed(smooth_sigma) for t in theoretical_spectra
            ]
        emp = empirical_spectrum.normalized()
        theos = [t.normalized() for t in theoretical_spectra]
        # p == 1 keeps the historical caps bit-identically.  For p != 1 a
        # match is profitable while distance**p <= trash alternative
        # (bounded by MTD + MTD_th), so the cap must reach the profitable
        # radius (MTD + MTD_th)**(1/p) or the chain splitting severs
        # legitimate transport.
        if MTD_th is None:
            cap = MTD if p == 1.0 else max(MTD, (2.0 * MTD) ** (1.0 / p))
            super().__init__(
                emp,
                theos,
                distance,
                max_distance=cap,
                trash_cost=MTD,
                method=method,
                solver=solver,
                precision=precision,
                p=p,
            )
        elif independent_trash:
            # Independent trash: dumping a matched pair costs MTD + MTD_th, so
            # every match up to the profitable radius (MTD + MTD_th)**(1/p)
            # must exist; for p == 1 this is also exactly the chain/dense
            # factory parity threshold (split_distance >= C_exp + C_theo).
            cap = (MTD + MTD_th) ** (1.0 / p)
            super().__init__(
                emp,
                theos,
                distance,
                max_distance=cap,
                experimental_trash_cost=MTD,
                theoretical_trash_cost=MTD_th,
                method=method,
                solver=solver,
                precision=precision,
                p=p,
                independent_trash=True,
            )
        else:
            cap = (
                max(MTD, MTD_th)
                if p == 1.0
                else max(MTD, MTD_th, (MTD + MTD_th) ** (1.0 / p))
            )
            super().__init__(
                emp,
                theos,
                distance,
                max_distance=cap,
                experimental_trash_cost=MTD,
                theoretical_trash_cost=MTD_th,
                method=method,
                solver=solver,
                precision=precision,
                p=p,
            )


class _MassersteinBase(DeconvSolver):
    """Shared outer loop used by both Masserstein variants.  Not for direct
    use — instantiate :class:`MassersteinSolver2` (dualdeconv2-equivalent) or
    :class:`MassersteinSolver4` (dualdeconv4-equivalent).

    Default optimization uses cutting planes with ``w >= 0`` and
    ``sum(w) <= 1`` enforced in every master LP. ``deconvolve()`` preserves
    the dictionary return format used by the Masserstein compatibility API.
    Explicit ``optimize_descent()`` uses SLSQP under the same constraints.
    """

    _FTOL_CEILING = 1e-10  # cap relative-change tolerance for explicit descent

    def _face_start(self, x0):
        """A feasible starting point for the ``sum(w) <= 1`` polytope."""
        n = len(self.theoretical_spectra)
        if x0 is None:
            x0 = np.ones(n) / (2 * n)
        x0 = np.clip(np.asarray(x0, dtype=float), 0.0, None)
        total = x0.sum()
        return x0 / total if total > 1.0 else x0

    def optimize(self, x0=None, maxiter=None, *, bounds=None, tol=None,
                 polish=False, print_steps=False) -> OptimizeResult:
        """Default cutting planes, preserving positional ``x0, maxiter``."""
        return super().optimize(x0=x0, bounds=bounds, maxiter=maxiter,
                                tol=tol, polish=polish, print_steps=print_steps)

    def optimize_descent(
        self,
        x0: Optional[np.ndarray] = None,
        maxiter: Optional[int] = None,
        *,
        bounds: Optional[np.array] = None,
    ) -> OptimizeResult:
        """Descent under ``w >= 0`` and ``sum(w) <= 1``.

        Overrides :meth:`DeconvSolver.optimize_descent`, whose bounds-only L-BFGS-B
        would walk straight off this class's feasible set.  Used on its own
        and as the polish stage of :meth:`optimize_cutting_plane`; SLSQP is
        a poor primary optimizer here (see that method) but a serviceable
        finisher once the cutting planes have placed it inside the right
        linear piece.
        """
        n = len(self.theoretical_spectra)
        x0 = self._face_start(x0)

        def cost_and_grad(w):
            self.set_point(w)
            return self.total_cost(), self.gradient()

        constraints = [
            {
                "type": "ineq",
                "fun": lambda w: 1.0 - w.sum(),
                "jac": lambda w: -np.ones(n),
            }
        ]
        result = minimize(
            cost_and_grad,
            x0=x0,
            jac=True,
            method="SLSQP",
            bounds=self._budget_bounds() if bounds is None else bounds,
            constraints=constraints,
            options={
                "maxiter": 2000 if maxiter is None else maxiter,
                "ftol": min(self._ftol, self._FTOL_CEILING),
            },
        )
        self._warn_if_caps_binding(result.x)
        return result

    def optimize_cutting_plane(
        self,
        x0: Optional[np.ndarray] = None,
        max_iter: int = 200,
        tol: Optional[float] = None,
        polish: bool = False,
        *,
        bounds: Optional[np.array] = None,
        print_steps: bool = False,
    ) -> OptimizeResult:
        """Kelley cutting planes over ``w >= 0``, ``sum(w) <= 1``.

        The mass constraint is carried natively inside each master LP,
        and each evaluation contributes a supporting plane. This avoids
        the nonsmooth line-search stalls of explicit descent. Both
        :meth:`optimize` and :meth:`deconvolve` use this method by default.
        An interior solution simply leaves the mass constraint slack.

        Parameters and return value match
        :meth:`DeconvSolver.optimize_cutting_plane`; ``polish`` warm-starts
        :meth:`optimize_descent` (the constrained descent) and keeps the cheaper of
        the two evaluated points.
        """
        n = len(self.theoretical_spectra)
        result = self._kelley(
            x0=self._face_start(x0),
            a_eq=None,
            b_eq=None,
            max_iter=max_iter,
            tol=tol,
            a_ub=np.ones(n),
            b_ub=1.0,
            bounds=bounds, print_steps=print_steps,
        )
        if polish:
            result = self._polish(result, bounds=bounds)
        self._warn_if_caps_binding(result.x)
        return result

    def deconvolve(self, x0: Optional[np.ndarray] = None) -> dict:
        """Fit with cutting planes and return Masserstein-compatible fields.

        ``x0`` defaults to uniform ``1/(2k)``. Nonnegative weights and
        ``sum(w) <= 1`` are enforced throughout optimization. The returned
        ``on_simplex_face`` indicates whether the fitted weights sum to one
        within 1e-9; it does not describe an optimizer dispatch.
        """
        result = self.optimize(x0=x0)
        return {
            "probs": list(result.x),
            "fun": result.fun,
            "success": result.success,
            "on_simplex_face": bool(abs(result.x.sum() - 1.0) <= 1e-9),
        }


class MassersteinSolver2(_MassersteinBase):
    """
    Reproduces masserstein's ``dualdeconv2`` LP (one-sided / experimental-only
    trash).

    All spectra are normalised to sum to 1 internally (as dualdeconv2
    requires).  The distance is always LINF (= absolute distance in 1D, the
    dual of W1 / earth-mover's distance used by masserstein).

    dualdeconv2 prices transport at the true linear W1 cost with an
    experimental abyss at ``MTD``, and has *no theoretical abyss*: every unit
    of ``w_k * theo_k`` must reach an experimental position — a component is
    discarded only by driving ``w_k -> 0``, never by trashing theoretical
    mass.  Transporting a unit farther than ``MTD`` is never optimal in that
    LP (the experimental abyss at ``MTD`` is always cheaper), so ``MTD`` is
    already the LP's *effective* transport cap.  We reproduce that with:

      * ``max_distance = MTD`` — the effective cap; also keeps the 1D chain
        sparse (O(m+n)) instead of dense (O(m*n)) on real spectra.
      * ``experimental_trash_cost = MTD`` — the denoising penalty.
      * ``theoretical_trash_cost = theo_trash_mult * MTD`` — a numerical
        device only.  With experimental-only trash the inner min-cost-flow
        cost ``f(w)`` is degenerate / flat (un-routable theoretical mass is
        dropped for free, so the outer optimiser gets a zero gradient and
        returns its starting point).  Any cost strictly above the ``MTD``
        transport cap is never chosen over transporting or lowering ``w_k``,
        so it carries no flow at the optimum yet makes ``f(w)`` well-defined
        and convex for every ``w``.

    Residual caveats:
      * dualdeconv2 solves one joint LP (proportions = exact shadow prices);
        this is a nested optimisation (cutting planes over ``w``, inner MCF).  The
        objective and noise/sum behaviour match, but under degeneracy
        (near-collinear components) per-component proportions agree only to
        optimiser tolerance, not bit-exactly.
      * On raw unfiltered spectra the two formulations agree closely in
        controlled tests (single/multi-component, collinear decoys, dense
        overlapping + noise — see
        ``experiments/direct_dualdeconv2_{nofilter,multi,dense}.py``):
        objective to ~1e-5, signal fraction to ~1%, decoys zeroed.
      * On DENSE-noisy mass spectra (e.g. hemoglobin Part 2 in
        ``compare_dualdeconv2.py``) this reproduction breaks structurally:
        the nested empirical->theoretical MCF matches per peak with the
        sum ``Σ w_j*theo_j``, while dualdeconv2's joint LP couples all
        isotope positions of a component via ``Σ thr_ji Z_i ≤ 0``.  For
        inputs in this regime use ``masserstein.estimate_proportions``
        (which pre-filters to the theoretical envelope) or call
        ``dualdeconv2`` directly — not this class.

    ``deconvolve()`` uses cutting planes with bounds ``w_k >= 0`` and the explicit
    inequality constraint ``sum(w_k) <= 1``, which dualdeconv2 enforces
    implicitly via ``sum(probs) + sum(abyss) = 1, abyss >= 0``.

    Parameters
    ----------
    empirical_spectrum : Distribution
        Empirical spectrum (normalised internally to sum to 1).
    theoretical_spectra : Sequence[Distribution]
        Theoretical spectra (each normalised internally).
    MTD : float
        Maximum Transport Distance / denoising penalty (``penalty`` in
        dualdeconv2).
    theo_trash_mult : float, optional
        Multiplier on ``MTD`` for the +inf-proxy theoretical trash cost.
        Default 3×.  Sweeps against dualdeconv2 (deconvbench ms-hemoglobin
        and ``experiments/minimal_dense_noise_divergence.py``) show a plateau
        at 2-5× on which both cases agree with masserstein: below ~2× the
        nested MCF under-prices un-routable theoretical mass and proportions
        inflate, while above ~7× the overpriced theoretical abyss makes
        zeroing a small component cheaper than dumping its un-routable mass
        (the former 10× default silently zeroed a 5.6%-share component on
        the hemoglobin benchmark; L1 error 0.113 vs 0.016 at 3×).
    method : str, optional
        Min-cost flow algorithm.  Ignored when ``solver`` is provided.
    solver : NetworkSimplex | CostScaling | CycleCanceling | CapacityScaling, optional
        Solver configuration object.  Takes precedence over ``method``.
    """

    def __init__(
        self,
        empirical_spectrum: Distribution,
        theoretical_spectra: Sequence[Distribution],
        MTD: float,
        theo_trash_mult: float = 3.0,
        method: str = None,
        solver=None,
        precision: float = 1e-3,
    ) -> None:
        emp = empirical_spectrum.normalized()
        theos = [t.normalized() for t in theoretical_spectra]
        super().__init__(
            emp,
            theos,
            distance=DistanceMetric.LINF,
            max_distance=MTD,
            experimental_trash_cost=MTD,
            theoretical_trash_cost=theo_trash_mult * MTD,
            method=method,
            solver=solver,
            precision=precision,
        )


class MassersteinSolver4(_MassersteinBase):
    """
    Reproduces masserstein's ``dualdeconv4`` LP (symmetric two-sided trash).

    Like :class:`MassersteinSolver2` but with a *real* theoretical-side
    denoising penalty ``MTD_th`` instead of the +inf-proxy device.  Maps
    directly onto ``dualdeconv4(penalty=MTD, penalty_th=MTD_th)``.

    dualdeconv4 has two **independent** abysses (experimental at ``MTD``,
    theoretical at ``MTD_th``); transport between them costs ``MTD + MTD_th``
    and never occurs.  This is *not* wnet's default asymmetric trash, which
    lets an (unmatched-empirical, unfilled-theoretical) pair annihilate at
    ``min(MTD, MTD_th)`` — that discount inflates ``w`` and dumps forced
    theoretical mass for free.  We therefore use the network's
    ``add_independent_asymmetric_trash`` (``independent_trash=True``) and set
    the transport cap to ``MTD + MTD_th`` so every match that beats dumping
    both sides exists.  With this, the nested-MCF cost equals dualdeconv4's
    LP value (the matched/unmatched split and ``w`` agree).

    Parameters
    ----------
    empirical_spectrum : Distribution
        Empirical spectrum (normalised internally to sum to 1).
    theoretical_spectra : Sequence[Distribution]
        Theoretical spectra (each normalised internally).
    MTD : float
        Maximum Transport Distance / experimental-side denoising penalty
        (``penalty`` in dualdeconv4).
    MTD_th : float
        Theoretical-side denoising penalty (``penalty_th`` in dualdeconv4).
    method : str, optional
        Min-cost flow algorithm.  Ignored when ``solver`` is provided.
    solver : NetworkSimplex | CostScaling | CycleCanceling | CapacityScaling, optional
        Solver configuration object.  Takes precedence over ``method``.
    """

    def __init__(
        self,
        empirical_spectrum: Distribution,
        theoretical_spectra: Sequence[Distribution],
        MTD: float,
        MTD_th: float,
        method: str = None,
        solver=None,
        precision: float = 1e-3,
    ) -> None:
        _require_independent_trash_support()
        emp = empirical_spectrum.normalized()
        theos = [t.normalized() for t in theoretical_spectra]
        super().__init__(
            emp,
            theos,
            distance=DistanceMetric.LINF,
            max_distance=MTD + MTD_th,
            experimental_trash_cost=MTD,
            theoretical_trash_cost=MTD_th,
            method=method,
            solver=solver,
            precision=precision,
            independent_trash=True,
        )


def MassersteinSolver(
    empirical_spectrum: Distribution,
    theoretical_spectra: Sequence[Distribution],
    MTD: float,
    MTD_th: Optional[float] = None,
    theo_trash_mult: float = 3.0,
    method: str = None,
    solver=None,
    precision: float = 1e-3,
):
    """Backwards-compatibility shim.  Dispatches to :class:`MassersteinSolver2`
    when ``MTD_th`` is None and to :class:`MassersteinSolver4` otherwise.
    New code should instantiate the explicit class."""
    if MTD_th is None:
        return MassersteinSolver2(
            empirical_spectrum,
            theoretical_spectra,
            MTD=MTD,
            theo_trash_mult=theo_trash_mult,
            method=method,
            solver=solver,
            precision=precision,
        )
    return MassersteinSolver4(
        empirical_spectrum,
        theoretical_spectra,
        MTD=MTD,
        MTD_th=MTD_th,
        method=method,
        solver=solver,
        precision=precision,
    )
