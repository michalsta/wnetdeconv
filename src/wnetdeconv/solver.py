import warnings
from collections import namedtuple
from collections.abc import Sequence
from typing import Callable, Optional, Union, List, Tuple
import numpy as np
from scipy.optimize import minimize, OptimizeResult

from wnet import Distribution, WassersteinNetwork

_Flow = namedtuple("Flow", ["empirical_peak_idx", "theoretical_peak_idx", "flow"])
from wnet.distances import DistanceMetric


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
        Maximum allowed distance for matching peaks.
    trash_cost : int or float, optional
        Cost for assigning unmatched peaks to trash (symmetric). Used as fallback for
        experimental_trash_cost / theoretical_trash_cost when only one is set.
    scale_factor : None, int, or float, optional
        Scaling factor for intensities and costs. If None, it is computed from ``precision``.
    precision : float, optional
        Desired relative precision of the cost output (fraction of the theoretical cost
        upper bound ``max_cost_per_unit_flow * max_sum_intensity``).  Drives both the
        auto scale_factor and the ``ftol`` stop criterion passed to scipy optimizers.
        Ignored when ``scale_factor`` is supplied explicitly.  Default 1e-3 (≈ 3
        significant figures).
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
        factory (default False = chain in 1D).  Forwarded to
        :class:`WassersteinNetwork`.

    Attributes
    ----------
    scale_factor : float
        The scaling factor used for intensities and costs.
    empirical_spectrum : Distribution
        The scaled empirical spectrum.
    theoretical_spectra : list[Distribution]
        The scaled theoretical spectra.
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
        for name, val in [
            ("trash_cost", trash_cost),
            ("experimental_trash_cost", experimental_trash_cost),
            ("theoretical_trash_cost", theoretical_trash_cost),
        ]:
            if val is not None and not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be a number")
        if scale_factor is not None and not isinstance(scale_factor, (int, float)):
            raise TypeError("scale_factor must be a number")

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
            theoretical_trash_cost
            if theoretical_trash_cost is not None
            else trash_cost
        )
        if asymmetric:
            active_costs = [c for c in (eff_exp, eff_theo) if c is not None]
        else:
            active_costs = [trash_cost]

        ALMOST_MAXINT = 2**60
        empirical_sum_intensity = empirical_spectrum.sum_intensities
        theoretical_sum_intensity = sum(
            t.sum_intensities for t in theoretical_spectra
        )
        max_sum_intensity = max(empirical_sum_intensity, theoretical_sum_intensity)
        max_cost_per_unit_flow = max([max_distance] + active_costs)
        min_cost_per_unit_flow = min([max_distance] + active_costs)

        if scale_factor is None:
            # The cost integer is sum(int_flow * int_dist), with int_flow ≈
            # flow*sf_intensity and int_dist ≈ dist*sf_distance.  The two
            # scaling factors are independent — distance scaling controls
            # quantization of per-arc distances (applied after the metric, so
            # int_dist = int(dist * sf_distance)); intensity scaling controls
            # quantization of per-peak intensities (int_intensity =
            # int(intensity * sf_intensity)).  They affect different sources
            # of quantization error, so they are tuned separately from
            # `precision`.
            #
            # Distance: per-arc absolute distance error is 1/sf_distance.  To
            # keep the relative cost error per arc at most `precision`, the
            # bucket size 1/sf_distance must be ≤ precision * min_cost (the
            # smallest cost-per-unit-flow in use — typically MTD or the trash
            # cost).  Inverting:
            #   sf_distance = 1 / (precision * min_cost_per_unit_flow)
            # which gives int(min_cost * sf_distance) = 1/precision integer
            # ticks within the smallest cost class.
            sf_distance = 1.0 / (precision * min_cost_per_unit_flow)

            # Intensity: per-arc absolute intensity error is 1/sf_intensity.
            # Choose sf_intensity so that int(total_intensity) = 1/precision
            # discrete flow levels.  Inverting:
            #   sf_intensity = 1 / (precision * max_sum_intensity).
            sf_intensity = 1.0 / (precision * max_sum_intensity)

            # int64 cap: per-arc int cost ≤ max_cost * sf_distance *
            # max_sum_intensity * sf_intensity = (sf_distance * sf_intensity)
            # * (max_cost * max_sum_intensity).  Cap so total fits in 2^60.
            cap_product = ALMOST_MAXINT / (
                max_cost_per_unit_flow * max_sum_intensity
            )
            product = sf_distance * sf_intensity
            if product > cap_product:
                shrink = np.sqrt(cap_product / product)
                sf_distance *= shrink
                sf_intensity *= shrink
                warnings.warn(
                    f"Requested precision {precision} exceeds int64 capacity "
                    f"for this dataset (max_cost={max_cost_per_unit_flow:.3g}, "
                    f"max_sum_intensity={max_sum_intensity:.3g}); shrinking "
                    f"sf_distance/sf_intensity by {shrink:.3g}.  Achieved "
                    f"relative precision ~{precision/shrink:.2e}."
                )
            assert sf_distance > 0 and sf_intensity > 0, (
                "Can't auto-compute sensible sf_distance/sf_intensity. "
                "You might have some luck with setting scale_factor manually, "
                "but it probably means something about your data or trash_cost "
                "is off."
            )
            if int(min_cost_per_unit_flow * sf_distance) < 1:
                raise ValueError(
                    f"Auto-computed sf_distance={sf_distance:.3g} cannot "
                    f"represent min_cost_per_unit_flow={min_cost_per_unit_flow:.3g} "
                    f"as a positive integer (the graph would have no edges).  "
                    f"empirical_sum_intensity={empirical_sum_intensity:.3g}, "
                    f"theoretical_sum_intensity={theoretical_sum_intensity:.3g}.  "
                    f"Normalize the spectra, pass an explicit scale_factor, or "
                    f"relax precision."
                )
        else:
            # Backwards-compat: explicit scale_factor sets both factors equal.
            sf_distance = float(scale_factor)
            sf_intensity = float(scale_factor)

        self.sf_distance = sf_distance
        self.sf_intensity = sf_intensity
        # Compatibility alias.  When auto-computed the factors are unequal;
        # `scale_factor` then reports the geometric mean (matches the legacy
        # quadratic unscaling factor sf_distance*sf_intensity = scale_factor^2).
        self.scale_factor = float(np.sqrt(sf_distance * sf_intensity))
        self._ftol = 1.0 / (sf_distance * sf_intensity)

        def _scale_spec(spec):
            new_pos = np.asarray(spec.positions, dtype=np.float64) * sf_distance
            new_int = (
                np.asarray(getattr(spec, "original_intensities", spec.intensities),
                           dtype=np.float64) * sf_intensity
            )
            return type(spec)(new_pos, new_int, label=spec.label)

        self.empirical_spectrum = _scale_spec(empirical_spectrum)
        self.theoretical_spectra = [_scale_spec(t) for t in theoretical_spectra]

        self.graph = WassersteinNetwork(
            self.empirical_spectrum,
            self.theoretical_spectra,
            distance,
            int(max_distance * sf_distance),
            force_dense_1d=force_dense_1d,
            method=method,
            solver=solver,
        )
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
            self.graph.add_independent_asymmetric_trash(
                int(eff_exp * sf_distance), int(eff_theo * sf_distance)
            )
        elif asymmetric:
            if eff_exp is not None:
                self.graph.add_experimental_trash(int(eff_exp * sf_distance))
            if eff_theo is not None:
                self.graph.add_theoretical_trash(int(eff_theo * sf_distance))
        else:
            self.graph.add_simple_trash(int(trash_cost * sf_distance))
        self.graph.build()
        self.point = None

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
        return self.graph.total_cost() / (self.sf_distance * self.sf_intensity)

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
            result.append(_Flow(empirical_peak_idx, theoretical_peak_idx, flow / self.sf_intensity))
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
        return (
            self.graph.spectrum_proportion_derivatives().astype(float)
            / (self.sf_distance * self.sf_intensity)
        )

    def gradient_fast_approx(self) -> np.ndarray:
        """Fast, APPROXIMATE gradient (dual-potential difference instead of the
        residual shortest-path marginal).

        Much cheaper (skips the per-subgraph Dijkstra) but returns a
        different, basis-dependent gradient: a lower bound on the true
        marginal, exact only on the optimal flow support.  Opt-in; do not use
        as a drop-in replacement for gradient() without validating convergence.
        """
        return (
            self.graph.spectrum_proportion_derivatives_fast_approx().astype(float)
            / (self.sf_distance * self.sf_intensity)
        )

    def optimize(self, x0: Optional[np.ndarray] = None) -> OptimizeResult:
        """
        Minimize total transport cost over non-negative spectrum proportions.

        Parameters
        ----------
        x0 : np.ndarray, optional
            Initial proportions. Defaults to a vector of ones.

        Returns
        -------
        scipy.optimize.OptimizeResult
            Standard scipy result; .x holds the optimal proportions.
        """
        n = len(self.theoretical_spectra)
        if x0 is None:
            x0 = np.ones(n)

        def cost_and_grad(w):
            self.set_point(w)
            return self.total_cost(), self.gradient()

        return minimize(
            cost_and_grad,
            x0=x0,
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, None)] * n,
            options={"ftol": self._ftol},
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
        print("No theoretical-to-sink edges:", self.graph.count_theoretical_to_sink_edges())
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
            print("  No. theoretical-to-sink edges:", s.count_theoretical_to_sink_edges())
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
    optimize(), which uses SLSQP instead of L-BFGS-B.

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

    def optimize(self, x0: Optional[np.ndarray] = None) -> OptimizeResult:
        """
        Minimize total transport cost subject to the total-mass constraint.

        Parameters
        ----------
        x0 : np.ndarray, optional
            Initial proportions.  Must satisfy the constraint.  Defaults to
            equal weights scaled to satisfy sum_s(w_s * I_s) = I_emp.

        Returns
        -------
        scipy.optimize.OptimizeResult
            Standard scipy result; .x holds the optimal proportions.
        """
        n = len(self.theoretical_spectra)
        if x0 is None:
            w0 = self._emp_total / self._theo_totals.sum()
            x0 = np.full(n, w0)

        def cost_and_grad(w):
            self.set_point(w)
            return self.total_cost(), self.gradient()

        constraint = {
            "type": "eq",
            "fun": lambda w: np.dot(w, self._theo_totals) - self._emp_total,
            "jac": lambda w: self._theo_totals,
        }

        return minimize(
            cost_and_grad,
            x0=x0,
            jac=True,
            method="SLSQP",
            bounds=[(0.0, None)] * n,
            constraints=constraint,
            options={"maxiter": 2000, "ftol": self._ftol},
        )


class MagnetsteinSolver(ConstrainedSolver):
    """
    ConstrainedSolver that normalizes all spectra to sum to 1 internally,
    reproducing magnetstein's dual-LP problem formulation.

    With unit-norm spectra the total-mass equality constraint reduces to
    sum(w) = 1, matching the LP's implicit mass-balance condition.
    experimental_trash_cost = MTD and theoretical_trash_cost = MTD_th
    correspond directly to magnetstein's penalty and penalty_th parameters.

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
        If None, uses symmetric trash with cost MTD.
    method : str, optional
        Min-cost flow algorithm (default: ``"network_simplex"``). Ignored when ``solver`` is provided.
    solver : NetworkSimplex | CostScaling | CycleCanceling | CapacityScaling, optional
        Solver configuration object.  Takes precedence over ``method``.
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
    ) -> None:
        emp = empirical_spectrum.normalized()
        theos = [t.normalized() for t in theoretical_spectra]
        if MTD_th is None:
            super().__init__(
                emp,
                theos,
                distance,
                max_distance=MTD,
                trash_cost=MTD,
                method=method,
                solver=solver,
                precision=precision,
            )
        else:
            super().__init__(
                emp,
                theos,
                distance,
                max_distance=max(MTD, MTD_th),
                experimental_trash_cost=MTD,
                theoretical_trash_cost=MTD_th,
                method=method,
                solver=solver,
                precision=precision,
            )


class _MassersteinBase(DeconvSolver):
    """Shared outer loop used by both Masserstein variants.  Not for direct
    use — instantiate :class:`MassersteinSolver2` (dualdeconv2-equivalent) or
    :class:`MassersteinSolver4` (dualdeconv4-equivalent).

    Two-pass solve:
      1.  L-BFGS-B with bounds ``w >= 0`` only (no sum constraint).  ``f(w)``
          is convex in ``w``, so the unconstrained minimum is the constrained
          minimum *iff* it satisfies ``sum(w) <= 1``.  When that holds (the
          common case) the cheap bounds-only path is the answer.
      2.  If the L-BFGS-B output violates ``sum(w) > 1``, the constraint is
          binding; re-solve with SLSQP on the explicit ``sum(w) <= 1`` face.

    A naive "check the gradient sign at the face centre" dispatch was tried
    first and fails: ``f(w)`` is piecewise linear, so the gradient near the
    constraint can sit at a kink where the right-side subgradient has the
    wrong sign for the KKT test.  The run-then-check above sidesteps that.
    Both inner solves clamp their tolerance to a safe ceiling so the cheap
    relative-change-stopping rules don't terminate on flat-plateau regions
    before the optimum is reached (the auto ``self._ftol`` derived from
    ``precision`` is calibrated for cost-output accuracy, not optimiser
    stopping).
    """

    _FTOL_CEILING = 1e-10  # see deconvolve() for why

    def deconvolve(self, x0: Optional[np.ndarray] = None) -> dict:
        """
        Find optimal component proportions, matching dualdeconv2/4's output.

        Parameters
        ----------
        x0 : np.ndarray, optional
            Initial proportions.  Defaults to uniform ``1/(2k)`` (interior of
            feasible set, away from the ``sum(w)=1`` boundary).

        Returns
        -------
        dict
            probs   : list[float]  – weight of each theoretical spectrum
            fun     : float        – optimal transport cost
            success : bool
            on_simplex_face : bool – True iff the ``sum(w) = 1`` constraint
                                     was active and SLSQP was used; False
                                     iff bounds-only L-BFGS-B sufficed.
        """
        n = len(self.theoretical_spectra)
        if x0 is None:
            x0 = np.ones(n) / (2 * n)

        def cost_and_grad(w):
            self.set_point(w)
            return self.total_cost(), self.gradient()

        # Pass 1: bounds-only L-BFGS-B with the user's auto ``ftol`` (no
        # extra clamp).  Its only job is to decide which side of the
        # ``sum(w) = 1`` face the optimum lies on; the location of the
        # optimum itself is refined in pass 2 when needed, so the loose
        # default is fine here and saves iterations.
        result = minimize(
            cost_and_grad,
            x0=x0,
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, None)] * n,
            options={"maxiter": 2000, "ftol": self._ftol, "gtol": 1e-10},
        )

        on_simplex_face = bool(result.x.sum() > 1.0 + 1e-9)

        if on_simplex_face:
            # Pass 2: SLSQP with the explicit ``sum(w) <= 1`` constraint.
            # Re-start from the L-BFGS-B output projected onto the face so
            # SLSQP doesn't have to traverse the same descent again.
            # SLSQP's relative-change stopping is what gets tripped by the
            # auto-ftol, so clamp to a safer ceiling here (and only here).
            x_init = result.x
            if x_init.sum() > 1.0:
                x_init = x_init / x_init.sum()  # project onto sum=1
            constraints = [{
                "type": "ineq",
                "fun": lambda w: 1.0 - w.sum(),
                "jac": lambda w: -np.ones(n),
            }]
            result = minimize(
                cost_and_grad,
                x0=x_init,
                jac=True,
                method="SLSQP",
                bounds=[(0.0, None)] * n,
                constraints=constraints,
                options={"maxiter": 2000,
                         "ftol": min(self._ftol, self._FTOL_CEILING)},
            )
        return {"probs": list(result.x), "fun": result.fun,
                "success": result.success,
                "on_simplex_face": on_simplex_face}


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
        this is a nested optimisation (SLSQP over ``w``, inner MCF).  The
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

    ``deconvolve()`` uses SLSQP with bounds ``w_k >= 0`` and the explicit
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
        Default 10× is what fixes the minimal-divergence example
        (``experiments/minimal_dense_noise_divergence.py``); below ~10× the
        nested MCF under-prices un-routable theoretical mass relative to
        masserstein's real-distance transport.  Should be at least as large
        as the maximum inter-isotope distance you expect un-routed mass to
        need to travel (in m/z units of ``MTD``).  Above ~few hundred it can
        lose precision via the auto ``scale_factor``.
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
        theo_trash_mult: float = 10.0,
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
    theo_trash_mult: float = 10.0,
    method: str = None,
    solver=None,
    precision: float = 1e-3,
):
    """Backwards-compatibility shim.  Dispatches to :class:`MassersteinSolver2`
    when ``MTD_th`` is None and to :class:`MassersteinSolver4` otherwise.
    New code should instantiate the explicit class."""
    if MTD_th is None:
        return MassersteinSolver2(
            empirical_spectrum, theoretical_spectra, MTD=MTD,
            theo_trash_mult=theo_trash_mult,
            method=method, solver=solver, precision=precision,
        )
    return MassersteinSolver4(
        empirical_spectrum, theoretical_spectra, MTD=MTD, MTD_th=MTD_th,
        method=method, solver=solver, precision=precision,
    )
