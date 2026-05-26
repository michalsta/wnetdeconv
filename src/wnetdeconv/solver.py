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
        if asymmetric:
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
            active_costs = [c for c in (eff_exp, eff_theo) if c is not None]
        else:
            active_costs = [trash_cost]

        if scale_factor is None:
            ALMOST_MAXINT = 2**60
            empirical_sum_intensity = empirical_spectrum.sum_intensities
            theoretical_sum_intensity = sum(
                t.sum_intensities for t in theoretical_spectra
            )
            max_sum_intensity = max(empirical_sum_intensity, theoretical_sum_intensity)

            # Output-precision constraint (original): integer resolution
            # 1/sf^2 should be ~precision of the worst-case absolute cost
            # max_cost*max_sum_intensity, so
            #   sf >= sqrt(1/(precision * cost_scale)).
            # This used to be the only constraint.  When the experimental
            # spectrum has huge unnormalized intensities (raw MS counts ~1e7+),
            # the formula drops sf so low that int(max_distance*sf) rounds to 0
            # and the graph factory builds zero edges (silent failure).
            #
            # Per-edge floor: int(min_cost_per_unit_flow * sf) must be at least
            # MIN_COST_TICKS so the cost map has usable resolution.  Below ~25
            # the gradient signal is too coarse for L-BFGS-B to make progress
            # (empirical: scaled_MTD=10 on pbttt → 1 iter, scaled_MTD=25 → 36
            # iters with a real optimum).  Going higher than ~25 produces more
            # accurate cost numbers but multiplies LEMON's pivot count on
            # large graphs (cold solve scales roughly with sf), so we cap the
            # auto floor at MIN_COST_TICKS rather than tying it to precision.
            # Pass scale_factor explicitly (or tighten precision) when more
            # input precision is needed.
            MIN_COST_TICKS = 25
            max_cost_per_unit_flow = max([max_distance] + active_costs)
            min_cost_per_unit_flow = min([max_distance] + active_costs)
            cost_scale = max_cost_per_unit_flow * max_sum_intensity
            sf_output = np.sqrt(1.0 / (precision * cost_scale))
            sf_floor  = MIN_COST_TICKS / min_cost_per_unit_flow
            desired_sf = max(sf_output, sf_floor)
            max_sf = np.sqrt(ALMOST_MAXINT / cost_scale)
            if desired_sf > max_sf:
                achieved_ticks = max_sf * min_cost_per_unit_flow
                achieved_out   = 1.0 / (max_sf**2 * cost_scale)
                warnings.warn(
                    f"Requested precision {precision} exceeds int64 capacity for this "
                    f"dataset (cost_scale={cost_scale:.3g}, "
                    f"min_cost={min_cost_per_unit_flow:.3g}); clamping scale_factor to "
                    f"{max_sf:.3g}.  Achieved cost precision {achieved_out:.2e} "
                    f"(relative), min-cost integer ticks {achieved_ticks:.1f}."
                )
                scale_factor = max_sf
            else:
                scale_factor = desired_sf
            assert (
                scale_factor > 0
            ), "Can't auto-compute a sensible scale factor. You might have some luck with setting it manually, but it probably means something about your data or trash_cost is off."
            if int(min_cost_per_unit_flow * scale_factor) < 1:
                raise ValueError(
                    f"Auto-computed scale_factor={scale_factor:.3g} cannot represent "
                    f"min_cost_per_unit_flow={min_cost_per_unit_flow:.3g} as a "
                    f"positive integer (the graph would have no edges).  "
                    f"empirical_sum_intensity={empirical_sum_intensity:.3g}, "
                    f"theoretical_sum_intensity={theoretical_sum_intensity:.3g}.  "
                    f"Normalize the spectra, pass an explicit scale_factor, or "
                    f"relax precision."
                )

        self.scale_factor = scale_factor
        self._ftol = 1.0 / (scale_factor * scale_factor)
        self.empirical_spectrum = empirical_spectrum.positions_intensities_scaled(
            scale_factor
        )
        self.theoretical_spectra = [
            t.positions_intensities_scaled(scale_factor) for t in theoretical_spectra
        ]

        self.graph = WassersteinNetwork(
            self.empirical_spectrum,
            self.theoretical_spectra,
            distance,
            int(max_distance * scale_factor),
            force_dense_1d=force_dense_1d,
            method=method,
            solver=solver,
        )
        if asymmetric:
            if eff_exp is not None:
                self.graph.add_experimental_trash(int(eff_exp * scale_factor))
            if eff_theo is not None:
                self.graph.add_theoretical_trash(int(eff_theo * scale_factor))
        else:
            self.graph.add_simple_trash(int(trash_cost * scale_factor))
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
        return self.graph.total_cost() / (self.scale_factor * self.scale_factor)

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
            result.append(_Flow(empirical_peak_idx, theoretical_peak_idx, flow / self.scale_factor))
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
            / (self.scale_factor * self.scale_factor)
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
            / (self.scale_factor * self.scale_factor)
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


class MassersteinSolver(DeconvSolver):
    """
    Reproduces masserstein's ``dualdeconv2`` / ``dualdeconv4``.

    All spectra are normalized to sum to 1 internally (as dualdeconv2
    requires).  The distance is always LINF (= absolute distance in 1D, the
    dual of W1 / earth mover's distance used by masserstein).

    Faithful model of dualdeconv2's LP
    ----------------------------------
    dualdeconv2 prices transport at the true linear W1 cost with an
    experimental abyss at ``MTD``, and has *no theoretical abyss*: every unit
    of ``w_k * theo_k`` must reach an experimental position — a component is
    discarded only by driving ``w_k -> 0``, never by trashing theoretical
    mass.  Transporting a unit farther than ``MTD`` is never optimal in that
    LP (the experimental abyss at ``MTD`` is always cheaper), so ``MTD`` is
    already the LP's *effective* transport cap.  We reproduce that with:

      * ``max_distance = MTD`` — the effective cap; also keeps the 1D chain
        sparse (O(m+n)) instead of dense (O(m*n)) on real spectra;
      * ``experimental_trash_cost = MTD`` — the denoising penalty;
      * ``theoretical_trash_cost = 2*MTD`` (dualdeconv2 case).  This is a
        numerical device only: with experimental-only trash the inner
        min-cost-flow cost ``f(w)`` is degenerate/flat (un-routable
        theoretical mass is dropped for free, so the outer optimizer gets a
        zero gradient and returns its starting point — the old bug).  Any
        cost strictly above the ``MTD`` transport cap is never chosen over
        transporting or lowering ``w_k``, so it carries no flow at the
        optimum (= "no theoretical abyss; drop the component by lowering
        w_k") yet makes ``f(w)`` well-defined and convex for every ``w``.
        The multiplier is kept small (2x) on purpose: the auto
        ``scale_factor`` divides by ``max_cost_per_unit_flow``, so a large
        value would shrink it and lose m/z precision.  A sweep (2/4/8/20x)
        showed 2x gives the best Part-1 agreement (L1 ~2e-7 vs dualdeconv2)
        while 8x already degrades it ~4x, with no compensating gain — the
        fixed-integer network's dynamic range makes a true +inf infeasible,
        so this is a deliberate approximation, exact for fully-placeable and
        fully-unplaceable components, slightly soft for partial placement.

    Residual caveats:
      * dualdeconv2 solves one joint LP (proportions = exact shadow prices);
        this is a nested optimization (SLSQP over ``w``, inner MCF).  The
        objective and noise/sum behaviour match, but under degeneracy
        (near-collinear components) per-component proportions agree only to
        optimizer tolerance, not bit-exactly.
      * On raw unfiltered spectra the two formulations agree closely in
        controlled tests (single/multi-component, collinear decoys, dense
        overlapping + noise — see
        ``experiments/direct_dualdeconv2_{nofilter,multi,dense}.py``):
        objective to ~1e-5, signal fraction to ~1%, decoys zeroed.
      * On DENSE-noisy mass spectra (e.g. hemoglobin Part 2 in
        ``compare_dualdeconv2.py``) this reproduction breaks structurally:
        the nested empirical->theoretical MCF matches per peak with the
        sum Σ w_j*theo_j, while dualdeconv2's joint LP couples all isotope
        positions of a component via Σ thr_ji Z_i ≤ 0.  An 11-config grid
        search (``experiments/grid_search_masserstein.py``) over
        max_distance and theoretical_trash_cost found that NO setting
        bridges the gap — larger max_distance makes it worse (more noise
        targets), larger theo_trash does nothing (theo-trash never fires at
        the optimum on dense noise), and either breaks the minimal case
        first.  Cross-scoring confirms it: at w_wnet, masserstein's own LP
        gives ~100x worse cost than at w_dd2 — i.e. wnetdeconv's reported
        ``fun`` is its own (lenient) model, not a competitive solution to
        masserstein's LP.  For inputs in this regime use
        ``masserstein.estimate_proportions`` (which pre-filters to the
        theoretical envelope, the agreement regime) or call
        ``dualdeconv2`` directly — not this class.

    ``deconvolve()`` uses SLSQP with bounds w_k >= 0 and the explicit
    inequality constraint sum(w_k) <= 1, which dualdeconv2 enforces implicitly
    via sum(probs) + sum(abyss) = 1, abyss >= 0.

    For the symmetric case (MTD_th=None) this reproduces dualdeconv2;
    with MTD_th set it reproduces dualdeconv4 (real theoretical penalty
    MTD_th, still with the unbounded transport metric).

    Parameters
    ----------
    empirical_spectrum : Distribution
        Empirical spectrum (normalized internally to sum to 1).
    theoretical_spectra : Sequence[Distribution]
        Theoretical spectra (each normalized internally).
    MTD : float
        Maximum Transport Distance / denoising penalty (``penalty`` in dualdeconv2).
    MTD_th : float, optional
        Separate theoretical trash cost.  None → symmetric = dualdeconv2;
        non-None → asymmetric = dualdeconv4.
    theo_trash_mult : float, optional
        Multiplier on MTD for the +inf-proxy theoretical trash cost
        (dualdeconv2 path only).  Default 10x is what fixes the
        minimal-divergence example
        (``experiments/minimal_dense_noise_divergence.py``); below ~10x the
        nested MCF under-prices un-routable theoretical mass relative to
        masserstein's real-distance transport.  Should be at least as large as
        the maximum inter-isotope distance you expect un-routed mass to need
        to travel (in m/z units of MTD).  Above ~few hundred it can lose
        precision via the auto ``scale_factor``.
    method : str, optional
        Min-cost flow algorithm. Ignored when ``solver`` is provided.
    solver : NetworkSimplex | CostScaling | CycleCanceling | CapacityScaling, optional
        Solver configuration object.  Takes precedence over ``method``.
    """

    def __init__(
        self,
        empirical_spectrum: Distribution,
        theoretical_spectra: Sequence[Distribution],
        MTD: float,
        MTD_th: Optional[float] = None,
        theo_trash_mult: float = 10.0,
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
                distance=DistanceMetric.LINF,
                max_distance=MTD,
                experimental_trash_cost=MTD,
                # effective +inf: large enough that the optimizer prefers
                # lowering w_k over carrying flow on this edge — i.e. mimics
                # masserstein's "no theoretical abyss; transport at real
                # distance".  Default 10x covers the typical asymmetric-isotope
                # case; user can dial up if inter-isotope distances >> MTD.
                theoretical_trash_cost=theo_trash_mult * MTD,
                method=method,
                solver=solver,
                precision=precision,
            )
        else:
            super().__init__(
                emp,
                theos,
                distance=DistanceMetric.LINF,
                max_distance=max(MTD, MTD_th),
                experimental_trash_cost=MTD,
                theoretical_trash_cost=MTD_th,
                method=method,
                solver=solver,
                precision=precision,
            )

    def deconvolve(self, x0: Optional[np.ndarray] = None) -> dict:
        """
        Find optimal component proportions, matching dualdeconv2's output format.

        Parameters
        ----------
        x0 : np.ndarray, optional
            Initial proportions. Defaults to uniform 1/(2k) (interior of feasible set).

        Returns
        -------
        dict
            probs : list[float]  – weight of each theoretical spectrum
            fun   : float        – optimal transport cost (= dual LP objective)
            success : bool
        """
        n = len(self.theoretical_spectra)
        if x0 is None:
            x0 = np.ones(n) / (2 * n)

        def cost_and_grad(w):
            self.set_point(w)
            return self.total_cost(), self.gradient()

        constraints = [{
            "type": "ineq",
            "fun": lambda w: 1.0 - w.sum(),
            "jac": lambda w: -np.ones(n),
        }]

        result = minimize(
            cost_and_grad,
            x0=x0,
            jac=True,
            method="SLSQP",
            bounds=[(0.0, None)] * n,
            constraints=constraints,
            options={"maxiter": 2000, "ftol": self._ftol},
        )
        return {"probs": list(result.x), "fun": result.fun, "success": result.success}
