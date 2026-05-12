from collections import namedtuple
from collections.abc import Sequence
from typing import Callable, Optional, Union, List, Tuple
import numpy as np
from scipy.optimize import minimize, OptimizeResult

from wnet import Distribution, WassersteinNetwork
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
        Scaling factor for intensities and costs. If None, it is computed automatically.
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

            # Bound the maximum cost per unit of flow in the scaled integer network.
            # Matching edges pay distance * sf per unit of flow (flow scaled by sf);
            # trash edges pay trash_cost * sf per unit of flow.  Total cost is at most
            # max_cost_per_unit_flow * max_sum_intensity * sf^2, which must stay < 2^60.
            max_cost_per_unit_flow = max([max_distance] + active_costs)
            scale_factor = np.sqrt(
                ALMOST_MAXINT / (max_sum_intensity * max_cost_per_unit_flow)
            )
            assert (
                scale_factor > 0
            ), "Can't auto-compute a sensible scale factor. You might have some luck with setting it manually, but it probably means something about your data or trash_cost is off."

        self.scale_factor = scale_factor
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
        return self.graph.total_cost() / self.scale_factor / self.scale_factor

    def print(self) -> None:
        """
        Prints a string representation of the graph associated with this aligner instance.

        Returns:
            None
        """
        print(str(self.graph))

    def flows(self) -> list[namedtuple]:
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
                namedtuple(
                    "Flow", ["empirical_peak_idx", "theoretical_peak_idx", "flow"]
                )(empirical_peak_idx, theoretical_peak_idx, flow / self.scale_factor)
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
        derivs = self.graph.spectrum_proportion_derivatives()
        result = np.array(
            [derivs.get(i, 0) for i in range(len(self.theoretical_spectra))]
        )
        return result / self.scale_factor / self.scale_factor

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
            options={"maxiter": 2000, "ftol": 1e-14},
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
            )
