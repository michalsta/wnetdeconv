"""Shared support functions for deconvolution experiments."""

import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from wnetdeconv import Spectrum_1D


def generate_random_spectra(
    n_peaks_E=15,
    n_theoretical=2,
    n_peaks_theoretical=10,
    theoretical_peak_counts=None,
):
    """Generate random overlapping spectra for one empirical and N theoretical spectra.

    Args:
        n_peaks_E: Number of peaks in empirical spectrum.
        n_theoretical: Number of theoretical spectra to generate.
        n_peaks_theoretical: Default number of peaks per theoretical spectrum.
        theoretical_peak_counts: Optional explicit per-spectrum peak counts.
    """
    if n_theoretical < 1:
        raise ValueError("n_theoretical must be >= 1")

    if theoretical_peak_counts is None:
        theoretical_peak_counts = [n_peaks_theoretical] * n_theoretical
    elif len(theoretical_peak_counts) != n_theoretical:
        raise ValueError(
            "theoretical_peak_counts length must match n_theoretical "
            f"({len(theoretical_peak_counts)} != {n_theoretical})"
        )

    pos_E = np.sort(np.random.uniform(1, 100, n_peaks_E))
    int_E = np.random.uniform(5, 50, n_peaks_E)
    E = Spectrum_1D(pos_E.tolist(), int_E.tolist())

    theoretical_spectra = []
    centers = np.linspace(20, 80, n_theoretical)
    window_half_width = 35

    for idx, (center, n_peaks_t) in enumerate(zip(centers, theoretical_peak_counts)):
        low = max(1, center - window_half_width)
        high = min(100, center + window_half_width)

        pos_T = np.sort(np.random.uniform(low, high, n_peaks_t))
        int_T = np.random.uniform(2, 10, n_peaks_t)
        theoretical_spectra.append(Spectrum_1D(pos_T.tolist(), int_T.tolist()))

        print(
            f"Generated T{idx + 1} with {n_peaks_t} peaks, "
            f"range [{pos_T[0]:.1f}, {pos_T[-1]:.1f}]"
        )

    print(f"Generated E with {n_peaks_E} peaks")
    return E, theoretical_spectra


def compute_cost_grid(solver, p1_range, p2_range, n_points, verbose=True):
    """Compute cost values and analytical gradients on a grid.

    Args:
        verbose: If False, suppress tqdm progress bar (useful in worker processes)
    """
    p1 = np.linspace(p1_range[0], p1_range[1], n_points)
    p2 = np.linspace(p2_range[0], p2_range[1], n_points)
    P1, P2 = np.meshgrid(p1, p2)
    C = np.empty_like(P1)
    Grad_p1_analytical = np.empty_like(P1)
    Grad_p2_analytical = np.empty_like(P1)

    iterator = range(n_points)
    if verbose:
        iterator = tqdm(iterator, desc="    Computing grid", leave=False)

    for i in iterator:
        for j in range(n_points):
            solver.set_point([P1[i, j], P2[i, j]])
            C[i, j] = solver.total_cost()
            grad = solver.gradient()
            Grad_p1_analytical[i, j] = grad[0]
            Grad_p2_analytical[i, j] = grad[1]

    return P1, P2, C, p1, p2, Grad_p1_analytical, Grad_p2_analytical


def compute_cost_grid_axis_pair_slice(
    solver,
    center_point,
    dim1,
    dim2,
    dim1_range,
    dim2_range,
    n_points,
    verbose=True,
):
    """Compute a 2D cost/gradient grid for an axis-aligned slice through an N-D point.

    The slice varies dimensions ``dim1`` and ``dim2`` while all other coordinates are
    fixed at ``center_point``.
    """
    center_point = np.asarray(center_point, dtype=float)
    p1 = np.linspace(dim1_range[0], dim1_range[1], n_points)
    p2 = np.linspace(dim2_range[0], dim2_range[1], n_points)
    P1, P2 = np.meshgrid(p1, p2)
    C = np.empty_like(P1)
    Grad_dim1_analytical = np.empty_like(P1)
    Grad_dim2_analytical = np.empty_like(P1)

    iterator = range(n_points)
    if verbose:
        iterator = tqdm(iterator, desc="    Computing axis-pair slice", leave=False)

    for i in iterator:
        for j in range(n_points):
            point = center_point.copy()
            point[dim1] = P1[i, j]
            point[dim2] = P2[i, j]
            solver.set_point(point)
            C[i, j] = solver.total_cost()
            grad = solver.gradient()
            Grad_dim1_analytical[i, j] = grad[dim1]
            Grad_dim2_analytical[i, j] = grad[dim2]

    return P1, P2, C, p1, p2, Grad_dim1_analytical, Grad_dim2_analytical


def run_optimization(
    solver,
    start_point,
    bounds,
    method="L-BFGS-B",
    use_numerical_grad=False,
    max_iterations=200,
):
    """Run optimization and return trajectory.

    Args:
        solver: DeconvSolver instance
        start_point: Starting point for optimization
        bounds: Bounds for each dimension
        method: Scipy optimization method name
        use_numerical_grad: If True, use numerical gradients; otherwise analytical
                   (ignored for gradient-free methods)
        max_iterations: Maximum number of optimization iterations/function calls.
    """
    trajectory = []

    def cost_function(point):
        solver.set_point(point)
        cost = solver.total_cost()
        trajectory.append(point.copy())
        return cost

    def grad_function(point):
        solver.set_point(point)
        return np.array(solver.gradient())

    # Configure options based on method
    # Different methods use different option names for iteration limits
    options = {"disp": False}

    # Set max iterations using the correct option name for each method
    if method == "TNC":
        options["maxfun"] = max_iterations  # TNC uses maxfun, not maxiter
    else:
        options["maxiter"] = max_iterations  # Most methods use maxiter

    # Gradient-free methods
    gradient_free_methods = ["Nelder-Mead", "Powell", "COBYLA"]

    if method in gradient_free_methods:
        # Gradient-free methods don't use gradients
        result = minimize(
            cost_function, start_point, method=method, bounds=bounds, options=options
        )
    elif use_numerical_grad:
        # Use numerical gradients (automatic finite differences)
        # Use a larger epsilon for finite differences since cost function is discrete
        options["eps"] = 1e-2
        result = minimize(
            cost_function, start_point, method=method, bounds=bounds, options=options
        )
    else:
        # Use analytical gradients
        result = minimize(
            cost_function,
            start_point,
            method=method,
            jac=grad_function,
            bounds=bounds,
            options=options,
        )

    return result, np.array(trajectory)


def find_global_optimum(solver, bounds, n_starts=20, verbose=True):
    """Find global optimum using multiple random starts.

    Args:
        verbose: If False, suppress tqdm progress bar (useful in worker processes)
    """
    best_result = None

    iterator = range(n_starts)
    if verbose:
        iterator = tqdm(iterator, desc="    Global search starts", leave=False)

    for i in iterator:
        random_start = np.array(
            [
                np.random.uniform(dim_bounds[0], dim_bounds[1])
                for dim_bounds in bounds
            ]
        )

        def temp_cost(point):
            solver.set_point(point)
            return solver.total_cost()

        def temp_grad(point):
            solver.set_point(point)
            return np.array(solver.gradient())

        try:
            temp_result = minimize(
                temp_cost,
                random_start,
                method="L-BFGS-B",
                jac=temp_grad,
                bounds=bounds,
                options={"disp": False, "maxiter": 100},
            )
            if best_result is None or temp_result.fun < best_result.fun:
                best_result = temp_result
        except:
            pass

    return best_result
