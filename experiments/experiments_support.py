"""Shared support functions for deconvolution experiments."""

import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from wnetdeconv import Spectrum_1D


def generate_random_spectra():
    """Generate random overlapping spectra."""
    n_peaks_E = 15
    n_peaks_T1 = 10
    n_peaks_T2 = 10

    pos_E = np.sort(np.random.uniform(1, 100, n_peaks_E))
    pos_T1 = np.sort(np.random.uniform(1, 80, n_peaks_T1))
    pos_T2 = np.sort(np.random.uniform(20, 100, n_peaks_T2))

    int_E = np.random.uniform(5, 50, n_peaks_E)
    int_T1 = np.random.uniform(2, 10, n_peaks_T1)
    int_T2 = np.random.uniform(2, 10, n_peaks_T2)

    E = Spectrum_1D(pos_E.tolist(), int_E.tolist())
    T1 = Spectrum_1D(pos_T1.tolist(), int_T1.tolist())
    T2 = Spectrum_1D(pos_T2.tolist(), int_T2.tolist())

    print(f"Generated E with {n_peaks_E} peaks")
    print(
        f"Generated T1 with {n_peaks_T1} peaks, range [{pos_T1[0]:.1f}, {pos_T1[-1]:.1f}]"
    )
    print(
        f"Generated T2 with {n_peaks_T2} peaks, range [{pos_T2[0]:.1f}, {pos_T2[-1]:.1f}]"
    )

    return E, T1, T2


def compute_cost_grid(solver, p1_range, p2_range, n_points):
    """Compute cost values and analytical gradients on a grid."""
    p1 = np.linspace(p1_range[0], p1_range[1], n_points)
    p2 = np.linspace(p2_range[0], p2_range[1], n_points)
    P1, P2 = np.meshgrid(p1, p2)
    C = np.empty_like(P1)
    Grad_p1_analytical = np.empty_like(P1)
    Grad_p2_analytical = np.empty_like(P1)

    for i in tqdm(range(n_points), desc="    Computing grid", leave=False):
        for j in range(n_points):
            solver.set_point([P1[i, j], P2[i, j]])
            C[i, j] = solver.total_cost()
            grad = solver.gradient()
            Grad_p1_analytical[i, j] = grad[0]
            Grad_p2_analytical[i, j] = grad[1]

    return P1, P2, C, p1, p2, Grad_p1_analytical, Grad_p2_analytical


def run_optimization(
    solver, start_point, bounds, method="L-BFGS-B", use_numerical_grad=False
):
    """Run optimization and return trajectory.

    Args:
        solver: DeconvSolver instance
        start_point: Starting point for optimization
        bounds: Bounds for each dimension
        method: Scipy optimization method name
        use_numerical_grad: If True, use numerical gradients; otherwise analytical
                           (ignored for gradient-free methods)
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
    options = {"disp": False, "maxiter": 200}

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


def find_global_optimum(solver, bounds, n_starts=20):
    """Find global optimum using multiple random starts."""
    best_result = None

    for i in tqdm(range(n_starts), desc="    Global search starts", leave=False):
        random_start = np.array(
            [
                np.random.uniform(bounds[0][0], bounds[0][1]),
                np.random.uniform(bounds[1][0], bounds[1][1]),
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
