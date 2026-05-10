"""Shared support functions for deconvolution experiments."""
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from wnetdeconv import Spectrum_1D

# Module-level solver used by worker functions so that forked child processes
# inherit it without pickling.
_solver = None


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
    print(f"Generated T1 with {n_peaks_T1} peaks, range [{pos_T1[0]:.1f}, {pos_T1[-1]:.1f}]")
    print(f"Generated T2 with {n_peaks_T2} peaks, range [{pos_T2[0]:.1f}, {pos_T2[-1]:.1f}]")

    return E, T1, T2


def _compute_row(row_i, p1_vals, p2_vals):
    """Worker function: compute one row of the cost/gradient grid.

    Reads the solver from the module-level ``_solver`` global, which is
    inherited from the parent process via fork — no pickling required.

    Args:
        row_i:   Row index (returned unchanged so callers can re-order results).
        p1_vals: 1-D array of p1 values for this row (length n_points).
        p2_vals: 1-D array of p2 values for this row (length n_points).

    Returns:
        (row_i, C_row, grad_p1_row, grad_p2_row) — all 1-D arrays of length n_points.
    """
    n = len(p1_vals)
    C_row = np.empty(n)
    grad_p1_row = np.empty(n)
    grad_p2_row = np.empty(n)
    for j in range(n):
        _solver.set_point([p1_vals[j], p2_vals[j]])
        C_row[j] = _solver.total_cost()
        grad = _solver.gradient()
        grad_p1_row[j] = grad[0]
        grad_p2_row[j] = grad[1]
    return row_i, C_row, grad_p1_row, grad_p2_row


def _compute_tagged_row(tag, row_i, p1_vals, p2_vals):
    """Like ``_compute_row`` but prefixes the result with an opaque *tag*.

    Used when dispatching rows from many different grids in a single
    ``pool.starmap`` call so that results can be grouped by grid identity
    after collection.

    Args:
        tag:     Arbitrary hashable identifier (e.g. ``(method_name, zoom_idx)``).
        row_i:   Row index passed through unchanged.
        p1_vals: 1-D array of p1 values for this row.
        p2_vals: 1-D array of p2 values for this row.

    Returns:
        ``(tag, row_i, C_row, grad_p1_row, grad_p2_row)``
    """
    _, C_row, grad_p1_row, grad_p2_row = _compute_row(row_i, p1_vals, p2_vals)
    return tag, row_i, C_row, grad_p1_row, grad_p2_row


def build_grid_tasks(p1_range, p2_range, n_points):
    """Return the list of per-row task arguments for parallel grid computation.

    Each element is ``(row_i, p1_vals, p2_vals)`` ready to pass to
    ``_compute_row`` (or ``pool.starmap(_compute_row, tasks)``).

    Args:
        p1_range: (min, max) tuple for the p1 axis.
        p2_range: (min, max) tuple for the p2 axis.
        n_points: Number of points along each axis.

    Returns:
        tasks: list of (row_i, p1_vals, p2_vals) tuples — one per row.
        p1:    1-D array of p1 coordinates (length n_points).
        p2:    1-D array of p2 coordinates (length n_points).
    """
    p1 = np.linspace(p1_range[0], p1_range[1], n_points)
    p2 = np.linspace(p2_range[0], p2_range[1], n_points)
    P1, P2 = np.meshgrid(p1, p2)
    tasks = [(i, P1[i, :], P2[i, :]) for i in range(n_points)]
    return tasks, p1, p2


def assemble_grid(n_points, p1, p2, raw_results):
    """Assemble per-row worker results into full 2-D grid arrays.

    Args:
        n_points:    Number of grid points per axis.
        p1:          1-D array of p1 coordinates (from ``build_grid_tasks``).
        p2:          1-D array of p2 coordinates (from ``build_grid_tasks``).
        raw_results: Iterable of ``(row_i, C_row, grad_p1_row, grad_p2_row)``
                     as returned by ``_compute_row``.  May arrive in any order.

    Returns:
        (P1, P2, C, p1, p2, Grad_p1, Grad_p2) — same layout as the old
        ``compute_cost_grid`` return value.
    """
    P1, P2 = np.meshgrid(p1, p2)
    C = np.empty((n_points, n_points))
    Grad_p1 = np.empty((n_points, n_points))
    Grad_p2 = np.empty((n_points, n_points))
    for row_i, C_row, grad_p1_row, grad_p2_row in raw_results:
        C[row_i, :] = C_row
        Grad_p1[row_i, :] = grad_p1_row
        Grad_p2[row_i, :] = grad_p2_row
    return P1, P2, C, p1, p2, Grad_p1, Grad_p2


def compute_cost_grid(solver, p1_range, p2_range, n_points):
    """Compute cost values and analytical gradients on a grid (serial).

    Existing call sites continue to work unchanged.  Internally delegates to
    ``build_grid_tasks`` / ``_compute_row`` / ``assemble_grid`` so the logic
    lives in one place.
    """
    global _solver
    _solver = solver
    tasks, p1, p2 = build_grid_tasks(p1_range, p2_range, n_points)
    raw = [_compute_row(*t) for t in tqdm(tasks, desc="    Computing grid", leave=False)]
    return assemble_grid(n_points, p1, p2, raw)


def run_optimization(solver, start_point, bounds, method='L-BFGS-B', use_numerical_grad=False):
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
    options = {'disp': False, 'maxiter': 200}

    # Gradient-free methods
    gradient_free_methods = ['Nelder-Mead', 'Powell', 'COBYLA']

    if method in gradient_free_methods:
        # Gradient-free methods don't use gradients
        result = minimize(
            cost_function,
            start_point,
            method=method,
            bounds=bounds,
            options=options
        )
    elif use_numerical_grad:
        # Use numerical gradients (automatic finite differences)
        # Use a larger epsilon for finite differences since cost function is discrete
        options['eps'] = 1e-2
        result = minimize(
            cost_function,
            start_point,
            method=method,
            bounds=bounds,
            options=options
        )
    else:
        # Use analytical gradients
        result = minimize(
            cost_function,
            start_point,
            method=method,
            jac=grad_function,
            bounds=bounds,
            options=options
        )

    return result, np.array(trajectory)


def find_global_optimum(solver, bounds, n_starts=20):
    """Find global optimum using multiple random starts (serial).

    Works with N-dimensional bounds.
    Existing call sites continue to work unchanged; internally delegates to
    ``build_global_search_tasks`` / ``_run_global_start`` / ``reduce_global_results``.
    """
    global _solver
    _solver = solver
    tasks = build_global_search_tasks(bounds, n_starts)
    raw = [_run_global_start(*t) for t in tqdm(tasks, desc="    Global search starts", leave=False)]
    return reduce_global_results(raw)


# ---------------------------------------------------------------------------
# Task-list builders and worker functions for parallel global search
# ---------------------------------------------------------------------------

def build_global_search_tasks(bounds, n_starts):
    """Return a list of random start points for the global optimum search.

    Sampling is done here in the parent process so that the RNG state is
    deterministic (respects any seed set before calling this function) and
    independent of worker scheduling order.

    Args:
        bounds:   List of (min, max) tuples, one per dimension.
        n_starts: Number of random starting points.

    Returns:
        tasks: list of ``(start_point, bounds)`` tuples, one per start.
    """
    n_dims = len(bounds)
    tasks = []
    for _ in range(n_starts):
        start = np.array([np.random.uniform(bounds[j][0], bounds[j][1])
                          for j in range(n_dims)])
        tasks.append((start, bounds))
    return tasks


def _run_global_start(start_point, bounds):
    """Worker: run one L-BFGS-B minimisation from *start_point*.

    Uses the module-level ``_solver`` inherited via fork.

    Returns:
        scipy OptimizeResult on success, or ``None`` if an exception is raised.
    """
    def cost(point):
        _solver.set_point(point)
        return _solver.total_cost()

    def grad(point):
        _solver.set_point(point)
        return np.array(_solver.gradient())

    try:
        return minimize(cost, start_point, method='L-BFGS-B',
                        jac=grad, bounds=bounds,
                        options={'disp': False, 'maxiter': 100})
    except Exception:
        return None


def reduce_global_results(results):
    """Pick the best (lowest cost) result from a list of OptimizeResults.

    ``None`` entries (failed starts) are silently skipped.

    Returns the best OptimizeResult, or ``None`` if all starts failed.
    """
    best = None
    for r in results:
        if r is not None and (best is None or r.fun < best.fun):
            best = r
    return best


# ---------------------------------------------------------------------------
# Task-list builders and worker functions for parallel per-method optimisation
# ---------------------------------------------------------------------------

def build_optim_tasks(methods, gradient_free_methods, start_point, bounds):
    """Return a flat list of optimisation tasks for all methods.

    For gradient-free methods only one task is generated (no num/ana split).
    For gradient-based methods two tasks are generated: one with numerical
    gradients and one with analytical gradients.

    Args:
        methods:               List of ``(method_name, display_name)`` pairs.
        gradient_free_methods: Collection of method names that are gradient-free.
        start_point:           Starting point array.
        bounds:                List of ``(min, max)`` bound tuples.

    Returns:
        tasks: list of ``(method_name, use_numerical_grad, start_point, bounds)``
               tuples.  The ``use_numerical_grad`` flag is ``None`` for
               gradient-free methods (signals that gradients are not used).
    """
    tasks = []
    for method_name, _ in methods:
        if method_name in gradient_free_methods:
            tasks.append((method_name, None, start_point, bounds))
        else:
            tasks.append((method_name, True,  start_point, bounds))
            tasks.append((method_name, False, start_point, bounds))
    return tasks


def _run_optim_task(method_name, use_numerical_grad, start_point, bounds):
    """Worker: run one optimisation task using the module-level ``_solver``.

    Args:
        method_name:        Scipy method string.
        use_numerical_grad: ``True`` → numerical, ``False`` → analytical,
                            ``None`` → gradient-free.
        start_point:        Starting point array.
        bounds:             List of ``(min, max)`` bound tuples.

    Returns:
        ``(method_name, use_numerical_grad, result, trajectory)``
    """
    result, trajectory = run_optimization(
        _solver, start_point, bounds,
        method=method_name,
        use_numerical_grad=bool(use_numerical_grad),  # False for None (gradient-free)
    )
    return method_name, use_numerical_grad, result, trajectory
