"""2D map of (p1, p2) -> total_cost for the two-spectrum deconvolution example."""

import argparse
import multiprocessing
from types import SimpleNamespace
import numpy as np
from tqdm import tqdm
from wnetdeconv import DeconvSolver
from wnet.distances import DistanceMetric
from experiments_support import (
    generate_random_spectra,
    load_hemoglobin_benchmark_spectra,
    compute_cost_grid,
    compute_cost_grid_axis_pair_slice,
    run_optimization,
    find_global_optimum,
)
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Force fork method for multiprocessing (required for global variable inheritance)
multiprocessing.set_start_method("fork", force=True)

# ============================================================================
# GLOBAL VARIABLES FOR PARALLEL PROCESSING (inherited via fork)
# ============================================================================

# Dictionaries indexed by run_num for multi-run parallel processing
_solvers = {}  # {run_num: solver}
_start_points = {}  # {run_num: start_point}
_bounds_dict = {}  # {run_num: bounds}
_p1_ranges = {}  # {run_num: (p1_min, p1_max)}
_p2_ranges = {}  # {run_num: (p2_min, p2_max)}
_global_optima = {}  # {run_num: result}

# ============================================================================
# ARGUMENT PARSING
# ============================================================================


def _build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Control flags
    parser.add_argument(
        "--no-plot", action="store_true", help="Disable all plot generation"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        metavar="N",
        help="Number of runs with different random spectra",
    )

    # Optimization settings
    parser.add_argument(
        "--start-point",
        nargs="+",
        type=float,
        default=[15.0],
        metavar="P",
        help="Starting point values: provide 1 value (broadcast to all dims) or N values",
    )
    parser.add_argument(
        "--bounds",
        nargs="+",
        type=float,
        default=None,
        metavar="B",
        help="Bounds values: provide 2 values (min max, broadcast to all dims) or 2*N values. "
        "Default: [0, 1] with --hemoglobin, [0, 25] otherwise",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=200,
        metavar="N",
        help="Maximum iterations for optimization",
    )
    parser.add_argument(
        "--global-search-starts",
        type=int,
        default=20,
        metavar="N",
        help="Number of random starts for global optimum search",
    )

    # Grid computation settings
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=200,
        metavar="N",
        help="Number of points per axis for cost grid",
    )
    parser.add_argument(
        "--zoom-levels",
        nargs="+",
        type=int,
        default=[10, 100, 1000, 10000, 100000],
        metavar="Z",
        help="Zoom levels to compute",
    )

    # Plotting settings (only used without --no-plot)
    parser.add_argument(
        "--figure-size",
        nargs=2,
        type=float,
        default=[30.0, 30.0],
        metavar=("W", "H"),
        help="Figure size in inches",
    )
    parser.add_argument(
        "--dpi", type=int, default=150, help="Resolution for saved plots"
    )
    parser.add_argument(
        "--arrow-subsample",
        type=int,
        default=20,
        metavar="N",
        help="Show every Nth arrow in gradient field",
    )
    parser.add_argument(
        "--arrow-alpha", type=float, default=0.6, help="Transparency of gradient arrows"
    )

    # Deconvolution solver parameters
    parser.add_argument(
        "--distance-metric",
        choices=["L1", "L2", "LINF"],
        default="L2",
        help="Distance metric for the deconvolution solver",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=10,
        help="Maximum distance for the deconvolution solver",
    )
    parser.add_argument(
        "--trash-cost",
        type=float,
        default=100,
        help="Trash cost for the deconvolution solver",
    )
    parser.add_argument(
        "--auto-scale",
        action="store_true",
        default=True,
        help="Enable automatic scale factor computation based on data characteristics (default: True)",
    )
    parser.add_argument(
        "--no-auto-scale",
        action="store_false",
        dest="auto_scale",
        help="Disable automatic scale factor computation and use manual value instead",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=None,
        metavar="SF",
        help="Manual scale factor for numerical stability (only used if --no-auto-scale is set). "
        "Default is auto-computed from max_sum_intensity and trash_cost using formula: "
        "sqrt(2^60 / (max_intensity * trash_cost))",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel worker processes (default: number of CPUs)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Random seed for reproducibility (default: None, use random seed)",
    )
    parser.add_argument(
        "--hemoglobin",
        action="store_true",
        help="Use masserstein hemoglobin benchmark spectra and solver parameters instead of random spectra",
    )
    parser.add_argument(
        "--hemoglobin-iso-coverage",
        type=float,
        default=0.99,
        metavar="P",
        help="Total isotope probability coverage used for hemoglobin theoretical spectra",
    )
    parser.add_argument(
        "--n-peaks-empirical",
        type=int,
        default=15,
        metavar="N",
        help="Number of peaks in randomly generated empirical spectrum (default: 15)",
    )
    parser.add_argument(
        "--n-theoretical",
        type=int,
        default=2,
        metavar="N",
        help="Number of randomly generated theoretical spectra",
    )
    parser.add_argument(
        "--n-peaks-theoretical",
        type=int,
        default=10,
        metavar="N",
        help="Default number of peaks in each randomly generated theoretical spectrum",
    )
    parser.add_argument(
        "--n-peaks-theoretical-1",
        type=int,
        default=None,
        metavar="N",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--n-peaks-theoretical-2",
        type=int,
        default=None,
        metavar="N",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        metavar="DIR",
        help="Output directory for saved plots (default: current directory)",
    )

    return parser


_args = _build_parser().parse_args()


def _expand_start_point(start_values, n_dims):
    if len(start_values) == 1:
        return [start_values[0]] * n_dims
    if len(start_values) == n_dims:
        return list(start_values)
    raise ValueError(
        "--start-point must have either 1 value or N values "
        f"(got {len(start_values)}, N={n_dims})"
    )


def _expand_bounds(bounds_values, n_dims):
    if len(bounds_values) == 2:
        b_min, b_max = bounds_values
        if b_min >= b_max:
            raise ValueError("Broadcast bounds require min < max")
        return [(b_min, b_max)] * n_dims

    if len(bounds_values) == 2 * n_dims:
        bounds = []
        for i in range(n_dims):
            b_min = bounds_values[2 * i]
            b_max = bounds_values[2 * i + 1]
            if b_min >= b_max:
                raise ValueError(f"Invalid bounds for dim {i}: min must be < max")
            bounds.append((b_min, b_max))
        return bounds

    raise ValueError(
        "--bounds must have either 2 values or 2*N values "
        f"(got {len(bounds_values)}, N={n_dims})"
    )


def _get_axis_pairs(n_dims):
    """Return all axis-aligned dimension pairs (i, j) with i < j."""
    return [(i, j) for i in range(n_dims) for j in range(i + 1, n_dims)]


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Control flags
ENABLE_PLOTTING = not _args.no_plot
NUM_RUNS = _args.num_runs
USE_HEMOGLOBIN = _args.hemoglobin
HEMOGLOBIN_ISO_COVERAGE = _args.hemoglobin_iso_coverage

# Solver parameter defaults can be overridden by benchmark modes.
distance_metric_name = _args.distance_metric
max_distance_value = _args.max_distance
trash_cost_value = _args.trash_cost
enable_auto_scale_value = _args.auto_scale
scale_factor_value = None if enable_auto_scale_value else _args.scale_factor

if USE_HEMOGLOBIN:
    # Match masserstein hemoglobin benchmark defaults.
    distance_metric_name = "LINF"
    max_distance_value = 0.025
    trash_cost_value = 0.1
    enable_auto_scale_value = False
    scale_factor_value = 1e4
    # Use hemoglobin default bounds if user didn't explicitly set them
    if _args.bounds is None:
        _args.bounds = [0, 1]

# Set default bounds if not specified
if _args.bounds is None:
    _args.bounds = [0, 25]

# Determine number of dimensions.
if USE_HEMOGLOBIN:
    # Hemoglobin benchmark uses 10 theoretical spectra (fixed charge-state set).
    N_THEORETICAL = 10
else:
    N_THEORETICAL = _args.n_theoretical

AXIS_PAIRS = _get_axis_pairs(N_THEORETICAL)

# Peak count configuration (supports deprecated two-spectrum flags)
N_PEAKS_THEORETICAL = _args.n_peaks_theoretical
if _args.n_peaks_theoretical_1 is not None or _args.n_peaks_theoretical_2 is not None:
    if N_THEORETICAL != 2:
        raise ValueError(
            "Deprecated --n-peaks-theoretical-1/2 flags can only be used with --n-theoretical 2"
        )
    THEORETICAL_PEAK_COUNTS = [
        (
            _args.n_peaks_theoretical_1
            if _args.n_peaks_theoretical_1 is not None
            else N_PEAKS_THEORETICAL
        ),
        (
            _args.n_peaks_theoretical_2
            if _args.n_peaks_theoretical_2 is not None
            else N_PEAKS_THEORETICAL
        ),
    ]
else:
    THEORETICAL_PEAK_COUNTS = [N_PEAKS_THEORETICAL] * N_THEORETICAL

# Optimization settings
START_POINT = _expand_start_point(_args.start_point, N_THEORETICAL)
BOUNDS = _expand_bounds(_args.bounds, N_THEORETICAL)
MAX_ITERATIONS = _args.max_iterations
GLOBAL_SEARCH_STARTS = _args.global_search_starts

# Grid computation settings
GRID_RESOLUTION = _args.grid_resolution
ZOOM_LEVELS = _args.zoom_levels

# Plotting settings (only used if ENABLE_PLOTTING=True)
FIGURE_SIZE = tuple(_args.figure_size)
DPI = _args.dpi
ARROW_SUBSAMPLE = _args.arrow_subsample
ARROW_ALPHA = _args.arrow_alpha

# Optimization methods to test
# Each entry: (method_name, display_name)
METHODS = [
    ("L-BFGS-B", "Limited-memory BFGS with Bounds"),
    ("TNC", "Truncated Newton Conjugate-Gradient"),
    ("SLSQP", "Sequential Least Squares Quadratic Programming"),
    ("Nelder-Mead", "Nelder-Mead Simplex"),
    ("Powell", "Powell Direction Set"),
    ("COBYLA", "Constrained Optimization BY Linear Approximation"),
]

# Gradient-free methods (subset of METHODS)
GRADIENT_FREE_METHODS = ["Nelder-Mead", "Powell", "COBYLA"]

# Deconvolution solver parameters
DISTANCE_METRIC = DistanceMetric[distance_metric_name]
MAX_DISTANCE = max_distance_value
TRASH_COST = trash_cost_value
ENABLE_AUTO_SCALE = enable_auto_scale_value
SCALE_FACTOR = scale_factor_value

# Random seed
RANDOM_SEED = _args.random_seed

# Spectrum generation parameters
N_PEAKS_EMPIRICAL = _args.n_peaks_empirical

# Output directory
OUT_DIR = _args.out_dir

# ============================================================================
# PARALLEL WORKER FUNCTIONS (access globals via fork inheritance)
# ============================================================================


def _run_optimization_task(args):
    """Worker function to run optimization for a method.

    Args:
        args: Tuple of (run_num, method_name, display_name, use_numerical_grad)

    Returns:
        Tuple of (run_num, method_name, use_numerical_grad, result, trajectory)
    """
    run_num, method_name, display_name, use_numerical_grad = args

    solver = _solvers[run_num]
    start_point = _start_points[run_num]
    bounds = _bounds_dict[run_num]

    result, trajectory = run_optimization(
        solver,
        start_point,
        bounds,
        method=method_name,
        use_numerical_grad=use_numerical_grad,
        max_iterations=MAX_ITERATIONS,
    )
    return (run_num, method_name, use_numerical_grad, result, trajectory)


def _format_point(point, precision=2):
    """Format an N-dimensional point for logs."""
    coords = ", ".join(f"p{i + 1}={v:.{precision}f}" for i, v in enumerate(point))
    return f"({coords})"


def _all_zoom_levels_with_full(zoom_levels):
    """Return sorted zoom list including full-view zoom=1."""
    return sorted(set([1] + list(zoom_levels)))


def _pair_labels(dim_pair):
    """Return (x_label, y_label, pair_tag) for a dimension pair."""
    dim1, dim2 = dim_pair
    x_label = f"p{dim1 + 1}"
    y_label = f"p{dim2 + 1}"
    pair_tag = f"{x_label}-{y_label}"
    return x_label, y_label, pair_tag


def _project_point_to_axis_pair_plane(point, plane_center, dim_pair):
    """Orthogonally project an N-D point onto an axis-pair slice plane.

    The target plane is defined by varying dimensions in ``dim_pair`` and fixing all
    other coordinates to ``plane_center``.
    """
    point = np.asarray(point, dtype=float)
    plane_center = np.asarray(plane_center, dtype=float)
    dim1, dim2 = dim_pair

    projected_nd = plane_center.copy()
    projected_nd[dim1] = point[dim1]
    projected_nd[dim2] = point[dim2]

    orth_distance = np.linalg.norm(point - projected_nd)
    projected_2d = np.array([projected_nd[dim1], projected_nd[dim2]], dtype=float)
    return projected_2d, float(orth_distance)


def _project_trajectory_to_axis_pair_plane(trajectory, plane_center, dim_pair):
    """Project an N-D trajectory onto an axis-pair slice plane."""
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.size == 0:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)

    projected_points = []
    orth_distances = []
    for point in trajectory:
        p2, d = _project_point_to_axis_pair_plane(point, plane_center, dim_pair)
        projected_points.append(p2)
        orth_distances.append(d)

    return np.array(projected_points), np.array(orth_distances)


def _project_result_to_axis_pair_plane(result, plane_center, dim_pair):
    """Create a lightweight result-like object with projected 2D coordinates."""
    p2, d = _project_point_to_axis_pair_plane(result.x, plane_center, dim_pair)
    return SimpleNamespace(
        x=p2,
        fun=result.fun,
        success=getattr(result, "success", None),
        orth_distance=d,
    )


def _distance_to_alpha(distances, min_alpha=0.2):
    """Map orthogonal projection distances to alpha values in [min_alpha, 1]."""
    distances = np.asarray(distances, dtype=float)
    if distances.size == 0:
        return distances
    if np.allclose(distances, 0):
        return np.ones_like(distances)

    # Robust scale to avoid single outliers flattening the fade mapping.
    scale = np.percentile(distances, 90)
    if scale <= 0:
        return np.ones_like(distances)

    normalized = np.clip(distances / scale, 0, 1)
    return 1.0 - (1.0 - min_alpha) * normalized


def _compute_zoom_grid_task(args):
    """Worker function to compute a zoom grid.

    Args:
        args: Tuple of (run_num, method_name, dim_pair, zoom_level, center_point)

    Returns:
        Tuple of (run_num, method_name, dim_pair, zoom_level, grid_data_dict)
    """
    run_num, method_name, dim_pair, zoom, center_point = args
    dim1, dim2 = dim_pair

    solver = _solvers[run_num]
    bounds = _bounds_dict[run_num]
    center_point = np.array(center_point, dtype=float)

    def _compute_zoom_bounds(center, dim_bounds, zoom_value):
        min_bound, max_bound = dim_bounds
        width = (max_bound - min_bound) / zoom_value
        low = center - width / 2
        high = center + width / 2

        if low < min_bound:
            low = min_bound
            high = low + width
        elif high > max_bound:
            high = max_bound
            low = high - width
        return low, high

    p1_min, p1_max = _compute_zoom_bounds(center_point[dim1], bounds[dim1], zoom)
    p2_min, p2_max = _compute_zoom_bounds(center_point[dim2], bounds[dim2], zoom)

    # Compute grid (without verbose output in worker process)
    P1_z, P2_z, C_z, p1_z, p2_z, grad_p1_ana_z, grad_p2_ana_z = (
        compute_cost_grid_axis_pair_slice(
            solver,
            center_point,
            dim1,
            dim2,
            (p1_min, p1_max),
            (p2_min, p2_max),
            GRID_RESOLUTION,
            verbose=False,
        )
    )
    grad_p2_num_z, grad_p1_num_z = np.gradient(C_z, p2_z, p1_z)

    # Track grid minimum
    min_idx_zoom = np.unravel_index(np.argmin(C_z), C_z.shape)
    grid_min = (P1_z[min_idx_zoom], P2_z[min_idx_zoom], C_z[min_idx_zoom])

    grid_data = {
        "zoom": zoom,
        "dim_pair": dim_pair,
        "P1": P1_z,
        "P2": P2_z,
        "C": C_z,
        "grad_p1_num": grad_p1_num_z,
        "grad_p2_num": grad_p2_num_z,
        "grad_p1_ana": grad_p1_ana_z,
        "grad_p2_ana": grad_p2_ana_z,
        "bounds": (p1_min, p1_max, p2_min, p2_max),
        "grid_min": grid_min,
    }

    return (run_num, method_name, dim_pair, zoom, grid_data)


def plot_trajectory(ax, trajectory, zoom_bounds=None, point_alpha=None):
    """Plot optimization trajectory with black-to-white gradient."""
    point_alpha = (
        np.asarray(point_alpha, dtype=float)
        if point_alpha is not None
        else np.ones(len(trajectory))
    )

    if zoom_bounds:
        p1_min, p1_max, p2_min, p2_max = zoom_bounds
        mask = (
            (trajectory[:, 0] >= p1_min)
            & (trajectory[:, 0] <= p1_max)
            & (trajectory[:, 1] >= p2_min)
            & (trajectory[:, 1] <= p2_max)
        )
        if not np.any(mask):
            return
        traj = trajectory[mask]
        alpha = point_alpha[mask]
    else:
        traj = trajectory
        alpha = point_alpha

    if len(traj) < 2:
        return

    # Plot line segments
    points = traj.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = np.linspace(0, 1, len(segments))
    seg_alpha = 0.5 * (alpha[:-1] + alpha[1:])
    seg_colors = np.stack([colors, colors, colors, seg_alpha], axis=1)
    lc = LineCollection(segments, cmap="gray", linewidth=2, zorder=4, label="trace")
    lc.set_color(seg_colors)
    ax.add_collection(lc)

    # Plot points
    traj_colors = np.linspace(0, 1, len(traj))
    point_colors = np.stack([traj_colors, traj_colors, traj_colors, alpha], axis=1)
    ax.scatter(
        traj[:, 0],
        traj[:, 1],
        c=point_colors,
        s=15,
        zorder=4,
        edgecolors="black",
        linewidths=0.3,
    )

    # Mark start point
    ax.scatter(
        [traj[0, 0]],
        [traj[0, 1]],
        color="red",
        marker="o",
        s=40,
        zorder=5,
        edgecolors="white",
        linewidths=1.0,
        label="start",
    )


def add_optima_markers(ax, scipy_result, global_result, grid_optimum, precision=2):
    """Add three optima markers with costs in legend."""
    fmt = f"{{:.{precision}f}}"

    # Scipy endpoint
    scipy_label = f"scipy end ({fmt.format(scipy_result.x[0])}, {fmt.format(scipy_result.x[1])}, cost={scipy_result.fun:.0f})"
    ax.scatter(
        [scipy_result.x[0]],
        [scipy_result.x[1]],
        color="cyan",
        marker="X",
        s=50,
        zorder=7,
        edgecolors="black",
        linewidths=1.0,
        label=scipy_label,
    )

    # Global optimum
    global_label = f"global opt ({fmt.format(global_result.x[0])}, {fmt.format(global_result.x[1])}, cost={global_result.fun:.0f})"
    ax.scatter(
        [global_result.x[0]],
        [global_result.x[1]],
        color="magenta",
        marker="D",
        s=50,
        zorder=7,
        edgecolors="white",
        linewidths=1.0,
        label=global_label,
    )

    # Grid minimum
    grid_label = f"grid min ({fmt.format(grid_optimum[0])}, {fmt.format(grid_optimum[1])}, cost={grid_optimum[2]:.0f})"
    ax.scatter(
        [grid_optimum[0]],
        [grid_optimum[1]],
        color="yellow",
        marker="s",
        s=50,
        zorder=7,
        edgecolors="black",
        linewidths=1.0,
        label=grid_label,
    )


def plot_single_map(
    ax,
    P1,
    P2,
    data,
    trajectory,
    scipy_result,
    global_result,
    grid_optimum,
    title,
    dim_pair,
    cmap,
    label,
    zoom_bounds=None,
    trajectory_alpha=None,
    precision=2,
    vmin=None,
    vmax=None,
):
    """Plot a single map (cost or gradient magnitude) with trajectory."""
    if vmin is None or vmax is None:
        vmin, vmax = np.percentile(data, [5, 95])

    pcm = ax.pcolormesh(P1, P2, data, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(pcm, ax=ax, label=label)

    plot_trajectory(ax, trajectory, zoom_bounds, point_alpha=trajectory_alpha)
    add_optima_markers(ax, scipy_result, global_result, grid_optimum, precision)

    # Set axis limits if zoom bounds provided
    if zoom_bounds:
        ax.set_xlim(zoom_bounds[0], zoom_bounds[1])
        ax.set_ylim(zoom_bounds[2], zoom_bounds[3])

    x_label, y_label, _ = _pair_labels(dim_pair)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=6)


def plot_zoom_row(
    axes,
    data_dict,
    trajectory_num,
    trajectory_ana,
    trajectory_num_alpha,
    trajectory_ana_alpha,
    scipy_result_num,
    scipy_result_ana,
    global_result,
    grid_optimum,
    dim_pair,
    zoom_label,
    precision=2,
):
    """Plot one row showing: cost+numeric grad walk, numeric gradient, cost+analytic grad walk, analytic gradient, difference."""
    P1 = data_dict["P1"]
    P2 = data_dict["P2"]
    C = data_dict["C"]
    grad_p1_num = data_dict["grad_p1_num"]
    grad_p2_num = data_dict["grad_p2_num"]
    grad_p1_ana = data_dict["grad_p1_ana"]
    grad_p2_ana = data_dict["grad_p2_ana"]
    bounds = data_dict["bounds"]

    # Compute gradient magnitudes
    grad_norm_num = np.sqrt(grad_p1_num**2 + grad_p2_num**2)
    grad_norm_ana = np.sqrt(grad_p1_ana**2 + grad_p2_ana**2)
    grad_diff = np.abs(grad_norm_ana - grad_norm_num)

    # Use consistent color scale for gradients
    all_grad_data = np.concatenate([grad_norm_num.flatten(), grad_norm_ana.flatten()])
    gmin, gmax = np.percentile(all_grad_data, [5, 95])

    # Column 1: Cost landscape with numerical gradient walk
    plot_single_map(
        axes[0],
        P1,
        P2,
        C,
        trajectory_num,
        scipy_result_num,
        global_result,
        grid_optimum,
        f"Cost + Numerical Walk ({_pair_labels(dim_pair)[2]}, zoom {zoom_label})",
        dim_pair,
        "viridis",
        "cost",
        bounds,
        trajectory_num_alpha,
        precision,
    )

    # Column 2: Numerical gradient with arrows and numerical walk
    vmin, vmax = np.percentile(grad_norm_num, [5, 95])
    pcm = axes[1].pcolormesh(
        P1, P2, grad_norm_num, shading="auto", cmap="plasma", vmin=gmin, vmax=gmax
    )
    plt.colorbar(pcm, ax=axes[1], label="gradient magnitude")

    # Add gradient arrows (normalized to same length)
    step = max(1, len(P1) // ARROW_SUBSAMPLE)
    # Normalize gradients to unit vectors
    grad_mag = np.sqrt(
        grad_p1_num[::step, ::step] ** 2 + grad_p2_num[::step, ::step] ** 2
    )
    grad_mag = np.where(grad_mag == 0, 1, grad_mag)  # Avoid division by zero
    grad_p1_norm = grad_p1_num[::step, ::step] / grad_mag
    grad_p2_norm = grad_p2_num[::step, ::step] / grad_mag
    axes[1].quiver(
        P1[::step, ::step],
        P2[::step, ::step],
        grad_p1_norm,
        grad_p2_norm,
        alpha=ARROW_ALPHA,
        color="black",
    )

    plot_trajectory(axes[1], trajectory_num, bounds, point_alpha=trajectory_num_alpha)
    add_optima_markers(
        axes[1], scipy_result_num, global_result, grid_optimum, precision
    )
    if bounds:
        axes[1].set_xlim(bounds[0], bounds[1])
        axes[1].set_ylim(bounds[2], bounds[3])
    x_label, y_label, pair_tag = _pair_labels(dim_pair)
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(y_label)
    axes[1].set_title(
        f"Gradient Numerical + Numerical Walk ({pair_tag}, zoom {zoom_label})",
        fontsize=9,
    )
    axes[1].set_aspect("equal")
    axes[1].legend(loc="best", fontsize=6)

    # Column 3: Cost landscape with analytical gradient walk
    plot_single_map(
        axes[2],
        P1,
        P2,
        C,
        trajectory_ana,
        scipy_result_ana,
        global_result,
        grid_optimum,
        f"Cost + Analytical Walk ({pair_tag}, zoom {zoom_label})",
        dim_pair,
        "viridis",
        "cost",
        bounds,
        trajectory_ana_alpha,
        precision,
    )

    # Column 4: Analytical gradient with arrows and analytical walk
    pcm = axes[3].pcolormesh(
        P1, P2, grad_norm_ana, shading="auto", cmap="plasma", vmin=gmin, vmax=gmax
    )
    plt.colorbar(pcm, ax=axes[3], label="gradient magnitude")

    # Add gradient arrows (normalized to same length)
    # Normalize gradients to unit vectors
    grad_mag_ana = np.sqrt(
        grad_p1_ana[::step, ::step] ** 2 + grad_p2_ana[::step, ::step] ** 2
    )
    grad_mag_ana = np.where(
        grad_mag_ana == 0, 1, grad_mag_ana
    )  # Avoid division by zero
    grad_p1_ana_norm = grad_p1_ana[::step, ::step] / grad_mag_ana
    grad_p2_ana_norm = grad_p2_ana[::step, ::step] / grad_mag_ana
    axes[3].quiver(
        P1[::step, ::step],
        P2[::step, ::step],
        grad_p1_ana_norm,
        grad_p2_ana_norm,
        alpha=ARROW_ALPHA,
        color="black",
    )

    plot_trajectory(axes[3], trajectory_ana, bounds, point_alpha=trajectory_ana_alpha)
    add_optima_markers(
        axes[3], scipy_result_ana, global_result, grid_optimum, precision
    )
    if bounds:
        axes[3].set_xlim(bounds[0], bounds[1])
        axes[3].set_ylim(bounds[2], bounds[3])
    axes[3].set_xlabel(x_label)
    axes[3].set_ylabel(y_label)
    axes[3].set_title(
        f"Gradient Analytical + Analytical Walk ({pair_tag}, zoom {zoom_label})",
        fontsize=9,
    )
    axes[3].set_aspect("equal")
    axes[3].legend(loc="best", fontsize=6)

    # Column 5: Difference
    plot_single_map(
        axes[4],
        P1,
        P2,
        grad_diff,
        trajectory_ana,
        scipy_result_ana,
        global_result,
        grid_optimum,
        f"Gradient Difference ({pair_tag}, zoom {zoom_label})",
        dim_pair,
        "hot",
        "difference",
        bounds,
        trajectory_ana_alpha,
        precision,
    )


def create_visualization(
    solver,
    bounds,
    full_data,
    zoom_data_list,
    zoom_levels,
    result_num,
    trajectory_num,
    trajectory_num_alpha,
    result_ana,
    trajectory_ana,
    trajectory_ana_alpha,
    best_result,
    grid_optimum,
    method_name,
    dim_pair,
    run_num=1,
):
    """Create and save visualization for a specific optimization method."""
    n_rows = 1 + len(zoom_levels)
    # Create figure with dynamic rows x 5 columns
    fig = plt.figure(figsize=FIGURE_SIZE)
    _, _, pair_tag = _pair_labels(dim_pair)
    fig.suptitle(f"{method_name} | Slice {pair_tag}", fontsize=12)

    # Row 1: Full view (1x)
    axes_row1 = [plt.subplot(n_rows, 5, i + 1) for i in range(5)]
    plot_zoom_row(
        axes_row1,
        full_data,
        trajectory_num,
        trajectory_ana,
        trajectory_num_alpha,
        trajectory_ana_alpha,
        result_num,
        result_ana,
        best_result,
        grid_optimum,
        dim_pair,
        "1x",
        precision=1,
    )

    # Rows 2-6: Zoom levels
    for i, (zoom, data) in enumerate(zip(zoom_levels, zoom_data_list)):
        row = i + 2
        axes_row = [plt.subplot(n_rows, 5, 5 * (row - 1) + j + 1) for j in range(5)]
        precision = (
            2 if zoom <= 100 else (3 if zoom <= 1000 else (4 if zoom <= 10000 else 5))
        )
        plot_zoom_row(
            axes_row,
            data,
            trajectory_num,
            trajectory_ana,
            trajectory_num_alpha,
            trajectory_ana_alpha,
            result_num,
            result_ana,
            best_result,
            grid_optimum,
            dim_pair,
            f"{zoom}x",
            precision,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Include run number in filename if NUM_RUNS > 1
    dim1, dim2 = dim_pair
    if NUM_RUNS > 1:
        filename = (
            f"cost_map_2d_{method_name}_p{dim1 + 1}_p{dim2 + 1}_run{run_num:03d}.png"
        )
    else:
        filename = f"cost_map_2d_{method_name}_p{dim1 + 1}_p{dim2 + 1}.png"

    # Build full output path
    import os

    output_path = os.path.join(OUT_DIR, filename)
    plt.savefig(output_path, dpi=DPI)
    plt.close(fig)
    return f"  Saved {output_path}"


# ============================================================================
# STATISTICS FUNCTIONS
# ============================================================================


def print_statistics_summary(all_run_stats):
    """Print summary statistics across all runs."""
    print("\n" + "=" * 100)
    print("STATISTICS SUMMARY ACROSS ALL RUNS")
    print("=" * 100)

    # Group results by method
    method_stats = {}
    for run_stats in all_run_stats:
        for method_result in run_stats:
            method_name = method_result["method"]
            if method_name not in method_stats:
                method_stats[method_name] = {
                    "numerical": {"iterations": [], "final_cost": [], "success": []},
                    "analytical": {"iterations": [], "final_cost": [], "success": []},
                    "is_gradient_free": method_result["is_gradient_free"],
                }

            if not method_result["is_gradient_free"]:
                method_stats[method_name]["numerical"]["iterations"].append(
                    method_result["num_iterations"]
                )
                method_stats[method_name]["numerical"]["final_cost"].append(
                    method_result["num_final_cost"]
                )
                method_stats[method_name]["numerical"]["success"].append(
                    method_result["num_success"]
                )

            method_stats[method_name]["analytical"]["iterations"].append(
                method_result["ana_iterations"]
            )
            method_stats[method_name]["analytical"]["final_cost"].append(
                method_result["ana_final_cost"]
            )
            method_stats[method_name]["analytical"]["success"].append(
                method_result["ana_success"]
            )

    # Per-run best cost across all methods and gradient variants
    n_runs = len(all_run_stats)
    per_run_best = []
    for run_stats in all_run_stats:
        costs = []
        for r in run_stats:
            costs.append(r["ana_final_cost"])
            if not r["is_gradient_free"] and r["num_final_cost"] is not None:
                costs.append(r["num_final_cost"])
        per_run_best.append(min(costs))

    # Count wins: how many runs each method/variant achieved the per-run best
    # key: (method_name, 'num'|'ana')
    win_counts = {
        (m, v): 0
        for m in method_stats
        for v in (["ana"] if method_stats[m]["is_gradient_free"] else ["num", "ana"])
    }
    for run_idx, run_stats in enumerate(all_run_stats):
        best = per_run_best[run_idx]
        for r in run_stats:
            mn = r["method"]
            if abs(r["ana_final_cost"] - best) < 1e-9:
                win_counts[(mn, "ana")] += 1
            if not r["is_gradient_free"] and r["num_final_cost"] is not None:
                if abs(r["num_final_cost"] - best) < 1e-9:
                    win_counts[(mn, "num")] += 1

    # Print summary for each method
    print(
        f"Overall best-cost across runs: mean={np.mean(per_run_best):.2f}, "
        f"std={np.std(per_run_best):.2f}, min={np.min(per_run_best):.2f}, max={np.max(per_run_best):.2f}"
    )

    for method_name in sorted(method_stats):
        stats = method_stats[method_name]
        print(f"\n{method_name}:")
        print("-" * 100)

        if stats["is_gradient_free"]:
            # Gradient-free method
            iters = stats["analytical"]["iterations"]
            costs = stats["analytical"]["final_cost"]
            success_rate = (
                100
                * sum(stats["analytical"]["success"])
                / len(stats["analytical"]["success"])
            )
            excess = [c - b for c, b in zip(costs, per_run_best)]
            wins = win_counts[(method_name, "ana")]

            print(
                f"  Iterations:  mean={np.mean(iters):.1f}, std={np.std(iters):.1f}, min={np.min(iters)}, max={np.max(iters)}"
            )
            print(
                f"  Final cost:  mean={np.mean(costs):.2f}, std={np.std(costs):.2f}, min={np.min(costs):.2f}, max={np.max(costs):.2f}"
            )
            print(
                f"  Excess vs best: mean={np.mean(excess):.2f}, std={np.std(excess):.2f}, min={np.min(excess):.2f}, max={np.max(excess):.2f}"
            )
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Times best: {wins}/{n_runs}")
        else:
            # Gradient-based method - show both numerical and analytical
            print("  NUMERICAL gradients:")
            num_iters = stats["numerical"]["iterations"]
            num_costs = stats["numerical"]["final_cost"]
            num_success_rate = (
                100
                * sum(stats["numerical"]["success"])
                / len(stats["numerical"]["success"])
            )
            num_excess = [c - b for c, b in zip(num_costs, per_run_best)]
            num_wins = win_counts[(method_name, "num")]

            print(
                f"    Iterations:  mean={np.mean(num_iters):.1f}, std={np.std(num_iters):.1f}, min={np.min(num_iters)}, max={np.max(num_iters)}"
            )
            print(
                f"    Final cost:  mean={np.mean(num_costs):.2f}, std={np.std(num_costs):.2f}, min={np.min(num_costs):.2f}, max={np.max(num_costs):.2f}"
            )
            print(
                f"    Excess vs best: mean={np.mean(num_excess):.2f}, std={np.std(num_excess):.2f}, min={np.min(num_excess):.2f}, max={np.max(num_excess):.2f}"
            )
            print(f"    Success rate: {num_success_rate:.1f}%")
            print(f"    Times best: {num_wins}/{n_runs}")

            print("  ANALYTICAL gradients:")
            ana_iters = stats["analytical"]["iterations"]
            ana_costs = stats["analytical"]["final_cost"]
            ana_success_rate = (
                100
                * sum(stats["analytical"]["success"])
                / len(stats["analytical"]["success"])
            )
            ana_excess = [c - b for c, b in zip(ana_costs, per_run_best)]
            ana_wins = win_counts[(method_name, "ana")]

            print(
                f"    Iterations:  mean={np.mean(ana_iters):.1f}, std={np.std(ana_iters):.1f}, min={np.min(ana_iters)}, max={np.max(ana_iters)}"
            )
            print(
                f"    Final cost:  mean={np.mean(ana_costs):.2f}, std={np.std(ana_costs):.2f}, min={np.min(ana_costs):.2f}, max={np.max(ana_costs):.2f}"
            )
            print(
                f"    Excess vs best: mean={np.mean(ana_excess):.2f}, std={np.std(ana_excess):.2f}, min={np.min(ana_excess):.2f}, max={np.max(ana_excess):.2f}"
            )
            print(f"    Success rate: {ana_success_rate:.1f}%")
            print(f"    Times best: {ana_wins}/{n_runs}")

    print("\n" + "=" * 100)


# ============================================================================
# RUN DATA INITIALIZATION AND PROCESSING
# ============================================================================


def _initialize_run_data(run_nums):
    """Pre-generate all run data before parallel processing.

    Populates global dicts: _solvers, _start_points, _bounds_dict, etc.
    """
    global _solvers, _start_points, _bounds_dict, _p1_ranges, _p2_ranges, _global_optima

    print("\n" + "=" * 100)
    print(f"INITIALIZING {len(run_nums)} RUN(S)")
    print("=" * 100)

    for run_num in tqdm(run_nums, desc="Generating run data"):
        if USE_HEMOGLOBIN:
            E, theoretical_spectra, hemo_params = load_hemoglobin_benchmark_spectra(
                iso_coverage=HEMOGLOBIN_ISO_COVERAGE
            )
            print(
                f"  Run {run_num}: Loaded hemoglobin benchmark with {len(theoretical_spectra)} theoretical spectra"
            )
            # Sanity check: constants should already be configured to benchmark defaults.
            if DISTANCE_METRIC.name != hemo_params["distance_metric"]:
                raise RuntimeError("Hemoglobin distance metric override mismatch")
        else:
            # Generate random spectra
            E, theoretical_spectra = generate_random_spectra(
                n_peaks_E=N_PEAKS_EMPIRICAL,
                n_theoretical=N_THEORETICAL,
                n_peaks_theoretical=N_PEAKS_THEORETICAL,
                theoretical_peak_counts=THEORETICAL_PEAK_COUNTS,
            )

        # Create solver
        solver = DeconvSolver(
            empirical_spectrum=E,
            theoretical_spectra=theoretical_spectra,
            distance=DISTANCE_METRIC,
            max_distance=MAX_DISTANCE,
            trash_cost=TRASH_COST,
            scale_factor=SCALE_FACTOR,
        )

        # Print solver diagnostics
        print(f"  Run {run_num}: Solver created")
        print(f"    Distance metric: {DISTANCE_METRIC}")
        print(f"    Max distance: {MAX_DISTANCE}")
        print(f"    Trash cost: {TRASH_COST}")
        if ENABLE_AUTO_SCALE:
            print(f"    Auto scale factor: {solver.scale_factor:.4e}")
        else:
            print(f"    Manual scale factor: {SCALE_FACTOR:.4e}")

        # Print network diagnostics
        try:
            # Set a point first so diagnostics can access graph info
            solver.set_point(START_POINT)

            if hasattr(solver, "print_diagnostics"):
                print(f"    === Solver Diagnostics ===")
                solver.print_diagnostics(subgraphs_too=True)

            if hasattr(solver, "graph"):
                graph = solver.graph  # It's a property, not a method
                if hasattr(graph, "print_diagnostics"):
                    print(f"    === Graph Diagnostics ===")
                    graph.print_diagnostics()
        except Exception as e:
            print(f"    Could not print full diagnostics: {e}")

        _solvers[run_num] = solver
        _start_points[run_num] = np.array(START_POINT)
        _bounds_dict[run_num] = BOUNDS

        # Find global optimum for marker comparison in plots.
        if ENABLE_PLOTTING:
            print(
                f"  Run {run_num}: Finding global optimum with {GLOBAL_SEARCH_STARTS} random starts..."
            )
            _global_optima[run_num] = find_global_optimum(
                solver, BOUNDS, n_starts=GLOBAL_SEARCH_STARTS, verbose=True
            )


def _create_plot_task(args):
    """Worker function to create a single plot for one method in one run.

    Args:
        args: Tuple of (run_num, method_name, dim_pair, opt_results, zoom_results)

    Returns:
        Tuple of (run_num, method_name, dim_pair, output_path)
    """
    run_num, method_name, dim_pair, opt_results, zoom_results = args

    solver = _solvers[run_num]
    best_result = _global_optima.get(run_num)

    # Organize optimization results
    opt_results_dict = {}
    for r_num, mname, use_numerical_grad, result, trajectory in opt_results:
        if r_num != run_num or mname != method_name:
            continue
        key = "num" if use_numerical_grad else "ana"
        opt_results_dict[key] = (result, trajectory)

    # Organize zoom results
    zoom_results_dict = {}
    for r_num, mname, result_dim_pair, zoom, grid_data in zoom_results:
        if r_num != run_num or mname != method_name:
            continue
        if result_dim_pair != dim_pair:
            continue
        zoom_results_dict[zoom] = grid_data

    # Get results
    result_ana, trajectory_ana = opt_results_dict["ana"]
    is_gradient_free = method_name in GRADIENT_FREE_METHODS

    if is_gradient_free:
        result_num = result_ana
        trajectory_num = trajectory_ana
    else:
        result_num, trajectory_num = opt_results_dict["num"]

    # Axis-pair slice is centered at the analytical optimum for this method.
    # Project all optimization outputs orthogonally onto the (p1, p2) slice plane.
    plane_center = result_ana.x
    trajectory_num_proj, trajectory_num_plane_distance = (
        _project_trajectory_to_axis_pair_plane(trajectory_num, plane_center, dim_pair)
    )
    trajectory_ana_proj, trajectory_ana_plane_distance = (
        _project_trajectory_to_axis_pair_plane(trajectory_ana, plane_center, dim_pair)
    )
    trajectory_num_alpha = _distance_to_alpha(trajectory_num_plane_distance)
    trajectory_ana_alpha = _distance_to_alpha(trajectory_ana_plane_distance)
    result_num_proj = _project_result_to_axis_pair_plane(
        result_num, plane_center, dim_pair
    )
    result_ana_proj = _project_result_to_axis_pair_plane(
        result_ana, plane_center, dim_pair
    )
    best_result_proj = (
        _project_result_to_axis_pair_plane(best_result, plane_center, dim_pair)
        if best_result is not None
        else None
    )

    # Gather full and zoomed axis-pair slice grids
    all_zooms = _all_zoom_levels_with_full(ZOOM_LEVELS)
    full_data = zoom_results_dict[1]
    zoom_data_list = []
    method_grid_results = [full_data["grid_min"]]

    for zoom in all_zooms:
        if zoom == 1:
            continue
        grid_data = zoom_results_dict[zoom]
        zoom_data_list.append(grid_data)
        method_grid_results.append(grid_data["grid_min"])

    # Find best grid minimum
    best_grid_idx = np.argmin([cost for _, _, cost in method_grid_results])
    opt_p1, opt_p2, opt_cost = method_grid_results[best_grid_idx]
    grid_optimum = (opt_p1, opt_p2, opt_cost)

    # Create visualization
    save_msg = create_visualization(
        solver,
        _bounds_dict[run_num],
        full_data,
        zoom_data_list,
        [z for z in all_zooms if z != 1],
        result_num_proj,
        trajectory_num_proj,
        trajectory_num_alpha,
        result_ana_proj,
        trajectory_ana_proj,
        trajectory_ana_alpha,
        best_result_proj,
        grid_optimum,
        method_name,
        dim_pair,
        run_num,
    )

    # Include projection diagnostics in saved message for traceability.
    if (
        trajectory_num_plane_distance.size > 0
        and trajectory_ana_plane_distance.size > 0
    ):
        save_msg += (
            " "
            f"(proj dist num mean={np.mean(trajectory_num_plane_distance):.3f}, "
            f"ana mean={np.mean(trajectory_ana_plane_distance):.3f})"
        )

    return (run_num, method_name, dim_pair, save_msg)


def _process_run_results(run_num, opt_results, zoom_results, plot_messages):
    """Process optimization and zoom results for a single run, generate output.

    Args:
        run_num: Run number
        opt_results: List of (run_num, method_name, use_numerical_grad, result, trajectory) tuples
        zoom_results: List of (run_num, method_name, dim_pair, zoom_level, grid_data) tuples
        plot_messages: Dict of {(run_num, method_name): save_message}

    Returns:
        (run_num, run_stats, output_string)
    """
    out = []
    p = out.append

    p(f"\n{'='*100}")
    p(f"RUN {run_num}/{NUM_RUNS}")
    p(f"{'='*100}")

    solver = _solvers[run_num]

    # Report scale factor
    if ENABLE_AUTO_SCALE:
        p(f"Auto-computed scale factor: {solver.scale_factor:.4e}")
    else:
        p(f"Using manual scale factor: {SCALE_FACTOR:.4e}")

    # Report global optimum if available
    if ENABLE_PLOTTING and run_num in _global_optima:
        best_result = _global_optima[run_num]
        p(
            f"  Global optimum: {_format_point(best_result.x)}, cost={best_result.fun:.2f}"
        )
    else:
        best_result = None

    # Organize optimization results by method
    opt_results_dict = (
        {}
    )  # {method_name: {'num': (result, traj), 'ana': (result, traj)}}
    for r_num, method_name, use_numerical_grad, result, trajectory in opt_results:
        if r_num != run_num:
            continue
        if method_name not in opt_results_dict:
            opt_results_dict[method_name] = {}
        key = "num" if use_numerical_grad else "ana"
        opt_results_dict[method_name][key] = (result, trajectory)

    # Process each method
    run_stats = []

    for method_name, display_name in METHODS:
        p(f"\n{'-'*100}")
        p(f"TESTING METHOD: {display_name} ({method_name})")
        p(f"{'-'*100}")

        is_gradient_free = method_name in GRADIENT_FREE_METHODS

        # Get optimization results
        result_ana, trajectory_ana = opt_results_dict[method_name]["ana"]

        if is_gradient_free:
            p(
                f"\nCompleted {display_name} (gradient-free) from starting point: {_format_point(START_POINT)}"
            )
            p(f"  Converged: {result_ana.success}")
            p(
                f"  Final point: {_format_point(result_ana.x)}, cost={result_ana.fun:.2f}"
            )
            p(f"  Iterations: {len(trajectory_ana)}")

            # Use same result for both columns (for plotting only)
            result_num = result_ana
            trajectory_num = trajectory_ana

            # Record statistics
            run_stats.append(
                {
                    "method": method_name,
                    "is_gradient_free": True,
                    "ana_iterations": len(trajectory_ana),
                    "ana_final_cost": result_ana.fun,
                    "ana_success": result_ana.success,
                    "num_iterations": None,
                    "num_final_cost": None,
                    "num_success": None,
                }
            )
        else:
            result_num, trajectory_num = opt_results_dict[method_name]["num"]

            p(
                f"\nCompleted {display_name} with NUMERICAL gradients from starting point: {_format_point(START_POINT)}"
            )
            p(f"  Converged: {result_num.success}")
            p(
                f"  Final point: {_format_point(result_num.x)}, cost={result_num.fun:.2f}"
            )
            p(f"  Iterations: {len(trajectory_num)}")

            p(
                f"\nCompleted {display_name} with ANALYTICAL gradients from starting point: {_format_point(START_POINT)}"
            )
            p(f"  Converged: {result_ana.success}")
            p(
                f"  Final point: {_format_point(result_ana.x)}, cost={result_ana.fun:.2f}"
            )
            p(f"  Iterations: {len(trajectory_ana)}")

            # Record statistics
            run_stats.append(
                {
                    "method": method_name,
                    "is_gradient_free": False,
                    "num_iterations": len(trajectory_num),
                    "num_final_cost": result_num.fun,
                    "num_success": result_num.success,
                    "ana_iterations": len(trajectory_ana),
                    "ana_final_cost": result_ana.fun,
                    "ana_success": result_ana.success,
                }
            )

        # Add plot save message if available
        if ENABLE_PLOTTING:
            for dim_pair in AXIS_PAIRS:
                key = (run_num, method_name, dim_pair)
                if key in plot_messages:
                    p(plot_messages[key])

    return run_num, run_stats, "\n".join(out)


def main():
    """Main entry point."""
    import os

    # Create output directory if it doesn't exist
    if ENABLE_PLOTTING and OUT_DIR != ".":
        os.makedirs(OUT_DIR, exist_ok=True)

    # Set random seed if provided
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)
        print(f"Random seed set to: {RANDOM_SEED}")

    print("=" * 100)
    print("COST MAP 2D - OPTIMIZATION COMPARISON")
    print("=" * 100)
    print(f"Configuration:")
    print(f"  Plotting enabled: {ENABLE_PLOTTING}")
    print(
        f"  Dataset mode: {'hemoglobin (masserstein)' if USE_HEMOGLOBIN else 'random'}"
    )
    print(f"  Number of runs: {NUM_RUNS}")
    print(f"  Number of theoretical spectra: {N_THEORETICAL}")
    print(f"  Axis pairs: {len(AXIS_PAIRS)}")
    if not USE_HEMOGLOBIN:
        print(f"  Theoretical peak counts: {THEORETICAL_PEAK_COUNTS}")
    else:
        print(f"  Hemoglobin iso coverage: {HEMOGLOBIN_ISO_COVERAGE}")
    print(f"  Methods to test: {len(METHODS)}")
    print(f"  Grid resolution: {GRID_RESOLUTION}")
    print(f"  Zoom levels: {ZOOM_LEVELS}")
    print(
        f"  Random seed: {RANDOM_SEED if RANDOM_SEED is not None else 'None (random)'}"
    )
    print(f"  Auto-scaling enabled: {ENABLE_AUTO_SCALE}")
    if not ENABLE_AUTO_SCALE and SCALE_FACTOR is not None:
        print(f"  Manual scale factor: {SCALE_FACTOR}")
    if ENABLE_PLOTTING:
        print(f"  Output directory: {os.path.abspath(OUT_DIR)}")

    n_workers = (
        _args.workers if _args.workers is not None else multiprocessing.cpu_count()
    )
    print(f"  Workers: {n_workers} (multiprocessing)")

    # ========================================================================
    # PHASE 0: Initialize all run data
    # ========================================================================
    run_nums = list(range(1, NUM_RUNS + 1))
    _initialize_run_data(run_nums)

    # ========================================================================
    # PHASE 1: Run ALL optimization tasks in parallel
    # ========================================================================
    print("\n" + "=" * 100)
    print("PHASE 1: RUNNING ALL OPTIMIZATIONS IN PARALLEL")
    print("=" * 100)

    opt_tasks = []
    for run_num in run_nums:
        for method_name, display_name in METHODS:
            is_gradient_free = method_name in GRADIENT_FREE_METHODS
            if not is_gradient_free:
                # Gradient-based: run both numerical and analytical
                opt_tasks.append(
                    (run_num, method_name, display_name, True)
                )  # numerical
                opt_tasks.append(
                    (run_num, method_name, display_name, False)
                )  # analytical
            else:
                # Gradient-free: run only once (analytical)
                opt_tasks.append((run_num, method_name, display_name, False))

    print(f"Total optimization tasks: {len(opt_tasks)}")

    with multiprocessing.Pool(processes=n_workers) as pool:
        all_opt_results = list(
            tqdm(
                pool.imap(_run_optimization_task, opt_tasks),
                total=len(opt_tasks),
                desc="All optimizations",
            )
        )

    # ========================================================================
    # PHASE 2: Run ALL zoom grid tasks in parallel
    # ========================================================================
    if ENABLE_PLOTTING:
        print("\n" + "=" * 100)
        print("PHASE 2: COMPUTING ALL ZOOM GRIDS IN PARALLEL")
        print("=" * 100)

        # Organize optimization results to extract zoom centers
        opt_by_run_method = {}  # {(run_num, method_name): {'num': ..., 'ana': ...}}
        for (
            run_num,
            method_name,
            use_numerical_grad,
            result,
            trajectory,
        ) in all_opt_results:
            key = (run_num, method_name)
            if key not in opt_by_run_method:
                opt_by_run_method[key] = {}
            opt_key = "num" if use_numerical_grad else "ana"
            opt_by_run_method[key][opt_key] = (result, trajectory)

        # Create all zoom tasks (including zoom=1 full view)
        zoom_tasks = []
        all_zooms = _all_zoom_levels_with_full(ZOOM_LEVELS)
        for run_num in run_nums:
            for method_name, display_name in METHODS:
                # Get analytical result for this method (slice center in N-D)
                result_ana, _ = opt_by_run_method[(run_num, method_name)]["ana"]
                center_point = result_ana.x

                # Create tasks for all axis-aligned dimension pairs and zoom levels
                for dim_pair in AXIS_PAIRS:
                    for zoom in all_zooms:
                        zoom_tasks.append(
                            (run_num, method_name, dim_pair, zoom, center_point)
                        )

        print(f"Total zoom grid tasks: {len(zoom_tasks)}")

        with multiprocessing.Pool(processes=n_workers) as pool:
            all_zoom_results = list(
                tqdm(
                    pool.imap(_compute_zoom_grid_task, zoom_tasks),
                    total=len(zoom_tasks),
                    desc="All zoom grids",
                )
            )
    else:
        all_zoom_results = []

    # ========================================================================
    # PHASE 3: Create all plots in parallel
    # ========================================================================
    print("\n" + "=" * 100)
    print("PHASE 3: CREATING VISUALIZATIONS")
    print("=" * 100)

    plot_messages = {}

    if ENABLE_PLOTTING:
        # Create plotting tasks for all run/method/dim_pair combinations
        plot_tasks = []
        for run_num in run_nums:
            for method_name, _ in METHODS:
                for dim_pair in AXIS_PAIRS:
                    plot_tasks.append(
                        (
                            run_num,
                            method_name,
                            dim_pair,
                            all_opt_results,
                            all_zoom_results,
                        )
                    )

        # Execute plots in parallel
        if n_workers > 1:
            print(f"Total plotting tasks: {len(plot_tasks)}")
            with multiprocessing.Pool(processes=n_workers) as pool:
                plot_results = list(
                    tqdm(
                        pool.imap(_create_plot_task, plot_tasks),
                        total=len(plot_tasks),
                        desc="All plots",
                    )
                )
        else:
            plot_results = [
                _create_plot_task(task) for task in tqdm(plot_tasks, desc="All plots")
            ]

        # Organize plot messages by (run_num, method_name, dim_pair)
        plot_messages = {(r, m, d): msg for r, m, d, msg in plot_results}

    # ========================================================================
    # PHASE 4: Print results in order
    # ========================================================================
    print("\n" + "=" * 100)
    print("PHASE 4: SUMMARY")
    print("=" * 100)

    all_run_stats = []
    for run_num in run_nums:
        run_num_result, run_stats, output = _process_run_results(
            run_num, all_opt_results, all_zoom_results, plot_messages
        )
        print(output)
        all_run_stats.append(run_stats)

    # Print summary statistics if multiple runs
    if NUM_RUNS > 1:
        print_statistics_summary(all_run_stats)

    print("\n" + "=" * 100)
    print("COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
