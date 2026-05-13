"""2D map of (p1, p2) -> total_cost for the two-spectrum deconvolution example."""

import argparse
import multiprocessing
import numpy as np
from tqdm import tqdm
from wnetdeconv import DeconvSolver
from wnet.distances import DistanceMetric
from experiments_support import (
    generate_random_spectra,
    compute_cost_grid,
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
_full_grids = {}  # {run_num: grid_data_dict}
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
        nargs=2,
        type=float,
        default=[15.0, 20.0],
        metavar=("P1", "P2"),
        help="Starting point for optimization",
    )
    parser.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        default=[0, 25, 0, 25],
        metavar=("P1_MIN", "P1_MAX", "P2_MIN", "P2_MAX"),
        help="Bounds for p1 and p2",
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
        choices=["L1", "L2"],
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
        "--out-dir",
        type=str,
        default=".",
        metavar="DIR",
        help="Output directory for saved plots (default: current directory)",
    )

    return parser


_args = _build_parser().parse_args()

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Control flags
ENABLE_PLOTTING = not _args.no_plot
NUM_RUNS = _args.num_runs

# Optimization settings
START_POINT = _args.start_point
BOUNDS = [(_args.bounds[0], _args.bounds[1]), (_args.bounds[2], _args.bounds[3])]
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
DISTANCE_METRIC = DistanceMetric[_args.distance_metric]
MAX_DISTANCE = _args.max_distance
TRASH_COST = _args.trash_cost
ENABLE_AUTO_SCALE = _args.auto_scale
SCALE_FACTOR = None if ENABLE_AUTO_SCALE else _args.scale_factor

# Random seed
RANDOM_SEED = _args.random_seed

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
    )
    return (run_num, method_name, use_numerical_grad, result, trajectory)


def _compute_zoom_grid_task(args):
    """Worker function to compute a zoom grid.

    Args:
        args: Tuple of (run_num, method_name, zoom_level, center_p1, center_p2)

    Returns:
        Tuple of (run_num, method_name, zoom_level, grid_data_dict)
    """
    run_num, method_name, zoom, center_p1, center_p2 = args

    solver = _solvers[run_num]
    bounds = _bounds_dict[run_num]
    p1_range = _p1_ranges[run_num]
    p2_range = _p2_ranges[run_num]

    # Calculate zoom bounds
    zoom_width = (p1_range[1] - p1_range[0]) / zoom

    # Center on provided center point
    p1_min = center_p1 - zoom_width / 2
    p1_max = center_p1 + zoom_width / 2
    p2_min = center_p2 - zoom_width / 2
    p2_max = center_p2 + zoom_width / 2

    # Shift if hitting boundaries, maintaining width
    if p1_min < bounds[0][0]:
        p1_min = bounds[0][0]
        p1_max = p1_min + zoom_width
    elif p1_max > bounds[0][1]:
        p1_max = bounds[0][1]
        p1_min = p1_max - zoom_width

    if p2_min < bounds[1][0]:
        p2_min = bounds[1][0]
        p2_max = p2_min + zoom_width
    elif p2_max > bounds[1][1]:
        p2_max = bounds[1][1]
        p2_min = p2_max - zoom_width

    # Compute grid (without verbose output in worker process)
    P1_z, P2_z, C_z, p1_z, p2_z, grad_p1_ana_z, grad_p2_ana_z = compute_cost_grid(
        solver, (p1_min, p1_max), (p2_min, p2_max), GRID_RESOLUTION, verbose=False
    )
    grad_p2_num_z, grad_p1_num_z = np.gradient(C_z, p2_z, p1_z)

    # Track grid minimum
    min_idx_zoom = np.unravel_index(np.argmin(C_z), C_z.shape)
    grid_min = (P1_z[min_idx_zoom], P2_z[min_idx_zoom], C_z[min_idx_zoom])

    grid_data = {
        "zoom": zoom,
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

    return (run_num, method_name, zoom, grid_data)


def plot_trajectory(ax, trajectory, zoom_bounds=None):
    """Plot optimization trajectory with black-to-white gradient."""
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
    else:
        traj = trajectory

    if len(traj) < 2:
        return

    # Plot line segments
    points = traj.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = np.linspace(0, 1, len(segments))
    lc = LineCollection(segments, cmap="gray", linewidth=2, zorder=4, label="trace")
    lc.set_array(colors)
    ax.add_collection(lc)

    # Plot points
    traj_colors = np.linspace(0, 1, len(traj))
    ax.scatter(
        traj[:, 0],
        traj[:, 1],
        c=traj_colors,
        cmap="gray",
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
    cmap,
    label,
    zoom_bounds=None,
    precision=2,
    vmin=None,
    vmax=None,
):
    """Plot a single map (cost or gradient magnitude) with trajectory."""
    if vmin is None or vmax is None:
        vmin, vmax = np.percentile(data, [5, 95])

    pcm = ax.pcolormesh(P1, P2, data, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(pcm, ax=ax, label=label)

    plot_trajectory(ax, trajectory, zoom_bounds)
    add_optima_markers(ax, scipy_result, global_result, grid_optimum, precision)

    # Set axis limits if zoom bounds provided
    if zoom_bounds:
        ax.set_xlim(zoom_bounds[0], zoom_bounds[1])
        ax.set_ylim(zoom_bounds[2], zoom_bounds[3])

    ax.set_xlabel("proportion 1")
    ax.set_ylabel("proportion 2")
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=6)


def plot_zoom_row(
    axes,
    data_dict,
    trajectory_num,
    trajectory_ana,
    scipy_result_num,
    scipy_result_ana,
    global_result,
    grid_optimum,
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
        f"Cost + Numerical Walk (zoom {zoom_label})",
        "viridis",
        "cost",
        bounds,
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

    plot_trajectory(axes[1], trajectory_num, bounds)
    add_optima_markers(
        axes[1], scipy_result_num, global_result, grid_optimum, precision
    )
    if bounds:
        axes[1].set_xlim(bounds[0], bounds[1])
        axes[1].set_ylim(bounds[2], bounds[3])
    axes[1].set_xlabel("proportion 1")
    axes[1].set_ylabel("proportion 2")
    axes[1].set_title(
        f"Gradient Numerical + Numerical Walk (zoom {zoom_label})", fontsize=9
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
        f"Cost + Analytical Walk (zoom {zoom_label})",
        "viridis",
        "cost",
        bounds,
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

    plot_trajectory(axes[3], trajectory_ana, bounds)
    add_optima_markers(
        axes[3], scipy_result_ana, global_result, grid_optimum, precision
    )
    if bounds:
        axes[3].set_xlim(bounds[0], bounds[1])
        axes[3].set_ylim(bounds[2], bounds[3])
    axes[3].set_xlabel("proportion 1")
    axes[3].set_ylabel("proportion 2")
    axes[3].set_title(
        f"Gradient Analytical + Analytical Walk (zoom {zoom_label})", fontsize=9
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
        f"Gradient Difference (zoom {zoom_label})",
        "hot",
        "difference",
        bounds,
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
    result_ana,
    trajectory_ana,
    best_result,
    grid_optimum,
    method_name,
    run_num=1,
):
    """Create and save visualization for a specific optimization method."""
    # Create figure with 6 rows x 5 columns
    fig = plt.figure(figsize=FIGURE_SIZE)

    # Row 1: Full view (1x)
    axes_row1 = [plt.subplot(6, 5, i + 1) for i in range(5)]
    plot_zoom_row(
        axes_row1,
        full_data,
        trajectory_num,
        trajectory_ana,
        result_num,
        result_ana,
        best_result,
        grid_optimum,
        "1x",
        precision=1,
    )

    # Rows 2-6: Zoom levels
    for i, (zoom, data) in enumerate(zip(zoom_levels, zoom_data_list)):
        row = i + 2
        axes_row = [plt.subplot(6, 5, 5 * (row - 1) + j + 1) for j in range(5)]
        precision = (
            2 if zoom <= 100 else (3 if zoom <= 1000 else (4 if zoom <= 10000 else 5))
        )
        plot_zoom_row(
            axes_row,
            data,
            trajectory_num,
            trajectory_ana,
            result_num,
            result_ana,
            best_result,
            grid_optimum,
            f"{zoom}x",
            precision,
        )

    plt.tight_layout()

    # Include run number in filename if NUM_RUNS > 1
    if NUM_RUNS > 1:
        filename = f"cost_map_2d_{method_name}_run{run_num:03d}.png"
    else:
        filename = f"cost_map_2d_{method_name}.png"

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
    for method_name, stats in method_stats.items():
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
    global _solvers, _start_points, _bounds_dict, _p1_ranges, _p2_ranges, _full_grids, _global_optima

    print("\n" + "=" * 100)
    print(f"INITIALIZING {len(run_nums)} RUN(S)")
    print("=" * 100)

    for run_num in tqdm(run_nums, desc="Generating run data"):
        # Generate spectra
        E, T1, T2 = generate_random_spectra()

        # Create solver
        solver = DeconvSolver(
            empirical_spectrum=E,
            theoretical_spectra=[T1, T2],
            distance=DISTANCE_METRIC,
            max_distance=MAX_DISTANCE,
            trash_cost=TRASH_COST,
            scale_factor=SCALE_FACTOR,
        )

        _solvers[run_num] = solver
        _start_points[run_num] = np.array(START_POINT)
        _bounds_dict[run_num] = BOUNDS

        # Compute full grid if plotting enabled
        if ENABLE_PLOTTING:
            P1, P2, C, p1, p2, grad_p1_ana, grad_p2_ana = compute_cost_grid(
                solver, BOUNDS[0], BOUNDS[1], GRID_RESOLUTION, verbose=False
            )
            grad_p2_num, grad_p1_num = np.gradient(C, p2, p1)

            _p1_ranges[run_num] = (p1[0], p1[-1])
            _p2_ranges[run_num] = (p2[0], p2[-1])

            # Store grid minimum
            min_idx = np.unravel_index(np.argmin(C), C.shape)
            grid_min = (P1[min_idx], P2[min_idx], C[min_idx])

            _full_grids[run_num] = {
                "P1": P1,
                "P2": P2,
                "C": C,
                "grad_p1_num": grad_p1_num,
                "grad_p2_num": grad_p2_num,
                "grad_p1_ana": grad_p1_ana,
                "grad_p2_ana": grad_p2_ana,
                "bounds": None,
                "grid_min": grid_min,
            }

            # Find global optimum
            _global_optima[run_num] = find_global_optimum(
                solver, BOUNDS, n_starts=GLOBAL_SEARCH_STARTS, verbose=False
            )

        if ENABLE_AUTO_SCALE:
            print(
                f"  Run {run_num}: scale factor = {_solvers[run_num].scale_factor:.4e}"
            )
        else:
            print(f"  Run {run_num}: manual scale factor = {SCALE_FACTOR:.4e}")


def _create_plot_task(args):
    """Worker function to create a single plot for one method in one run.

    Args:
        args: Tuple of (run_num, method_name, opt_results, zoom_results)

    Returns:
        Tuple of (run_num, method_name, output_path)
    """
    run_num, method_name, opt_results, zoom_results = args

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
    for r_num, mname, zoom, grid_data in zoom_results:
        if r_num != run_num or mname != method_name:
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

    # Gather zoom data
    full_data = _full_grids[run_num]
    zoom_data_list = []
    method_grid_results = [full_data["grid_min"]]

    for zoom in ZOOM_LEVELS:
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
        ZOOM_LEVELS,
        result_num,
        trajectory_num,
        result_ana,
        trajectory_ana,
        best_result,
        grid_optimum,
        method_name,
        run_num,
    )

    return (run_num, method_name, save_msg)


def _process_run_results(run_num, opt_results, zoom_results, plot_messages):
    """Process optimization and zoom results for a single run, generate output.

    Args:
        run_num: Run number
        opt_results: List of (run_num, method_name, use_numerical_grad, result, trajectory) tuples
        zoom_results: List of (run_num, method_name, zoom_level, grid_data) tuples
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
            f"  Global optimum: p1={best_result.x[0]:.2f}, p2={best_result.x[1]:.2f}, cost={best_result.fun:.2f}"
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

    # Organize zoom results by method
    zoom_results_dict = {}  # {method_name: {zoom_level: grid_data}}
    if ENABLE_PLOTTING:
        for r_num, method_name, zoom, grid_data in zoom_results:
            if r_num != run_num:
                continue
            if method_name not in zoom_results_dict:
                zoom_results_dict[method_name] = {}
            zoom_results_dict[method_name][zoom] = grid_data

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
                f"\nCompleted {display_name} (gradient-free) from starting point: p1={START_POINT[0]:.2f}, p2={START_POINT[1]:.2f}"
            )
            p(f"  Converged: {result_ana.success}")
            p(
                f"  Final point: p1={result_ana.x[0]:.2f}, p2={result_ana.x[1]:.2f}, cost={result_ana.fun:.2f}"
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
                f"\nCompleted {display_name} with NUMERICAL gradients from starting point: p1={START_POINT[0]:.2f}, p2={START_POINT[1]:.2f}"
            )
            p(f"  Converged: {result_num.success}")
            p(
                f"  Final point: p1={result_num.x[0]:.2f}, p2={result_num.x[1]:.2f}, cost={result_num.fun:.2f}"
            )
            p(f"  Iterations: {len(trajectory_num)}")

            p(
                f"\nCompleted {display_name} with ANALYTICAL gradients from starting point: p1={START_POINT[0]:.2f}, p2={START_POINT[1]:.2f}"
            )
            p(f"  Converged: {result_ana.success}")
            p(
                f"  Final point: p1={result_ana.x[0]:.2f}, p2={result_ana.x[1]:.2f}, cost={result_ana.fun:.2f}"
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
        if ENABLE_PLOTTING and (run_num, method_name) in plot_messages:
            p(plot_messages[(run_num, method_name)])

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
    print(f"  Number of runs: {NUM_RUNS}")
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

        # Create all zoom tasks
        zoom_tasks = []
        for run_num in run_nums:
            for method_name, display_name in METHODS:
                # Get analytical result for this method (for centering zoom)
                result_ana, _ = opt_by_run_method[(run_num, method_name)]["ana"]
                center_p1 = result_ana.x[0]
                center_p2 = result_ana.x[1]

                # Create tasks for all zoom levels
                for zoom in ZOOM_LEVELS:
                    zoom_tasks.append(
                        (run_num, method_name, zoom, center_p1, center_p2)
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
        # Create plotting tasks for all run/method combinations
        plot_tasks = []
        for run_num in run_nums:
            for method_name, _ in METHODS:
                plot_tasks.append(
                    (run_num, method_name, all_opt_results, all_zoom_results)
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

        # Organize plot messages by (run_num, method_name)
        plot_messages = {(r, m): msg for r, m, msg in plot_results}

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
