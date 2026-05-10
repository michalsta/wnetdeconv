"""N-dimensional map with 2D slices through the true optimum for N-spectrum deconvolution."""
import argparse
import multiprocessing
import numpy as np
from tqdm import tqdm
from wnetdeconv import DeconvSolver
from wnet.distances import DistanceMetric
from experiments_support import (
    generate_random_spectra,
    run_optimization,
    find_global_optimum
)

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def _build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dimensionality
    parser.add_argument('--n-spectra', type=int, default=3, metavar='N',
                        help='Number of theoretical spectra (dimensionality)')

    # Slice configuration
    parser.add_argument('--n-principal-slices', type=int, default=None, metavar='N',
                        help='Number of principal axis slices (default: all pairs, i.e., N*(N-1)/2)')
    parser.add_argument('--n-random-slices', type=int, default=3, metavar='N',
                        help='Number of random orientation slices')

    # Control flags
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable all plot generation')
    parser.add_argument('--num-runs', type=int, default=1, metavar='N',
                        help='Number of runs with different random spectra')

    # Optimization settings
    parser.add_argument('--start-point', nargs='+', type=float,
                        default=None, metavar='P',
                        help='Starting point for optimization (must match n-spectra dimensions)')
    parser.add_argument('--bounds', nargs=2, type=float,
                        default=[0, 25], metavar=('MIN', 'MAX'),
                        help='Bounds for all proportions (same for all dimensions)')
    parser.add_argument('--max-iterations', type=int, default=200, metavar='N',
                        help='Maximum iterations for optimization')
    parser.add_argument('--global-search-starts', type=int, default=50, metavar='N',
                        help='Number of random starts for global optimum search')

    # Grid computation settings
    parser.add_argument('--grid-resolution', type=int, default=150, metavar='N',
                        help='Number of points per axis for cost grid (2D slices)')
    parser.add_argument('--zoom-levels', nargs='+', type=int,
                        default=[10, 100, 1000], metavar='Z',
                        help='Zoom levels to compute')

    # Plotting settings (only used without --no-plot)
    parser.add_argument('--figure-size', nargs=2, type=float,
                        default=[30.0, 30.0], metavar=('W', 'H'),
                        help='Figure size in inches (per slice)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Resolution for saved plots')
    parser.add_argument('--arrow-subsample', type=int, default=20, metavar='N',
                        help='Show every Nth arrow in gradient field')
    parser.add_argument('--arrow-alpha', type=float, default=0.6,
                        help='Transparency of gradient arrows')

    # Deconvolution solver parameters
    parser.add_argument('--distance-metric', choices=['L1', 'L2'], default='L2',
                        help='Distance metric for the deconvolution solver')
    parser.add_argument('--max-distance', type=float, default=10,
                        help='Maximum distance for the deconvolution solver')
    parser.add_argument('--trash-cost', type=float, default=100,
                        help='Trash cost for the deconvolution solver')
    parser.add_argument('--auto-scale', action='store_true', default=True,
                        help='Enable automatic scale factor computation based on data characteristics (default: True)')
    parser.add_argument('--no-auto-scale', action='store_false', dest='auto_scale',
                        help='Disable automatic scale factor computation and use manual value instead')
    parser.add_argument('--scale-factor', type=float, default=None, metavar='SF',
                        help='Manual scale factor for numerical stability (only used if --no-auto-scale is set)')
    parser.add_argument('--workers', type=int, default=None, metavar='N',
                        help='Number of parallel worker processes (default: number of CPUs)')
    parser.add_argument('--random-seed', type=int, default=None, metavar='SEED',
                        help='Random seed for reproducibility (default: None, use random seed)')
    parser.add_argument('--out-dir', type=str, default='.', metavar='DIR',
                        help='Output directory for saved plots (default: current directory)')

    return parser


_args = _build_parser().parse_args()

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Dimensionality
N_SPECTRA = _args.n_spectra
N_DIMS = N_SPECTRA

# Slice configuration
if _args.n_principal_slices is None:
    # Default: all pairs of principal axes
    N_PRINCIPAL_SLICES = (N_DIMS * (N_DIMS - 1)) // 2
else:
    N_PRINCIPAL_SLICES = _args.n_principal_slices
N_RANDOM_SLICES = _args.n_random_slices

# Control flags
ENABLE_PLOTTING = not _args.no_plot
NUM_RUNS = _args.num_runs

# Optimization settings
if _args.start_point is None:
    # Default starting point: middle of bounds for all dimensions
    START_POINT = [((_args.bounds[0] + _args.bounds[1]) / 2) * 0.8] * N_DIMS
else:
    START_POINT = _args.start_point
    if len(START_POINT) != N_DIMS:
        raise ValueError(f"--start-point must have {N_DIMS} values (got {len(START_POINT)})")

BOUNDS = [(_args.bounds[0], _args.bounds[1])] * N_DIMS
MAX_ITERATIONS = _args.max_iterations
GLOBAL_SEARCH_STARTS = _args.global_search_starts

# Grid computation settings
GRID_RESOLUTION = _args.grid_resolution
ZOOM_LEVELS = _args.zoom_levels

# Plotting settings
FIGURE_SIZE = tuple(_args.figure_size)
DPI = _args.dpi
ARROW_SUBSAMPLE = _args.arrow_subsample
ARROW_ALPHA = _args.arrow_alpha

# Optimization methods to test
METHODS = [
    ('L-BFGS-B', 'Limited-memory BFGS with Bounds'),
    ('TNC', 'Truncated Newton Conjugate-Gradient'),
    ('SLSQP', 'Sequential Least Squares Quadratic Programming'),
    ('Nelder-Mead', 'Nelder-Mead Simplex'),
    ('Powell', 'Powell Direction Set'),
    ('COBYLA', 'Constrained Optimization BY Linear Approximation'),
]

GRADIENT_FREE_METHODS = ['Nelder-Mead', 'Powell', 'COBYLA']

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
# N-DIMENSIONAL GRID AND SLICE FUNCTIONS
# ============================================================================

def generate_slice_directions(n_dims, n_principal, n_random, random_state=None):
    """Generate slice directions (pairs of orthonormal basis vectors in N-D space).

    Returns:
        List of tuples (axis1_idx, axis2_idx, u, v, label) where:
        - axis1_idx, axis2_idx: indices for principal slices (None for random)
        - u, v: orthonormal vectors defining the 2D slice plane
        - label: descriptive label for the slice
    """
    rng = np.random.RandomState(random_state)
    slices = []

    # Principal axis slices
    principal_pairs = []
    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            principal_pairs.append((i, j))

    # Select requested number of principal slices
    if n_principal > len(principal_pairs):
        n_principal = len(principal_pairs)

    selected_pairs = principal_pairs[:n_principal]

    for i, j in selected_pairs:
        u = np.zeros(n_dims)
        v = np.zeros(n_dims)
        u[i] = 1.0
        v[j] = 1.0
        slices.append((i, j, u, v, f"principal_p{i+1}_p{j+1}"))

    # Random slices
    for r in range(n_random):
        # Generate two random orthonormal vectors
        u = rng.randn(n_dims)
        u = u / np.linalg.norm(u)

        v = rng.randn(n_dims)
        # Gram-Schmidt orthogonalization
        v = v - np.dot(v, u) * u
        v = v / np.linalg.norm(v)

        slices.append((None, None, u, v, f"random_{r+1}"))

    return slices


def compute_cost_on_2d_slice(solver, center_point, u, v, bounds_2d, resolution):
    """Compute cost on a 2D slice through center_point along directions u and v.

    Args:
        solver: DeconvSolver instance
        center_point: N-D point to slice through (typically the global optimum)
        u, v: orthonormal vectors defining the slice plane
        bounds_2d: ((u_min, u_max), (v_min, v_max)) bounds in the 2D slice coordinates
        resolution: number of grid points per dimension

    Returns:
        U, V, C, u_1d, v_1d, grad_u_ana, grad_v_ana
    """
    u_min, u_max = bounds_2d[0]
    v_min, v_max = bounds_2d[1]

    u_1d = np.linspace(u_min, u_max, resolution)
    v_1d = np.linspace(v_min, v_max, resolution)
    U, V = np.meshgrid(u_1d, v_1d)

    C = np.zeros_like(U)
    grad_u_ana = np.zeros_like(U)
    grad_v_ana = np.zeros_like(U)

    # Compute cost and gradients at each grid point
    for i in range(resolution):
        for j in range(resolution):
            # Convert 2D slice coordinates to N-D space
            point_nd = center_point + U[i, j] * u + V[i, j] * v

            # Ensure we're within bounds and normalize
            point_nd = np.clip(point_nd, 0, None)

            # Compute cost and gradient
            solver.set_point(point_nd)
            cost = solver.total_cost()
            grad_nd = np.array(solver.gradient())

            C[i, j] = cost
            # Project gradient onto slice plane
            grad_u_ana[i, j] = np.dot(grad_nd, u)
            grad_v_ana[i, j] = np.dot(grad_nd, v)

    return U, V, C, u_1d, v_1d, grad_u_ana, grad_v_ana


def project_trajectory_to_slice(trajectory_nd, center_point, u, v):
    """Project N-D trajectory onto a 2D slice.

    Args:
        trajectory_nd: (n_points, n_dims) array of trajectory points
        center_point: N-D center of the slice
        u, v: orthonormal vectors defining the slice plane

    Returns:
        (n_points, 2) array of 2D coordinates in (u, v) basis
    """
    # Translate trajectory relative to center
    traj_centered = trajectory_nd - center_point

    # Project onto u and v directions
    traj_2d = np.zeros((len(trajectory_nd), 2))
    traj_2d[:, 0] = np.dot(traj_centered, u)
    traj_2d[:, 1] = np.dot(traj_centered, v)

    return traj_2d


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

if ENABLE_PLOTTING:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    def plot_trajectory(ax, trajectory, zoom_bounds=None):
        """Plot optimization trajectory with black-to-white gradient."""
        if zoom_bounds:
            u_min, u_max, v_min, v_max = zoom_bounds
            mask = ((trajectory[:, 0] >= u_min) & (trajectory[:, 0] <= u_max) &
                    (trajectory[:, 1] >= v_min) & (trajectory[:, 1] <= v_max))
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
        lc = LineCollection(segments, cmap='gray', linewidth=2, zorder=4, label='trace')
        lc.set_array(colors)
        ax.add_collection(lc)

        # Plot points
        traj_colors = np.linspace(0, 1, len(traj))
        ax.scatter(traj[:, 0], traj[:, 1], c=traj_colors, cmap='gray',
                   s=15, zorder=4, edgecolors='black', linewidths=0.3)

        # Mark start point
        ax.scatter([traj[0, 0]], [traj[0, 1]], color='red', marker='o',
                   s=40, zorder=5, edgecolors='white', linewidths=1.0, label='start')


    def add_optima_markers(ax, scipy_result_2d, global_result_2d, grid_optimum, precision=2):
        """Add three optima markers with costs in legend."""
        fmt = f"{{:.{precision}f}}"

        # Scipy endpoint
        scipy_label = f"scipy end ({fmt.format(scipy_result_2d[0])}, {fmt.format(scipy_result_2d[1])}, cost={scipy_result_2d[2]:.0f})"
        ax.scatter([scipy_result_2d[0]], [scipy_result_2d[1]], color='cyan', marker='X',
                   s=50, zorder=7, edgecolors='black', linewidths=1.0, label=scipy_label)

        # Global optimum (should be at origin since we center on it)
        global_label = f"global opt (0.0, 0.0, cost={global_result_2d[2]:.0f})"
        ax.scatter([0.0], [0.0], color='magenta', marker='D',
                   s=50, zorder=7, edgecolors='white', linewidths=1.0, label=global_label)

        # Grid minimum
        grid_label = f"grid min ({fmt.format(grid_optimum[0])}, {fmt.format(grid_optimum[1])}, cost={grid_optimum[2]:.0f})"
        ax.scatter([grid_optimum[0]], [grid_optimum[1]], color='yellow', marker='s',
                   s=50, zorder=7, edgecolors='black', linewidths=1.0, label=grid_label)


    def plot_single_map(ax, U, V, data, trajectory, scipy_result_2d, global_result_2d, grid_optimum,
                        title, cmap, label, zoom_bounds=None, precision=2, vmin=None, vmax=None):
        """Plot a single map (cost or gradient magnitude) with trajectory."""
        if vmin is None or vmax is None:
            vmin, vmax = np.percentile(data, [5, 95])

        pcm = ax.pcolormesh(U, V, data, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(pcm, ax=ax, label=label)

        plot_trajectory(ax, trajectory, zoom_bounds)
        add_optima_markers(ax, scipy_result_2d, global_result_2d, grid_optimum, precision)

        if zoom_bounds:
            ax.set_xlim(zoom_bounds[0], zoom_bounds[1])
            ax.set_ylim(zoom_bounds[2], zoom_bounds[3])

        ax.set_xlabel("u direction")
        ax.set_ylabel("v direction")
        ax.set_title(title, fontsize=9)
        ax.set_aspect('equal')
        ax.legend(loc='best', fontsize=6)


    def plot_zoom_row(axes, data_dict, trajectory_num, trajectory_ana, scipy_result_num_2d, scipy_result_ana_2d,
                      global_result_2d, grid_optimum, zoom_label, precision=2):
        """Plot one row showing: cost+numeric grad walk, numeric gradient, cost+analytic grad walk, analytic gradient, difference."""
        U = data_dict['U']
        V = data_dict['V']
        C = data_dict['C']
        grad_u_num = data_dict['grad_u_num']
        grad_v_num = data_dict['grad_v_num']
        grad_u_ana = data_dict['grad_u_ana']
        grad_v_ana = data_dict['grad_v_ana']
        bounds = data_dict['bounds']

        # Compute gradient magnitudes
        grad_norm_num = np.sqrt(grad_u_num**2 + grad_v_num**2)
        grad_norm_ana = np.sqrt(grad_u_ana**2 + grad_v_ana**2)
        grad_diff = np.abs(grad_norm_ana - grad_norm_num)

        # Use consistent color scale for gradients
        all_grad_data = np.concatenate([grad_norm_num.flatten(), grad_norm_ana.flatten()])
        gmin, gmax = np.percentile(all_grad_data, [5, 95])

        # Column 1: Cost landscape with numerical gradient walk
        plot_single_map(axes[0], U, V, C, trajectory_num, scipy_result_num_2d, global_result_2d, grid_optimum,
                        f"Cost + Numerical Walk (zoom {zoom_label})", "viridis", "cost",
                        bounds, precision)

        # Column 2: Numerical gradient with arrows
        pcm = axes[1].pcolormesh(U, V, grad_norm_num, shading="auto", cmap="plasma", vmin=gmin, vmax=gmax)
        plt.colorbar(pcm, ax=axes[1], label="gradient magnitude")

        # Add gradient arrows
        step = max(1, len(U) // ARROW_SUBSAMPLE)
        grad_mag = np.sqrt(grad_u_num[::step, ::step]**2 + grad_v_num[::step, ::step]**2)
        grad_mag = np.where(grad_mag == 0, 1, grad_mag)
        grad_u_norm = grad_u_num[::step, ::step] / grad_mag
        grad_v_norm = grad_v_num[::step, ::step] / grad_mag
        axes[1].quiver(U[::step, ::step], V[::step, ::step],
                       grad_u_norm, grad_v_norm,
                       alpha=ARROW_ALPHA, color='black')

        plot_trajectory(axes[1], trajectory_num, bounds)
        add_optima_markers(axes[1], scipy_result_num_2d, global_result_2d, grid_optimum, precision)
        if bounds:
            axes[1].set_xlim(bounds[0], bounds[1])
            axes[1].set_ylim(bounds[2], bounds[3])
        axes[1].set_xlabel("u direction")
        axes[1].set_ylabel("v direction")
        axes[1].set_title(f"Gradient Numerical + Numerical Walk (zoom {zoom_label})", fontsize=9)
        axes[1].set_aspect('equal')
        axes[1].legend(loc='best', fontsize=6)

        # Column 3: Cost landscape with analytical gradient walk
        plot_single_map(axes[2], U, V, C, trajectory_ana, scipy_result_ana_2d, global_result_2d, grid_optimum,
                        f"Cost + Analytical Walk (zoom {zoom_label})", "viridis", "cost",
                        bounds, precision)

        # Column 4: Analytical gradient with arrows
        pcm = axes[3].pcolormesh(U, V, grad_norm_ana, shading="auto", cmap="plasma", vmin=gmin, vmax=gmax)
        plt.colorbar(pcm, ax=axes[3], label="gradient magnitude")

        grad_mag_ana = np.sqrt(grad_u_ana[::step, ::step]**2 + grad_v_ana[::step, ::step]**2)
        grad_mag_ana = np.where(grad_mag_ana == 0, 1, grad_mag_ana)
        grad_u_ana_norm = grad_u_ana[::step, ::step] / grad_mag_ana
        grad_v_ana_norm = grad_v_ana[::step, ::step] / grad_mag_ana
        axes[3].quiver(U[::step, ::step], V[::step, ::step],
                       grad_u_ana_norm, grad_v_ana_norm,
                       alpha=ARROW_ALPHA, color='black')

        plot_trajectory(axes[3], trajectory_ana, bounds)
        add_optima_markers(axes[3], scipy_result_ana_2d, global_result_2d, grid_optimum, precision)
        if bounds:
            axes[3].set_xlim(bounds[0], bounds[1])
            axes[3].set_ylim(bounds[2], bounds[3])
        axes[3].set_xlabel("u direction")
        axes[3].set_ylabel("v direction")
        axes[3].set_title(f"Gradient Analytical + Analytical Walk (zoom {zoom_label})", fontsize=9)
        axes[3].set_aspect('equal')
        axes[3].legend(loc='best', fontsize=6)

        # Column 5: Difference
        plot_single_map(axes[4], U, V, grad_diff, trajectory_ana, scipy_result_ana_2d, global_result_2d, grid_optimum,
                        f"Gradient Difference (zoom {zoom_label})", "hot", "difference",
                        bounds, precision)


    def create_visualization_for_slice(solver, slice_info, full_data, zoom_data_list, zoom_levels,
                                      result_num_nd, trajectory_num_nd, result_ana_nd, trajectory_ana_nd,
                                      best_result_nd, method_name, run_num=1):
        """Create and save visualization for a specific slice and method."""
        axis1_idx, axis2_idx, u, v, slice_label = slice_info

        # Project N-D results to 2D
        center = best_result_nd.x

        # Project trajectories
        traj_num_2d = project_trajectory_to_slice(trajectory_num_nd, center, u, v)
        traj_ana_2d = project_trajectory_to_slice(trajectory_ana_nd, center, u, v)

        # Project scipy results
        scipy_num_2d = project_trajectory_to_slice(result_num_nd.x.reshape(1, -1), center, u, v)[0]
        scipy_ana_2d = project_trajectory_to_slice(result_ana_nd.x.reshape(1, -1), center, u, v)[0]
        scipy_num_2d_result = (scipy_num_2d[0], scipy_num_2d[1], result_num_nd.fun)
        scipy_ana_2d_result = (scipy_ana_2d[0], scipy_ana_2d[1], result_ana_nd.fun)

        # Global result is at origin (0, 0) since we center on it
        global_2d_result = (0.0, 0.0, best_result_nd.fun)

        # Create figure with rows for different zoom levels
        n_rows = 1 + len(zoom_levels)
        fig = plt.figure(figsize=FIGURE_SIZE)

        # Add slice information to title
        if axis1_idx is not None:
            slice_title = f"{slice_label} (axes {axis1_idx+1}, {axis2_idx+1})"
        else:
            slice_title = slice_label
        fig.suptitle(f"Method: {method_name} | Slice: {slice_title}", fontsize=12, y=0.995)

        # Row 1: Full view
        axes_row1 = [plt.subplot(n_rows, 5, i+1) for i in range(5)]
        plot_zoom_row(axes_row1, full_data, traj_num_2d, traj_ana_2d,
                     scipy_num_2d_result, scipy_ana_2d_result,
                     global_2d_result, full_data['grid_optimum'], "1x", precision=1)

        # Remaining rows: Zoom levels
        for i, (zoom, data) in enumerate(zip(zoom_levels, zoom_data_list)):
            row = i + 2
            axes_row = [plt.subplot(n_rows, 5, 5*(row-1) + j + 1) for j in range(5)]
            precision = 2 if zoom <= 100 else (3 if zoom <= 1000 else (4 if zoom <= 10000 else 5))
            plot_zoom_row(axes_row, data, traj_num_2d, traj_ana_2d,
                         scipy_num_2d_result, scipy_ana_2d_result,
                         global_2d_result, data['grid_optimum'], f"{zoom}x", precision)

        plt.tight_layout()

        # Filename includes slice label
        if NUM_RUNS > 1:
            filename = f"cost_map_Nd_{method_name}_{slice_label}_run{run_num:03d}.png"
        else:
            filename = f"cost_map_Nd_{method_name}_{slice_label}.png"

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
    print("\n" + "="*100)
    print("STATISTICS SUMMARY ACROSS ALL RUNS")
    print("="*100)

    # Group results by method
    method_stats = {}
    for run_stats in all_run_stats:
        for method_result in run_stats:
            method_name = method_result['method']
            if method_name not in method_stats:
                method_stats[method_name] = {
                    'numerical': {'iterations': [], 'final_cost': [], 'success': []},
                    'analytical': {'iterations': [], 'final_cost': [], 'success': []},
                    'is_gradient_free': method_result['is_gradient_free']
                }

            if not method_result['is_gradient_free']:
                method_stats[method_name]['numerical']['iterations'].append(method_result['num_iterations'])
                method_stats[method_name]['numerical']['final_cost'].append(method_result['num_final_cost'])
                method_stats[method_name]['numerical']['success'].append(method_result['num_success'])

            method_stats[method_name]['analytical']['iterations'].append(method_result['ana_iterations'])
            method_stats[method_name]['analytical']['final_cost'].append(method_result['ana_final_cost'])
            method_stats[method_name]['analytical']['success'].append(method_result['ana_success'])

    # Per-run best cost
    n_runs = len(all_run_stats)
    per_run_best = []
    for run_stats in all_run_stats:
        costs = []
        for r in run_stats:
            costs.append(r['ana_final_cost'])
            if not r['is_gradient_free'] and r['num_final_cost'] is not None:
                costs.append(r['num_final_cost'])
        per_run_best.append(min(costs))

    # Count wins
    win_counts = {(m, v): 0
                  for m in method_stats
                  for v in (['ana'] if method_stats[m]['is_gradient_free'] else ['num', 'ana'])}
    for run_idx, run_stats in enumerate(all_run_stats):
        best = per_run_best[run_idx]
        for r in run_stats:
            mn = r['method']
            if abs(r['ana_final_cost'] - best) < 1e-9:
                win_counts[(mn, 'ana')] += 1
            if not r['is_gradient_free'] and r['num_final_cost'] is not None:
                if abs(r['num_final_cost'] - best) < 1e-9:
                    win_counts[(mn, 'num')] += 1

    # Print summary for each method
    for method_name, stats in method_stats.items():
        print(f"\n{method_name}:")
        print("-" * 100)

        if stats['is_gradient_free']:
            iters = stats['analytical']['iterations']
            costs = stats['analytical']['final_cost']
            success_rate = 100 * sum(stats['analytical']['success']) / len(stats['analytical']['success'])
            excess = [c - b for c, b in zip(costs, per_run_best)]
            wins = win_counts[(method_name, 'ana')]

            print(f"  Iterations:  mean={np.mean(iters):.1f}, std={np.std(iters):.1f}, min={np.min(iters)}, max={np.max(iters)}")
            print(f"  Final cost:  mean={np.mean(costs):.2f}, std={np.std(costs):.2f}, min={np.min(costs):.2f}, max={np.max(costs):.2f}")
            print(f"  Excess vs best: mean={np.mean(excess):.2f}, std={np.std(excess):.2f}, min={np.min(excess):.2f}, max={np.max(excess):.2f}")
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Times best: {wins}/{n_runs}")
        else:
            print("  NUMERICAL gradients:")
            num_iters = stats['numerical']['iterations']
            num_costs = stats['numerical']['final_cost']
            num_success_rate = 100 * sum(stats['numerical']['success']) / len(stats['numerical']['success'])
            num_excess = [c - b for c, b in zip(num_costs, per_run_best)]
            num_wins = win_counts[(method_name, 'num')]

            print(f"    Iterations:  mean={np.mean(num_iters):.1f}, std={np.std(num_iters):.1f}, min={np.min(num_iters)}, max={np.max(num_iters)}")
            print(f"    Final cost:  mean={np.mean(num_costs):.2f}, std={np.std(num_costs):.2f}, min={np.min(num_costs):.2f}, max={np.max(num_costs):.2f}")
            print(f"    Excess vs best: mean={np.mean(num_excess):.2f}, std={np.std(num_excess):.2f}, min={np.min(num_excess):.2f}, max={np.max(num_excess):.2f}")
            print(f"    Success rate: {num_success_rate:.1f}%")
            print(f"    Times best: {num_wins}/{n_runs}")

            print("  ANALYTICAL gradients:")
            ana_iters = stats['analytical']['iterations']
            ana_costs = stats['analytical']['final_cost']
            ana_success_rate = 100 * sum(stats['analytical']['success']) / len(stats['analytical']['success'])
            ana_excess = [c - b for c, b in zip(ana_costs, per_run_best)]
            ana_wins = win_counts[(method_name, 'ana')]

            print(f"    Iterations:  mean={np.mean(ana_iters):.1f}, std={np.std(ana_iters):.1f}, min={np.min(ana_iters)}, max={np.max(ana_iters)}")
            print(f"    Final cost:  mean={np.mean(ana_costs):.2f}, std={np.std(ana_costs):.2f}, min={np.min(ana_costs):.2f}, max={np.max(ana_costs):.2f}")
            print(f"    Excess vs best: mean={np.mean(ana_excess):.2f}, std={np.std(ana_excess):.2f}, min={np.min(ana_excess):.2f}, max={np.max(ana_excess):.2f}")
            print(f"    Success rate: {ana_success_rate:.1f}%")
            print(f"    Times best: {ana_wins}/{n_runs}")

    print("\n" + "="*100)


def run_single_case(run_num):
    """Run optimization for all methods on a single random N-D spectrum."""
    out = []
    p = out.append

    p(f"\n{'='*100}")
    p(f"RUN {run_num}/{NUM_RUNS}")
    p(f"{'='*100}")

    # Generate N spectra for this run
    spectra = []
    E, T1, T2 = generate_random_spectra()  # Generate first two
    spectra.append(T1)
    spectra.append(T2)

    # Generate additional spectra if N > 2
    for _ in range(N_SPECTRA - 2):
        _, _, T_extra = generate_random_spectra()
        spectra.append(T_extra)

    solver = DeconvSolver(
        empirical_spectrum=E,
        theoretical_spectra=spectra,
        distance=DISTANCE_METRIC,
        max_distance=MAX_DISTANCE,
        trash_cost=TRASH_COST,
        scale_factor=SCALE_FACTOR,
    )

    if ENABLE_AUTO_SCALE:
        p(f"Auto-computed scale factor: {solver.scale_factor:.4e}")
    else:
        p(f"Using manual scale factor: {solver.scale_factor:.4e}")

    start_point = np.array(START_POINT)

    # Find global optimum in N-D space
    p(f"\nSearching for global optimum in {N_DIMS}-D space with {GLOBAL_SEARCH_STARTS} random starts...")
    best_result = find_global_optimum(solver, BOUNDS, n_starts=GLOBAL_SEARCH_STARTS)
    p(f"  Global optimum: {best_result.x}, cost={best_result.fun:.2f}")

    # Generate slice directions
    slice_directions = generate_slice_directions(N_DIMS, N_PRINCIPAL_SLICES, N_RANDOM_SLICES,
                                                 random_state=RANDOM_SEED)
    p(f"\nGenerated {len(slice_directions)} slices ({N_PRINCIPAL_SLICES} principal + {N_RANDOM_SLICES} random)")

    run_stats = []

    # Loop over each optimization method
    for method_name, display_name in tqdm(METHODS, desc=f"Run {run_num}: Methods", leave=False):
        p(f"\n{'-'*100}")
        p(f"TESTING METHOD: {display_name} ({method_name})")
        p(f"{'-'*100}")

        is_gradient_free = method_name in GRADIENT_FREE_METHODS

        if is_gradient_free:
            p(f"\nRunning {display_name} (gradient-free) from starting point")
            with tqdm(total=1, desc=f"  {method_name} optimization", leave=False) as pbar:
                result_ana, trajectory_ana = run_optimization(
                    solver, start_point, BOUNDS, method=method_name, use_numerical_grad=False
                )
                pbar.update(1)

            p(f"  Converged: {result_ana.success}")
            p(f"  Final point: {result_ana.x}, cost={result_ana.fun:.2f}")
            p(f"  Iterations: {len(trajectory_ana)}")

            result_num = result_ana
            trajectory_num = trajectory_ana

            run_stats.append({
                'method': method_name,
                'is_gradient_free': True,
                'ana_iterations': len(trajectory_ana),
                'ana_final_cost': result_ana.fun,
                'ana_success': result_ana.success,
                'num_iterations': None,
                'num_final_cost': None,
                'num_success': None
            })

        else:
            # Run with numerical gradients
            p(f"\nRunning {display_name} with NUMERICAL gradients")
            with tqdm(total=1, desc=f"  {method_name} (numerical)", leave=False) as pbar:
                result_num, trajectory_num = run_optimization(
                    solver, start_point, BOUNDS, method=method_name, use_numerical_grad=True
                )
                pbar.update(1)

            p(f"  Converged: {result_num.success}")
            p(f"  Final point: {result_num.x}, cost={result_num.fun:.2f}")
            p(f"  Iterations: {len(trajectory_num)}")

            # Run with analytical gradients
            p(f"\nRunning {display_name} with ANALYTICAL gradients")
            with tqdm(total=1, desc=f"  {method_name} (analytical)", leave=False) as pbar:
                result_ana, trajectory_ana = run_optimization(
                    solver, start_point, BOUNDS, method=method_name, use_numerical_grad=False
                )
                pbar.update(1)

            p(f"  Converged: {result_ana.success}")
            p(f"  Final point: {result_ana.x}, cost={result_ana.fun:.2f}")
            p(f"  Iterations: {len(trajectory_ana)}")

            run_stats.append({
                'method': method_name,
                'is_gradient_free': False,
                'num_iterations': len(trajectory_num),
                'num_final_cost': result_num.fun,
                'num_success': result_num.success,
                'ana_iterations': len(trajectory_ana),
                'ana_final_cost': result_ana.fun,
                'ana_success': result_ana.success
            })

        # Generate visualizations for each slice
        if ENABLE_PLOTTING:
            for slice_info in tqdm(slice_directions, desc=f"  {method_name} slices", leave=False):
                axis1_idx, axis2_idx, u, v, slice_label = slice_info

                # Determine bounds in 2D slice space
                # Use a range that covers the original bounds projected onto the slice
                max_range = max([b[1] - b[0] for b in BOUNDS])
                bounds_2d = ((-max_range/2, max_range/2), (-max_range/2, max_range/2))

                # Compute full grid for this slice
                U, V, C, u_1d, v_1d, grad_u_ana, grad_v_ana = compute_cost_on_2d_slice(
                    solver, best_result.x, u, v, bounds_2d, GRID_RESOLUTION
                )
                grad_v_num, grad_u_num = np.gradient(C, v_1d, u_1d)

                # Find grid minimum
                min_idx = np.unravel_index(np.argmin(C), C.shape)
                grid_optimum = (U[min_idx], V[min_idx], C[min_idx])

                full_data = {
                    'U': U, 'V': V, 'C': C,
                    'grad_u_num': grad_u_num, 'grad_v_num': grad_v_num,
                    'grad_u_ana': grad_u_ana, 'grad_v_ana': grad_v_ana,
                    'bounds': None,
                    'grid_optimum': grid_optimum
                }

                # Compute zoom levels
                zoom_data_list = []
                for zoom in ZOOM_LEVELS:
                    zoom_width = (u_1d[-1] - u_1d[0]) / zoom

                    u_min = -zoom_width/2
                    u_max = zoom_width/2
                    v_min = -zoom_width/2
                    v_max = zoom_width/2

                    bounds_zoom = ((u_min, u_max), (v_min, v_max))

                    U_z, V_z, C_z, u_1d_z, v_1d_z, grad_u_ana_z, grad_v_ana_z = compute_cost_on_2d_slice(
                        solver, best_result.x, u, v, bounds_zoom, GRID_RESOLUTION
                    )
                    grad_v_num_z, grad_u_num_z = np.gradient(C_z, v_1d_z, u_1d_z)

                    min_idx_zoom = np.unravel_index(np.argmin(C_z), C_z.shape)
                    grid_optimum_zoom = (U_z[min_idx_zoom], V_z[min_idx_zoom], C_z[min_idx_zoom])

                    zoom_data_list.append({
                        'U': U_z, 'V': V_z, 'C': C_z,
                        'grad_u_num': grad_u_num_z, 'grad_v_num': grad_v_num_z,
                        'grad_u_ana': grad_u_ana_z, 'grad_v_ana': grad_v_ana_z,
                        'bounds': (u_min, u_max, v_min, v_max),
                        'grid_optimum': grid_optimum_zoom
                    })

                # Create visualization
                save_msg = create_visualization_for_slice(
                    solver, slice_info, full_data, zoom_data_list, ZOOM_LEVELS,
                    result_num, trajectory_num, result_ana, trajectory_ana,
                    best_result, method_name, run_num
                )
                p(save_msg)

    return run_num, run_stats, '\n'.join(out)


def main():
    """Main entry point."""
    import os

    # Create output directory
    if ENABLE_PLOTTING and OUT_DIR != '.':
        os.makedirs(OUT_DIR, exist_ok=True)

    # Set random seed
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)
        print(f"Random seed set to: {RANDOM_SEED}")

    print("="*100)
    print("COST MAP N-D - N-DIMENSIONAL OPTIMIZATION WITH 2D SLICES")
    print("="*100)
    print(f"Configuration:")
    print(f"  Number of spectra (dimensions): {N_SPECTRA}")
    print(f"  Principal axis slices: {N_PRINCIPAL_SLICES}")
    print(f"  Random orientation slices: {N_RANDOM_SLICES}")
    print(f"  Total slices per method: {N_PRINCIPAL_SLICES + N_RANDOM_SLICES}")
    print(f"  Plotting enabled: {ENABLE_PLOTTING}")
    print(f"  Number of runs: {NUM_RUNS}")
    print(f"  Methods to test: {len(METHODS)}")
    print(f"  Grid resolution: {GRID_RESOLUTION}")
    print(f"  Zoom levels: {ZOOM_LEVELS}")
    print(f"  Random seed: {RANDOM_SEED if RANDOM_SEED is not None else 'None (random)'}")
    print(f"  Auto-scaling enabled: {ENABLE_AUTO_SCALE}")
    if not ENABLE_AUTO_SCALE and SCALE_FACTOR is not None:
        print(f"  Manual scale factor: {SCALE_FACTOR}")
    if ENABLE_PLOTTING:
        print(f"  Output directory: {os.path.abspath(OUT_DIR)}")

    # Run all cases
    run_nums = list(range(1, NUM_RUNS + 1))
    if NUM_RUNS > 1 and _args.workers != 1:
        n_workers = _args.workers
        print(f"  Workers: {n_workers or multiprocessing.cpu_count()} (multiprocessing)")
        ordered_stats = {}
        with multiprocessing.Pool(processes=n_workers) as pool:
            for run_num, run_stats, output in tqdm(pool.imap_unordered(run_single_case, run_nums),
                                                     total=NUM_RUNS, desc="Overall progress"):
                print(output)
                ordered_stats[run_num] = run_stats
        all_run_stats = [ordered_stats[n] for n in run_nums]
    else:
        all_run_stats = []
        for _, run_stats, output in tqdm((run_single_case(n) for n in run_nums),
                                          total=NUM_RUNS, desc="Overall progress"):
            print(output)
            all_run_stats.append(run_stats)

    # Print summary statistics if multiple runs
    if NUM_RUNS > 1:
        print_statistics_summary(all_run_stats)

    print("\n" + "="*100)
    print("COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
