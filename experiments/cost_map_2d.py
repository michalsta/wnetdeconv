"""2D map of (p1, p2) -> total_cost for the two-spectrum deconvolution example."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.cm import gray as gray_cmap
from scipy.optimize import minimize
from wnetdeconv import DeconvSolver, Spectrum_1D
from wnet.distances import DistanceMetric


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


def compute_cost_grid(solver, p1_range, p2_range, n_points):
    """Compute cost values and analytical gradients on a grid."""
    p1 = np.linspace(p1_range[0], p1_range[1], n_points)
    p2 = np.linspace(p2_range[0], p2_range[1], n_points)
    P1, P2 = np.meshgrid(p1, p2)
    C = np.empty_like(P1)
    Grad_p1_analytical = np.empty_like(P1)
    Grad_p2_analytical = np.empty_like(P1)

    for i in range(n_points):
        for j in range(n_points):
            solver.set_point([P1[i, j], P2[i, j]])
            C[i, j] = solver.total_cost()
            grad = solver.gradient()
            Grad_p1_analytical[i, j] = grad[0]
            Grad_p2_analytical[i, j] = grad[1]

    return P1, P2, C, p1, p2, Grad_p1_analytical, Grad_p2_analytical


def run_optimization(solver, start_point, bounds, use_numerical_grad=False):
    """Run gradient descent optimization and return trajectory."""
    trajectory = []

    def cost_function(point):
        solver.set_point(point)
        cost = solver.total_cost()
        trajectory.append(point.copy())
        return cost

    def grad_function(point):
        solver.set_point(point)
        return np.array(solver.gradient())

    if use_numerical_grad:
        # Use numerical gradients (automatic finite differences)
        # Use a larger epsilon for finite differences since cost function is discrete
        result = minimize(
            cost_function,
            start_point,
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': False, 'maxiter': 100, 'eps': 1e-2}
        )
    else:
        # Use analytical gradients
        result = minimize(
            cost_function,
            start_point,
            method='L-BFGS-B',
            jac=grad_function,
            bounds=bounds,
            options={'disp': False, 'maxiter': 100}
        )

    return result, np.array(trajectory)


def find_global_optimum(solver, bounds, n_starts=20):
    """Find global optimum using multiple random starts."""
    best_result = None

    for i in range(n_starts):
        random_start = np.array([
            np.random.uniform(bounds[0][0], bounds[0][1]),
            np.random.uniform(bounds[1][0], bounds[1][1])
        ])

        def temp_cost(point):
            solver.set_point(point)
            return solver.total_cost()

        def temp_grad(point):
            solver.set_point(point)
            return np.array(solver.gradient())

        try:
            temp_result = minimize(
                temp_cost, random_start, method='L-BFGS-B',
                jac=temp_grad, bounds=bounds,
                options={'disp': False, 'maxiter': 100}
            )
            if best_result is None or temp_result.fun < best_result.fun:
                best_result = temp_result
        except:
            pass

    return best_result


def plot_trajectory(ax, trajectory, zoom_bounds=None):
    """Plot optimization trajectory with black-to-white gradient."""
    if zoom_bounds:
        p1_min, p1_max, p2_min, p2_max = zoom_bounds
        mask = ((trajectory[:, 0] >= p1_min) & (trajectory[:, 0] <= p1_max) &
                (trajectory[:, 1] >= p2_min) & (trajectory[:, 1] <= p2_max))
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


def add_optima_markers(ax, scipy_result, global_result, grid_optimum, precision=2):
    """Add three optima markers with costs in legend."""
    fmt = f"{{:.{precision}f}}"

    # Scipy endpoint
    scipy_label = f"scipy end ({fmt.format(scipy_result.x[0])}, {fmt.format(scipy_result.x[1])}, cost={scipy_result.fun:.0f})"
    ax.scatter([scipy_result.x[0]], [scipy_result.x[1]], color='cyan', marker='X',
               s=50, zorder=7, edgecolors='black', linewidths=1.0, label=scipy_label)

    # Global optimum
    global_label = f"global opt ({fmt.format(global_result.x[0])}, {fmt.format(global_result.x[1])}, cost={global_result.fun:.0f})"
    ax.scatter([global_result.x[0]], [global_result.x[1]], color='magenta', marker='D',
               s=50, zorder=7, edgecolors='white', linewidths=1.0, label=global_label)

    # Grid minimum
    grid_label = f"grid min ({fmt.format(grid_optimum[0])}, {fmt.format(grid_optimum[1])}, cost={grid_optimum[2]:.0f})"
    ax.scatter([grid_optimum[0]], [grid_optimum[1]], color='yellow', marker='s',
               s=50, zorder=7, edgecolors='black', linewidths=1.0, label=grid_label)


def plot_single_map(ax, P1, P2, data, trajectory, scipy_result, global_result, grid_optimum,
                    title, cmap, label, zoom_bounds=None, precision=2, vmin=None, vmax=None):
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

    ax.set_xlabel("p1")
    ax.set_ylabel("p2")
    ax.set_title(title, fontsize=9)
    ax.set_aspect('equal')
    ax.legend(loc='best', fontsize=6)


def plot_zoom_row(axes, data_dict, trajectory_num, trajectory_ana, scipy_result_num, scipy_result_ana,
                  global_result, grid_optimum, zoom_label, precision=2):
    """Plot one row showing: cost+numeric grad walk, numeric gradient, cost+analytic grad walk, analytic gradient, difference."""
    P1 = data_dict['P1']
    P2 = data_dict['P2']
    C = data_dict['C']
    grad_p1_num = data_dict['grad_p1_num']
    grad_p2_num = data_dict['grad_p2_num']
    grad_p1_ana = data_dict['grad_p1_ana']
    grad_p2_ana = data_dict['grad_p2_ana']
    bounds = data_dict['bounds']

    # Compute gradient magnitudes
    grad_norm_num = np.sqrt(grad_p1_num**2 + grad_p2_num**2)
    grad_norm_ana = np.sqrt(grad_p1_ana**2 + grad_p2_ana**2)
    grad_diff = np.abs(grad_norm_ana - grad_norm_num)

    # Use consistent color scale for gradients
    all_grad_data = np.concatenate([grad_norm_num.flatten(), grad_norm_ana.flatten()])
    gmin, gmax = np.percentile(all_grad_data, [5, 95])

    # Column 1: Cost landscape with numerical gradient walk
    plot_single_map(axes[0], P1, P2, C, trajectory_num, scipy_result_num, global_result, grid_optimum,
                    f"Cost + Num Walk ({zoom_label})", "viridis", "cost",
                    bounds, precision)

    # Column 2: Numerical gradient with arrows and numerical walk
    vmin, vmax = np.percentile(grad_norm_num, [5, 95])
    pcm = axes[1].pcolormesh(P1, P2, grad_norm_num, shading="auto", cmap="plasma", vmin=gmin, vmax=gmax)
    plt.colorbar(pcm, ax=axes[1], label="grad mag")

    # Add gradient arrows (normalized to same length)
    step = max(1, len(P1) // 20)
    # Normalize gradients to unit vectors
    grad_mag = np.sqrt(grad_p1_num[::step, ::step]**2 + grad_p2_num[::step, ::step]**2)
    grad_mag = np.where(grad_mag == 0, 1, grad_mag)  # Avoid division by zero
    grad_p1_norm = grad_p1_num[::step, ::step] / grad_mag
    grad_p2_norm = grad_p2_num[::step, ::step] / grad_mag
    axes[1].quiver(P1[::step, ::step], P2[::step, ::step],
                   grad_p1_norm, grad_p2_norm,
                   alpha=0.6, color='black')

    plot_trajectory(axes[1], trajectory_num, bounds)
    add_optima_markers(axes[1], scipy_result_num, global_result, grid_optimum, precision)
    if bounds:
        axes[1].set_xlim(bounds[0], bounds[1])
        axes[1].set_ylim(bounds[2], bounds[3])
    axes[1].set_xlabel("p1")
    axes[1].set_ylabel("p2")
    axes[1].set_title(f"Grad Num + Num Walk ({zoom_label})", fontsize=9)
    axes[1].set_aspect('equal')
    axes[1].legend(loc='best', fontsize=6)

    # Column 3: Cost landscape with analytical gradient walk
    plot_single_map(axes[2], P1, P2, C, trajectory_ana, scipy_result_ana, global_result, grid_optimum,
                    f"Cost + Ana Walk ({zoom_label})", "viridis", "cost",
                    bounds, precision)

    # Column 4: Analytical gradient with arrows and analytical walk
    pcm = axes[3].pcolormesh(P1, P2, grad_norm_ana, shading="auto", cmap="plasma", vmin=gmin, vmax=gmax)
    plt.colorbar(pcm, ax=axes[3], label="grad mag")

    # Add gradient arrows (normalized to same length)
    # Normalize gradients to unit vectors
    grad_mag_ana = np.sqrt(grad_p1_ana[::step, ::step]**2 + grad_p2_ana[::step, ::step]**2)
    grad_mag_ana = np.where(grad_mag_ana == 0, 1, grad_mag_ana)  # Avoid division by zero
    grad_p1_ana_norm = grad_p1_ana[::step, ::step] / grad_mag_ana
    grad_p2_ana_norm = grad_p2_ana[::step, ::step] / grad_mag_ana
    axes[3].quiver(P1[::step, ::step], P2[::step, ::step],
                   grad_p1_ana_norm, grad_p2_ana_norm,
                   alpha=0.6, color='black')

    plot_trajectory(axes[3], trajectory_ana, bounds)
    add_optima_markers(axes[3], scipy_result_ana, global_result, grid_optimum, precision)
    if bounds:
        axes[3].set_xlim(bounds[0], bounds[1])
        axes[3].set_ylim(bounds[2], bounds[3])
    axes[3].set_xlabel("p1")
    axes[3].set_ylabel("p2")
    axes[3].set_title(f"Grad Ana + Ana Walk ({zoom_label})", fontsize=9)
    axes[3].set_aspect('equal')
    axes[3].legend(loc='best', fontsize=6)

    # Column 5: Difference
    plot_single_map(axes[4], P1, P2, grad_diff, trajectory_ana, scipy_result_ana, global_result, grid_optimum,
                    f"Difference ({zoom_label})", "hot", "diff",
                    bounds, precision)


def main():
    # Generate spectra
    E, T1, T2 = generate_random_spectra()

    solver = DeconvSolver(
        empirical_spectrum=E,
        theoretical_spectra=[T1, T2],
        distance=DistanceMetric.LINF,
        max_distance=10,
        trash_cost=100,
        scale_factor=1000,
    )

    # Compute full grid
    print("\nComputing full grid...")
    P1, P2, C, p1, p2, grad_p1_ana, grad_p2_ana = compute_cost_grid(solver, (0, 25), (0, 25), 200)
    grad_p2_num, grad_p1_num = np.gradient(C, p2, p1)

    # Store grid results
    min_idx = np.unravel_index(np.argmin(C), C.shape)
    grid_results = [(P1[min_idx], P2[min_idx], C[min_idx])]

    # Run optimization with numerical gradients
    print("\nRunning gradient descent with NUMERICAL gradients from starting point: p1=15.00, p2=20.00")
    start_point = np.array([15.0, 20.0])
    bounds = [(0, 25), (0, 25)]
    result_num, trajectory_num = run_optimization(solver, start_point, bounds, use_numerical_grad=True)

    print(f"Numerical optimization converged: {result_num.success}")
    print(f"Final point: p1={result_num.x[0]:.2f}, p2={result_num.x[1]:.2f}, cost={result_num.fun:.2f}")
    print(f"Number of iterations: {len(trajectory_num)}")

    # Run optimization with analytical gradients
    print("\nRunning gradient descent with ANALYTICAL gradients from starting point: p1=15.00, p2=20.00")
    result_ana, trajectory_ana = run_optimization(solver, start_point, bounds, use_numerical_grad=False)

    print(f"Analytical optimization converged: {result_ana.success}")
    print(f"Final point: p1={result_ana.x[0]:.2f}, p2={result_ana.x[1]:.2f}, cost={result_ana.fun:.2f}")
    print(f"Number of iterations: {len(trajectory_ana)}")

    # Find global optimum
    print("\nSearching for actual global optimum with multiple random starts...")
    best_result = find_global_optimum(solver, bounds)
    print(f"Actual optimum: p1={best_result.x[0]:.2f}, p2={best_result.x[1]:.2f}, cost={best_result.fun:.2f}")

    # Prepare full grid data
    full_data = {
        'P1': P1, 'P2': P2, 'C': C,
        'grad_p1_num': grad_p1_num, 'grad_p2_num': grad_p2_num,
        'grad_p1_ana': grad_p1_ana, 'grad_p2_ana': grad_p2_ana,
        'bounds': None
    }

    # Compute zoom grids
    zoom_levels = [10, 100, 1000, 10000, 100000]
    zoom_data_list = []

    for zoom in zoom_levels:
        print(f"\nComputing {zoom}x zoom with dense grid...")
        zoom_width = (p1[-1] - p1[0]) / zoom  # Use same width for both axes to keep square

        # Center on analytical optimum
        p1_center = result_ana.x[0]
        p2_center = result_ana.x[1]

        # Calculate initial bounds
        p1_min = p1_center - zoom_width/2
        p1_max = p1_center + zoom_width/2
        p2_min = p2_center - zoom_width/2
        p2_max = p2_center + zoom_width/2

        # Shift if hitting boundaries, maintaining width
        if p1_min < 0:
            p1_min = 0
            p1_max = zoom_width
        elif p1_max > 25:
            p1_max = 25
            p1_min = 25 - zoom_width

        if p2_min < 0:
            p2_min = 0
            p2_max = zoom_width
        elif p2_max > 25:
            p2_max = 25
            p2_min = 25 - zoom_width

        P1_z, P2_z, C_z, p1_z, p2_z, grad_p1_ana_z, grad_p2_ana_z = compute_cost_grid(
            solver, (p1_min, p1_max), (p2_min, p2_max), 200
        )
        grad_p2_num_z, grad_p1_num_z = np.gradient(C_z, p2_z, p1_z)

        # Track grid minimum
        min_idx_zoom = np.unravel_index(np.argmin(C_z), C_z.shape)
        grid_results.append((P1_z[min_idx_zoom], P2_z[min_idx_zoom], C_z[min_idx_zoom]))

        zoom_data_list.append({
            'P1': P1_z, 'P2': P2_z, 'C': C_z,
            'grad_p1_num': grad_p1_num_z, 'grad_p2_num': grad_p2_num_z,
            'grad_p1_ana': grad_p1_ana_z, 'grad_p2_ana': grad_p2_ana_z,
            'bounds': (p1_min, p1_max, p2_min, p2_max)
        })

    # Find best grid minimum
    best_grid_idx = np.argmin([cost for _, _, cost in grid_results])
    opt_p1, opt_p2, opt_cost = grid_results[best_grid_idx]
    grid_optimum = (opt_p1, opt_p2, opt_cost)
    print(f"\nGrid optimum from all levels: p1={opt_p1:.3f}, p2={opt_p2:.3f}, cost={opt_cost:.2f}")
    print(f"  (found at zoom level: {['1x', '10x', '100x', '1000x', '10000x', '100000x'][best_grid_idx]})")

    # Create figure with 6 rows x 5 columns
    fig = plt.figure(figsize=(30, 30))

    # Row 1: Full view (1x)
    axes_row1 = [plt.subplot(6, 5, i+1) for i in range(5)]
    plot_zoom_row(axes_row1, full_data, trajectory_num, trajectory_ana, result_num, result_ana,
                  best_result, grid_optimum, "1x", precision=1)

    # Rows 2-6: Zoom levels
    for i, (zoom, data) in enumerate(zip(zoom_levels, zoom_data_list)):
        row = i + 2
        axes_row = [plt.subplot(6, 5, 5*(row-1) + j + 1) for j in range(5)]
        precision = 2 if zoom <= 100 else (3 if zoom <= 1000 else (4 if zoom <= 10000 else 5))
        plot_zoom_row(axes_row, data, trajectory_num, trajectory_ana, result_num, result_ana,
                      best_result, grid_optimum, f"{zoom}x", precision)

    plt.tight_layout()
    plt.savefig("cost_map_2d.png", dpi=150)
    print("\nsaved cost_map_2d.png")
    plt.show()


if __name__ == "__main__":
    main()
