"""Test whether the optimization basin is convex."""
import numpy as np
import matplotlib.pyplot as plt
from wnetdeconv import DeconvSolver
from wnet.distances import DistanceMetric
from experiments_support import generate_random_spectra, find_global_optimum


def test_line_segment_convexity(solver, point_a, point_b, n_samples=100):
    """
    Test if the cost function is convex along a line segment between two points.

    For convexity, cost(t*a + (1-t)*b) <= t*cost(a) + (1-t)*cost(b)
    for all t in [0, 1].

    Returns:
        alphas: Sample points along [0, 1]
        actual_costs: Actual cost values along the line segment
        linear_costs: Linear interpolation (upper bound for convex function)
        is_convex: Whether the segment appears convex
    """
    alphas = np.linspace(0, 1, n_samples)
    actual_costs = np.zeros(n_samples)

    # Compute actual costs along the line segment
    for i, alpha in enumerate(alphas):
        point = alpha * np.array(point_a) + (1 - alpha) * np.array(point_b)
        solver.set_point(point)
        actual_costs[i] = solver.total_cost()

    # Linear interpolation (convex upper bound)
    solver.set_point(point_a)
    cost_a = solver.total_cost()
    solver.set_point(point_b)
    cost_b = solver.total_cost()
    linear_costs = alphas * cost_a + (1 - alphas) * cost_b

    # Check if convex (actual should be <= linear)
    differences = actual_costs - linear_costs
    max_violation = np.max(differences)

    # Compute relative violation to distinguish numerical errors from real non-convexity
    cost_scale = max(abs(cost_a), abs(cost_b), 1.0)
    relative_violation = max_violation / cost_scale

    # Use both absolute and relative thresholds
    is_convex = max_violation <= 1e-6 or relative_violation <= 1e-10

    return alphas, actual_costs, linear_costs, is_convex, cost_a, cost_b, max_violation, relative_violation


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

    bounds = [(0, 25), (0, 25)]

    # Find global optimum
    print("\nSearching for global optimum...")
    global_opt = find_global_optimum(solver, bounds, n_starts=50)
    print(f"Global optimum: p1={global_opt.x[0]:.3f}, p2={global_opt.x[1]:.3f}, cost={global_opt.fun:.2f}")

    # Test different types of line segments
    test_cases = []

    # Case 1: Random point to global optimum
    for i in range(20):
        random_point = [
            np.random.uniform(bounds[0][0], bounds[0][1]),
            np.random.uniform(bounds[1][0], bounds[1][1])
        ]
        test_cases.append({
            'name': f'Random #{i+1} -> Global Opt',
            'point_a': random_point,
            'point_b': global_opt.x.tolist()
        })

    # Case 2: Corner points to global optimum
    corners = [
        [0, 0], [0, 25], [25, 0], [25, 25]
    ]
    for i, corner in enumerate(corners):
        test_cases.append({
            'name': f'Corner {i+1} -> Global Opt',
            'point_a': corner,
            'point_b': global_opt.x.tolist()
        })

    # Case 3: Random point pairs
    for i in range(30):
        point_a = [
            np.random.uniform(bounds[0][0], bounds[0][1]),
            np.random.uniform(bounds[1][0], bounds[1][1])
        ]
        point_b = [
            np.random.uniform(bounds[0][0], bounds[0][1]),
            np.random.uniform(bounds[1][0], bounds[1][1])
        ]
        test_cases.append({
            'name': f'Random pair #{i+1}',
            'point_a': point_a,
            'point_b': point_b
        })

    # Run tests
    results = []
    convex_count = 0

    print("\n" + "="*80)
    print("CONVEXITY TESTS")
    print("="*80)

    for test in test_cases:
        alphas, actual, linear, is_convex, cost_a, cost_b, max_violation, relative_violation = test_line_segment_convexity(
            solver, test['point_a'], test['point_b']
        )

        results.append({
            'test': test,
            'alphas': alphas,
            'actual': actual,
            'linear': linear,
            'is_convex': is_convex,
            'cost_a': cost_a,
            'cost_b': cost_b,
            'max_violation': max_violation,
            'relative_violation': relative_violation
        })

        if is_convex:
            convex_count += 1

        status = "[OK] CONVEX" if is_convex else "[!!] NON-CONVEX"

        # Determine if violation is likely numerical or real
        if not is_convex:
            if relative_violation < 1e-6:
                violation_type = "(likely numerical)"
            else:
                violation_type = "(REAL violation)"
        else:
            violation_type = ""

        print(f"{test['name']:30s} {status:20s} max_viol={max_violation:10.2f} rel_viol={relative_violation:12.2e} {violation_type}")

    print("="*80)
    print(f"Summary: {convex_count}/{len(test_cases)} segments are convex")
    print("="*80)

    # Visualize results
    n_plots = len(results)
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    for i, result in enumerate(results):
        ax = axes[i]

        # Plot actual vs linear interpolation
        ax.plot(result['alphas'], result['actual'], 'b-', linewidth=2, label='Actual cost')
        ax.plot(result['alphas'], result['linear'], 'r--', linewidth=2, label='Linear interpolation')

        # Mark endpoints
        ax.plot(0, result['cost_a'], 'go', markersize=8, label='Point A')
        ax.plot(1, result['cost_b'], 'mo', markersize=8, label='Point B')

        # Shade violations if non-convex
        if not result['is_convex']:
            violation_mask = result['actual'] > result['linear']
            ax.fill_between(result['alphas'], result['linear'], result['actual'],
                           where=violation_mask, alpha=0.3, color='red',
                           label='Convexity violation')

        status = "CONVEX" if result['is_convex'] else "NON-CONVEX"
        ax.set_title(f"{result['test']['name']}\n{status}", fontsize=10)
        ax.set_xlabel('t (0=A, 1=B)')
        ax.set_ylabel('Cost')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("convexity_test.png", dpi=150)
    print("\nsaved convexity_test.png")
    plt.show()


if __name__ == "__main__":
    main()
