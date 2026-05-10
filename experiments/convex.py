"""Test whether the optimization basin is convex across multiple dimensions."""

import numpy as np
from scipy.optimize import minimize
from wnetdeconv import DeconvSolver, Spectrum_1D
from wnet.distances import DistanceMetric


def generate_random_spectra(n_spectra):
    """Generate random overlapping spectra for n_spectra theoretical spectra."""
    n_peaks_E = 15
    n_peaks_T = 10

    pos_E = np.sort(np.random.uniform(1, 100, n_peaks_E))
    int_E = np.random.uniform(5, 50, n_peaks_E)
    E = Spectrum_1D(pos_E.tolist(), int_E.tolist())

    theoreticals = []
    for i in range(n_spectra):
        pos_T = np.sort(np.random.uniform(1, 100, n_peaks_T))
        int_T = np.random.uniform(2, 10, n_peaks_T)
        theoreticals.append(Spectrum_1D(pos_T.tolist(), int_T.tolist()))

    return E, theoreticals


def find_global_optimum(solver, bounds, n_starts=20):
    """Find global optimum using multiple random starts."""
    best_result = None

    for i in range(n_starts):
        random_start = np.array(
            [np.random.uniform(bounds[j][0], bounds[j][1]) for j in range(len(bounds))]
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


def test_line_segment_convexity(solver, point_a, point_b, n_samples=100):
    """
    Test if the cost function is convex along a line segment between two points.

    For convexity, cost(t*a + (1-t)*b) <= t*cost(a) + (1-t)*cost(b)
    for all t in [0, 1].

    Returns:
        is_convex: Whether the segment appears convex
        max_violation: Maximum absolute violation
        relative_violation: Maximum violation relative to cost scale
    """
    alphas = np.linspace(0, 1, n_samples)
    actual_costs = np.zeros(n_samples)

    # Compute actual costs along the line segment
    for i, alpha in enumerate(alphas):
        point = alpha * np.array(point_a) + (1 - alpha) * np.array(point_b)
        solver.set_point(point.tolist())
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
    # For a discrete cost function, violations < 0.1% are likely just discretization
    is_convex = max_violation <= 1.0 or relative_violation <= 1e-3

    return is_convex, max_violation, relative_violation


def test_dimension(n_dim, n_cases=10):
    """Test convexity for a given number of dimensions across multiple random cases."""
    print(f"\n{'='*80}")
    print(f"Testing {n_dim}-dimensional case ({n_dim} theoretical spectra)")
    print(f"{'='*80}")

    total_tests = 0
    total_convex = 0
    total_violations = 0
    max_rel_violation_seen = 0.0

    for case_num in range(n_cases):
        print(f"\nCase {case_num+1}/{n_cases}:")

        # Generate spectra
        E, theoreticals = generate_random_spectra(n_dim)

        solver = DeconvSolver(
            empirical_spectrum=E,
            theoretical_spectra=theoreticals,
            distance=DistanceMetric.LINF,
            max_distance=10,
            trash_cost=100,
            scale_factor=1000,
        )

        bounds = [(0, 25) for _ in range(n_dim)]

        # Find global optimum
        global_opt = find_global_optimum(solver, bounds, n_starts=30)
        if global_opt is None:
            print(f"  Failed to find optimum, skipping case")
            continue
        print(f"  Global optimum: cost={global_opt.fun:.2f}")

        # Test different line segments
        n_random_to_opt = 10
        n_random_pairs = 20

        case_convex = 0
        case_total = 0

        # Random points to global optimum
        for i in range(n_random_to_opt):
            random_point = [np.random.uniform(0, 25) for _ in range(n_dim)]
            is_convex, max_viol, rel_viol = test_line_segment_convexity(
                solver, random_point, global_opt.x.tolist()
            )
            case_total += 1
            if is_convex:
                case_convex += 1
            else:
                total_violations += 1
            max_rel_violation_seen = max(max_rel_violation_seen, rel_viol)

        # Random point pairs
        for i in range(n_random_pairs):
            point_a = [np.random.uniform(0, 25) for _ in range(n_dim)]
            point_b = [np.random.uniform(0, 25) for _ in range(n_dim)]
            is_convex, max_viol, rel_viol = test_line_segment_convexity(
                solver, point_a, point_b
            )
            case_total += 1
            if is_convex:
                case_convex += 1
            else:
                total_violations += 1
            max_rel_violation_seen = max(max_rel_violation_seen, rel_viol)

        total_tests += case_total
        total_convex += case_convex

        print(f"  Convex segments: {case_convex}/{case_total}")

    print(f"\n{'-'*80}")
    print(f"Summary for {n_dim}D:")
    print(f"  Total segments tested: {total_tests}")
    print(
        f"  Convex: {total_convex}/{total_tests} ({100*total_convex/total_tests:.1f}%)"
    )
    print(f"  Non-convex violations: {total_violations}")
    print(f"  Max relative violation: {max_rel_violation_seen:.2e}")

    return total_tests, total_convex, total_violations, max_rel_violation_seen


def main():
    print("=" * 80)
    print("CONVEXITY ANALYSIS ACROSS DIMENSIONS")
    print("=" * 80)

    # Test dimensions from 2 to 10
    dimensions = [2, 3, 4, 5, 7, 10]
    n_cases_per_dim = 5

    all_results = []

    for n_dim in dimensions:
        total, convex, violations, max_viol = test_dimension(
            n_dim, n_cases=n_cases_per_dim
        )
        all_results.append(
            {
                "dim": n_dim,
                "total": total,
                "convex": convex,
                "violations": violations,
                "max_viol": max_viol,
            }
        )

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(
        f"{'Dim':<6} {'Tests':<8} {'Convex':<10} {'Rate':<10} {'Violations':<12} {'Max Rel Viol':<15}"
    )
    print("-" * 80)
    for r in all_results:
        rate = 100 * r["convex"] / r["total"]
        print(
            f"{r['dim']:<6} {r['total']:<8} {r['convex']:<10} {rate:>6.1f}%    {r['violations']:<12} {r['max_viol']:<15.2e}"
        )

    grand_total = sum(r["total"] for r in all_results)
    grand_convex = sum(r["convex"] for r in all_results)
    grand_violations = sum(r["violations"] for r in all_results)
    print("-" * 80)
    print(
        f"{'TOTAL':<6} {grand_total:<8} {grand_convex:<10} {100*grand_convex/grand_total:>6.1f}%    {grand_violations:<12}"
    )


if __name__ == "__main__":
    main()
