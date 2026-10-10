from wnetdeconv import DeconvSolver, Spectrum_1D
from wnet.distances import DistanceMetric

E = Spectrum_1D([1, 100], [10, 30])

T1 = Spectrum_1D([1], [2])  # optimal proportion: 5

T2 = Spectrum_1D([100], [3])  # optimal proportion: 10

solver = DeconvSolver(
    empirical_spectrum=E,
    theoretical_spectra=[T1, T2],
    distance=DistanceMetric.LINF,
    max_distance=10,
    trash_cost=100,
    scale_factor=1000,
)


# Absolute cost accuracy; allow for the numerical cushion at this cost scale.
result = solver.optimize(x0=[1.0, 1.0], tol=1e-8, print_steps=True)

print(f"Optimal point: {result.x}")
print(f"Cost at optimum: {result.fun:.6f}")
print(f"Expected: [5, 10] with cost 0")
print(f"Success: {result.success}")
print(f"Message: {result.message}")
