from wnetdeconv import DeconvSolver, Spectrum_1D
from wnet.distances import DistanceMetric
import numpy as np
from scipy.optimize import minimize


E = Spectrum_1D([1, 100], [10, 30])

T1 = Spectrum_1D([1], [2])    # optimal proportion: 5

T2 = Spectrum_1D([100], [3])  # optimal proportion: 10

solver = DeconvSolver(
    empirical_spectrum=E,
    theoretical_spectra=[T1, T2],
    distance=DistanceMetric.LINF,
    max_distance=10,
    trash_cost=100,
    scale_factor=1000,
)


step = [0]

def cost_and_grad(point):
    solver.set_point(point)
    c = solver.total_cost()
    g = solver.gradient()
    print(f"step {step[0]:3d}  point=[{point[0]:8.4f}, {point[1]:8.4f}]  cost={c:12.4f}  grad=[{g[0]:10.4f}, {g[1]:10.4f}]")
    step[0] += 1
    return c, g


result = minimize(
    cost_and_grad,
    x0=[1.0, 1.0],
    jac=True,
    method="L-BFGS-B",
    bounds=[(0, 100), (0, 100)],
)

print(f"Optimal point: {result.x}")
print(f"Cost at optimum: {result.fun:.6f}")
print(f"Expected: [5, 10] with cost 0")
print(f"Success: {result.success}")
print(f"Message: {result.message}")
