from wnetdeconv import DeconvSolver
from wnetdeconv import Spectrum_1D, Spectrum
from wnet.distances import DistanceMetric
import numpy as np
from scipy.optimize import minimize

from glob import glob

from utils import load_magnetstein_spectra
from utils import load_2d_spectrum

# 1D test using magnestein data 
print("1D test")
n = 2 # experiment number
data_path = f"data/Magnestein/e{3}"
print(data_path)

MD = 0.25 
TC = 0.25 # trash cost, for 1 trash set-up
MTD = 0.25 # kappa_mixture
MTD_TH = 0.22 # kappa_components
SF = None # scaling factor

print(f"e{n}, max-distance={MD}, trash_cost={TC}, mtd={MTD}, mtd_th={MTD_TH}, scale_factor={SF}")

# Load spectra (the last one is mixture)
spectra = []
for path in sorted(glob(data_path + "/*.csv")):
    s = load_magnetstein_spectra(path,
                                max_peak_fraction=0.01,
                                # intensity_threshold=0,
                                # verbose=True,
                                )
    spectra.append(s)

solver = DeconvSolver(
    empirical_spectrum=spectra[-1].normalized(),
    theoretical_spectra=[s.normalized() for s in spectra[:-1]],
    distance=DistanceMetric.LINF,
    max_distance=MD,
    trash_cost=TC,
    experimental_trash_cost = MTD,
    theoretical_trash_cost = MTD_TH,
    scale_factor=SF,
)

# starting point and bounds
x0 = np.ones(len(spectra) - 1)
x0 = x0 / x0.shape[0]
bounds = [(0, 2) for _ in x0]

# using optimize method
# result = solver.optimize(x0=x0)

# custom optimization
step = [0]
def cost_and_grad(point):
    solver.set_point(point)
    c = solver.total_cost()
    g = solver.gradient()
    # print(f"step {step[0]:3d}  point=[{point[0]:8.4f}, {point[1]:8.4f}]  cost={c:12.4f}  grad=[{g[0]:10.4f}, {g[1]:10.4f}]")
    if step[0]%5==0: print(f"step {step[0]:3d}  point=[{' '.join(f"{x:8.4f}," for x in point)}]  cost={c:8.4f}  grad=[{' '.join(f"{x:8.4f}," for x in g)}]")
    step[0] += 1
    return c, g

result = minimize(
    cost_and_grad,
    x0=x0,
    jac=True,
    method="L-BFGS-B",
    # method='Nelder-Mead',
    bounds=bounds,
)

print(f"Optimal point: {result.x}")
print(f"Cost at optimum: {result.fun:.6f}")
print(f"Success: {result.success}")
print(f"Message: {result.message}")
print()






