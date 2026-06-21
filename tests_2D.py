from wnetdeconv.solver import DeconvSolver
from wnet.distances import DistanceMetric
import numpy as np
from scipy.optimize import minimize

import pandas as pd
from glob import glob

from utils import load_2d_spectrum

MD = 0.07 # 0.025
TC = 0.1
# SF = None
SF = 1e5

MAX_PEAK = 0.01 # 0.001

MTD = 0.05
MTD_TH = 0.05

data_path = "data/benzen/2D_baseline_correction"
print(f"{data_path}, max-distance={MD}, trash_cost={TC}, mtd={MTD}, mtd_th={MTD_TH}, scale_factor={SF}")

spectra = []
for path in sorted(glob(data_path + "/*.csv.gz")):
    s = load_2d_spectrum(path,
                        max_peak_fraction=MAX_PEAK,
                        # intensity_threshold=0,
                        verbose=True,
                        )
    s = s.normalized()
    print(s.label)
    # s.plot()
    spectra.append(s)

# print("n spectra", len(real_spectra))

# real_proportions = [0.862, 0.0443, 0.0458, 0.0479]
# components = real_spectra

# # create mixture spectrum
# for i in range(len(components)):
#     if i == 0: mixture = components[i] * real_proportions[i]
#     else: mixture += components[i] * real_proportions[i]
# mixture = mixture.normalized()
# mixture.label = "mixture"
# print("created mixture spectrum")

# x0 = np.ones(len(components))
# x0 = x0 / x0.shape[0]
# bounds = [(0, 2) for _ in x0]

# solver = DeconvSolver(
#                     empirical_spectrum=mixture,
#                     theoretical_spectra=components,
#                     distance=DistanceMetric.LINF,
#                     max_distance=MD,
#                     experimental_trash_cost = MTD,
#                     theoretical_trash_cost = MTD_TH,
#                     scale_factor=SF,
#                     )
    
# # result = solver.optimize(x0=x0)
# result = solver.optimize()

# step = [0]

# def cost_and_grad(point):
#     solver.set_point(point)
#     c = solver.total_cost()
#     g = solver.gradient()
#     print(f"step {step[0]:3d}  point=[{', '.join(f"{x:8.4f}" for x in point)}]  cost={c:8.4f}  grad=[{', '.join(f"{x:8.4f}" for x in g)}]")
#     step[0] += 1
#     return c, g

# result = minimize(
#         cost_and_grad,
#         x0=x0,
#         jac=True,
#         method="L-BFGS-B",
#         bounds=bounds,
#         )
    
# print(f"Optimal point: {result.x}")
# print(f"Cost at optimum: {result.fun:.6f}")
# print(f"Success: {result.success}")
# print(f"Message: {result.message}")