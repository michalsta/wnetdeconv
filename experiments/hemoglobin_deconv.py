"""
Hemoglobin + myoglobin deconvolution benchmark
===============================================
Reproduces the example from masserstein's Package presentation notebook:
  - generate theoretical isotope envelopes for haemoglobin A/B and myoglobin
    at several charge states using masserstein / IsoSpecPy
  - mix them with known proportions, add chemical + Gaussian noise
  - run scipy.optimize on wnetdeconv's DeconvSolver to recover proportions
  - compare with masserstein's estimate_proportions
"""

from copy import deepcopy

import numpy as np
from scipy.optimize import minimize

# --- masserstein: spectrum simulation only -----------------------------------
from masserstein import Spectrum as MasserSpectrum
from masserstein import peptides, estimate_proportions

# --- wnetdeconv: our solver --------------------------------------------------
from wnetdeconv import DeconvSolver, Spectrum_1D
from wnet.distances import DistanceMetric


# =============================================================================
# 1. Protein sequences
# =============================================================================

myoglobin = (
    "GLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKFKHLKSEDEMKASE"
    "DLKKHGATVLTALGGILKKKGHHEAEIKPLAQSHATKHKIPVKYLEFISECIIQVLQSKH"
    "PGDFGADAQGAMNKALELFRKDMASNYKELGFQG"
)
haemoglobinB = (
    "VHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPK"
    "VKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFG"
    "KEFTPPVQAAYQKVVAGVANALAHKYH"
)
haemoglobinA = (
    "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHG"
    "KKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTP"
    "AVHASLDKFLASVSTVLTSKYR"
)

# =============================================================================
# 2. Generate theoretical spectra via masserstein / IsoSpecPy
# =============================================================================

print("Generating theoretical spectra...")
haemoglobinA_formula = peptides.get_protein_formula(haemoglobinA)
haemoglobinB_formula = peptides.get_protein_formula(haemoglobinB)
myoglobin_formula    = peptides.get_protein_formula(myoglobin)

hA19 = MasserSpectrum(haemoglobinA_formula, charge=19, adduct='H', label='hA 19+')
hA20 = MasserSpectrum(haemoglobinA_formula, charge=20, adduct='H', label='hA 20+')
hA21 = MasserSpectrum(haemoglobinA_formula, charge=21, adduct='H', label='hA 21+')
hB20 = MasserSpectrum(haemoglobinB_formula, charge=20, adduct='H', label='hB 20+')
hB21 = MasserSpectrum(haemoglobinB_formula, charge=21, adduct='H', label='hB 21+')
hB22 = MasserSpectrum(haemoglobinB_formula, charge=22, adduct='H', label='hB 22+')
m21  = MasserSpectrum(myoglobin_formula,    charge=21, adduct='H', label='myo 21+')
m22  = MasserSpectrum(myoglobin_formula,    charge=22, adduct='H', label='myo 22+')
m23  = MasserSpectrum(myoglobin_formula,    charge=23, adduct='H', label='myo 23+')
m24  = MasserSpectrum(myoglobin_formula,    charge=24, adduct='H', label='myo 24+')

masser_spectra = [hA19, hA20, hA21, hB20, hB21, hB22, m21, m22, m23, m24]
masser_spectra.sort(key=lambda x: x.confs[0][0])

for s in masser_spectra:
    s.normalize()

# =============================================================================
# 3. Simulate experimental spectrum
# =============================================================================

proportions_raw = [1, 2, 1.2, 0.5, 0.9, 0.6, 0.2, 0.3, 0.4, 0.0]
proportions = [p / sum(proportions_raw) for p in proportions_raw]

print("Simulating experimental spectrum...")
convolved = MasserSpectrum(label='Convolved')
for s, p in zip(masser_spectra, proportions):
    convolved += s * p

convolved.add_chemical_noise(100, 0.1)
convolved.gaussian_smoothing(0.01, 0.001)
convolved.add_gaussian_noise(0.01)
convolved.normalize()

# =============================================================================
# 4. Helper: masserstein Spectrum -> wnetdeconv Spectrum_1D
# =============================================================================

def masser_to_wnet(s: MasserSpectrum) -> Spectrum_1D:
    """Convert a masserstein Spectrum (list of (mz, intensity) confs) to a
    wnetdeconv Spectrum_1D."""
    mzs = np.array([mz for mz, _ in s.confs])
    ints = np.array([i  for _, i  in s.confs])
    return Spectrum_1D(mzs, ints, label=s.label)

empirical_wnet     = masser_to_wnet(convolved)
theoretical_wnets  = [masser_to_wnet(s) for s in masser_spectra]

# =============================================================================
# 5. Run wnetdeconv
# =============================================================================

print("Building DeconvSolver...")
solver = DeconvSolver(
    empirical_spectrum=empirical_wnet,
    theoretical_spectra=theoretical_wnets,
    distance=DistanceMetric.L1,
    max_distance=0.05,
    trash_cost=1.0,
)

step = [0]

def cost_and_grad(point):
    solver.set_point(point)
    c = solver.total_cost()
    g = solver.gradient()
    if step[0] % 20 == 0:
        print(f"  step {step[0]:4d}  cost={c:.6f}")
    step[0] += 1
    return c, g

print("Optimizing...")
n = len(theoretical_wnets)
result = minimize(
    cost_and_grad,
    x0=np.ones(n) / n,
    jac=True,
    method="SLSQP",
    bounds=[(0.0, None)] * n,
    options={"maxiter": 500, "ftol": 1e-12},
)
print(f"Optimization finished: {result.message}")

wnet_estimates_raw = result.x
wnet_total = wnet_estimates_raw.sum()
wnet_proportions = wnet_estimates_raw / wnet_total if wnet_total > 0 else wnet_estimates_raw
wnet_optimal_cost = result.fun

# =============================================================================
# 6. Run masserstein (reference)
# =============================================================================

print("\nRunning masserstein reference...")
masser_result = estimate_proportions(convolved, masser_spectra, MTD=0.05)
masser_raw   = masser_result['proportions']
masser_total = sum(masser_raw)
masser_proportions = [e / masser_total for e in masser_raw]

# Evaluate wnetdeconv cost at masserstein's (unnormalised) proportions
solver.set_point(np.array(masser_raw))
masser_cost_in_wnet = solver.total_cost()
# Evaluate at true proportions
solver.set_point(np.array(proportions))
true_cost_in_wnet = solver.total_cost()
# Also evaluate at wnetdeconv's raw (unnormalised) solution for a fair comparison
solver.set_point(wnet_estimates_raw)
wnet_cost_check = solver.total_cost()

# =============================================================================
# 7. Print comparison
# =============================================================================

labels = [s.label for s in masser_spectra]
w      = max(len(l) for l in labels)

wnet_l1   = sum(abs(p - q) for p, q in zip(proportions, wnet_proportions))
masser_l1 = sum(abs(p - q) for p, q in zip(proportions, masser_proportions))

print(f"\n{'Spectrum':<{w}}  {'True':>8}  {'wnetdeconv':>12}  {'masserstein':>12}")
print("-" * (w + 38))
for label, p_true, p_wnet, p_masser in zip(
        labels, proportions, wnet_proportions, masser_proportions):
    print(f"{label:<{w}}  {p_true:8.4f}  {p_wnet:12.4f}  {p_masser:12.4f}")
print("-" * (w + 38))
print(f"{'L1 distance':<{w}}  {'':>8}  {wnet_l1:12.4f}  {masser_l1:12.4f}")
print(f"\nwnetdeconv cost at wnetdeconv solution : {wnet_cost_check:.6f}")
print(f"wnetdeconv cost at masserstein solution: {masser_cost_in_wnet:.6f}")
print(f"wnetdeconv cost at true proportions    : {true_cost_in_wnet:.6f}")
if masser_cost_in_wnet < wnet_cost_check:
    print("=> masserstein solution has LOWER wnetdeconv cost (wnetdeconv did not find the global optimum)")
else:
    print(f"=> wnetdeconv solution is better by {masser_cost_in_wnet - wnet_cost_check:.6f}")