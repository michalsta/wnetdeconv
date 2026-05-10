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

import multiprocessing

import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

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
myoglobin_formula = peptides.get_protein_formula(myoglobin)

hA19 = MasserSpectrum(haemoglobinA_formula, charge=19, adduct="H", label="hA 19+")
hA20 = MasserSpectrum(haemoglobinA_formula, charge=20, adduct="H", label="hA 20+")
hA21 = MasserSpectrum(haemoglobinA_formula, charge=21, adduct="H", label="hA 21+")
hB20 = MasserSpectrum(haemoglobinB_formula, charge=20, adduct="H", label="hB 20+")
hB21 = MasserSpectrum(haemoglobinB_formula, charge=21, adduct="H", label="hB 21+")
hB22 = MasserSpectrum(haemoglobinB_formula, charge=22, adduct="H", label="hB 22+")
m21 = MasserSpectrum(myoglobin_formula, charge=21, adduct="H", label="myo 21+")
m22 = MasserSpectrum(myoglobin_formula, charge=22, adduct="H", label="myo 22+")
m23 = MasserSpectrum(myoglobin_formula, charge=23, adduct="H", label="myo 23+")
m24 = MasserSpectrum(myoglobin_formula, charge=24, adduct="H", label="myo 24+")

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
convolved = MasserSpectrum(label="Convolved")
for s, p in zip(masser_spectra, proportions):
    convolved += s * p

convolved.add_chemical_noise(100, 0.1)
convolved.gaussian_smoothing(0.01, 0.001)
convolved.add_gaussian_noise(0.01)

# Profile spectrum (normalized)
profile = MasserSpectrum(confs=list(convolved.confs), label="Profile")
profile.normalize()

# Centroided spectrum
peaks, _ = convolved.centroid(peak_height_fraction=0.5, max_width=0.03)
centroided = MasserSpectrum(confs=peaks, label="Centroided")
centroided.normalize()
print(f"Profile spectrum  : {len(profile.confs)} peaks")
print(f"Centroided spectrum: {len(centroided.confs)} peaks")

# =============================================================================
# 4. Helper: masserstein Spectrum -> wnetdeconv Spectrum_1D
# =============================================================================


def masser_to_wnet(s: MasserSpectrum) -> Spectrum_1D:
    """Convert a masserstein Spectrum (list of (mz, intensity) confs) to a
    wnetdeconv Spectrum_1D."""
    mzs = np.array([mz for mz, _ in s.confs])
    ints = np.array([i for _, i in s.confs])
    return Spectrum_1D(mzs, ints, label=s.label)


theoretical_wnets = [masser_to_wnet(s) for s in masser_spectra]
emprical_wnet_profile = masser_to_wnet(profile)
empirical_wnet = masser_to_wnet(centroided)  # default for grid search

# =============================================================================
# 5. Run wnetdeconv (helper + multiprocessing worker)
# =============================================================================


def _worker_init(emp_data, thr_data, true_props):
    """Initializer for each worker process: store shared data in globals."""
    global _emp_data, _thr_data, _true_props
    _emp_data = emp_data
    _thr_data = thr_data
    _true_props = true_props


def _run_solver(empirical, theoreticals, distance, max_distance, trash_cost):
    """Build solver and optimize; shared by both single-run and worker paths."""
    solver = DeconvSolver(
        empirical_spectrum=empirical,
        theoretical_spectra=theoreticals,
        distance=distance,
        max_distance=max_distance,
        trash_cost=trash_cost,
    )
    n = len(theoreticals)

    def cost_and_grad(point):
        solver.set_point(point)
        return solver.total_cost(), solver.gradient()

    result = minimize(
        cost_and_grad,
        x0=np.ones(n) / n,
        jac=True,
        method="SLSQP",
        bounds=[(0.0, None)] * n,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    raw = result.x
    total = raw.sum()
    props = raw / total if total > 0 else raw
    return props, result


def _grid_worker(args):
    """Top-level worker for Pool.imap — must be picklable.
    args = (dist, md, tc) or (dist, md, tc, task_timeout_secs).
    On Unix, a task_timeout uses SIGALRM to actually kill the computation.
    """
    import signal as _signal

    if len(args) == 4:
        dist, md, tc, task_timeout = args
    else:
        dist, md, tc = args
        task_timeout = None

    if task_timeout is not None:

        def _alarm_handler(signum, frame):
            raise TimeoutError()

        _signal.signal(_signal.SIGALRM, _alarm_handler)
        _signal.alarm(int(task_timeout))

    try:
        emp_mzs, emp_ints = _emp_data
        empirical = Spectrum_1D(emp_mzs, emp_ints)
        theoreticals = [Spectrum_1D(mzs, ints) for mzs, ints in _thr_data]
        props, res = _run_solver(empirical, theoreticals, dist, md, tc)
        l1 = sum(abs(p - q) for p, q in zip(_true_props, props))
        return (dist.name, md, tc, l1, res.success)
    except TimeoutError:
        return (dist.name, md, tc, float("nan"), False)
    except Exception:
        return (dist.name, md, tc, float("nan"), False)
    finally:
        if task_timeout is not None:
            _signal.alarm(0)


def run_wnetdeconv(empirical, distance, max_distance, trash_cost, verbose=False):
    """Build solver, optimize, return (proportions, raw_result)."""
    props, result = _run_solver(
        empirical, theoretical_wnets, distance, max_distance, trash_cost
    )
    if verbose:
        print(f"  Optimization: {result.message}  nit={result.nit}")
    return props, result


# =============================================================================
# 6. Run all combinations
# =============================================================================

DIST_CENTR, MD_CENTR, TC_CENTR = DistanceMetric.LINF, 0.025, 0.1
DIST_PROF, MD_PROF, TC_PROF = DistanceMetric.LINF, 0.025, 0.1

print("Running masserstein on centroided...")
masser_centr_result = estimate_proportions(centroided, masser_spectra, MTD=0.05)
masser_centr_raw = masser_centr_result["proportions"]
masser_centr_total = sum(masser_centr_raw)
masser_centr_props = [e / masser_centr_total for e in masser_centr_raw]

print("Running masserstein on profile...")
masser_prof_result = estimate_proportions(profile, masser_spectra, MTD=0.05)
masser_prof_raw = masser_prof_result["proportions"]
masser_prof_total = sum(masser_prof_raw)
masser_prof_props = [e / masser_prof_total for e in masser_prof_raw]

print(
    f"\nRunning wnetdeconv on centroided (LINF, max_dist={MD_CENTR}, trash={TC_CENTR})..."
)
wnet_centr_props, wnet_centr_res = run_wnetdeconv(
    empirical_wnet, DIST_CENTR, MD_CENTR, TC_CENTR, verbose=True
)

print(f"\nRunning wnetdeconv on profile (LINF, max_dist={MD_PROF}, trash={TC_PROF})...")
wnet_prof_props, wnet_prof_res = run_wnetdeconv(
    emprical_wnet_profile, DIST_PROF, MD_PROF, TC_PROF, verbose=True
)

# =============================================================================
# 7. Summary
# =============================================================================

labels = [s.label for s in masser_spectra]
w = max(len(l) for l in labels)


def l1(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


cols = [
    ("wnet/centr", wnet_centr_props),
    ("wnet/profile", wnet_prof_props),
    ("mass/centr", masser_centr_props),
    ("mass/profile", masser_prof_props),
]
cw = 12
header = f"{'Spectrum':<{w}}  {'True':>8}" + "".join(f"  {c:>{cw}}" for c, _ in cols)
print("\n" + header)
print("-" * len(header))
for i, label in enumerate(labels):
    row = f"{label:<{w}}  {proportions[i]:8.4f}"
    for _, props in cols:
        row += f"  {props[i]:>{cw}.4f}"
    print(row)
print("-" * len(header))
row = f"{'L1 distance':<{w}}  {'':>8}"
for _, props in cols:
    row += f"  {l1(proportions, props):>{cw}.4f}"
print(row)

# =============================================================================
# 8. Grid search over wnetdeconv parameters (parallel, separate for each mode)
# =============================================================================


def _run_grid(label, emp_confs, desc, grid=None, timeout=None):
    if grid is None:
        distances = [DistanceMetric.LINF]
        max_dists = [
            0.005,
            0.008,
            0.010,
            0.012,
            0.015,
            0.018,
            0.020,
            0.023,
            0.025,
            0.030,
            0.040,
            0.050,
        ]
        trash_costs = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        grid = [
            (d, md, tc) for d in distances for md in max_dists for tc in trash_costs
        ]

    emp_data = (
        np.array([mz for mz, _ in emp_confs]),
        np.array([i for _, i in emp_confs]),
    )
    thr_data = [
        (np.array([mz for mz, _ in s.confs]), np.array([i for _, i in s.confs]))
        for s in masser_spectra
    ]

    ncpus = multiprocessing.cpu_count()
    if timeout is not None:
        grid = [(d, md, tc, timeout) for d, md, tc in grid]
    with multiprocessing.Pool(
        processes=ncpus,
        initializer=_worker_init,
        initargs=(emp_data, thr_data, proportions),
    ) as pool:
        results = list(tqdm(pool.imap(_grid_worker, grid), total=len(grid), desc=desc))

    # Drop timed-out entries from sorted display but note them
    timed_out = [(d, md, tc) for d, md, tc, err, ok in results if np.isnan(err)]
    valid = [
        (d, md, tc, err, ok) for d, md, tc, err, ok in results if not np.isnan(err)
    ]
    valid.sort(key=lambda x: x[3])
    best = valid[0]
    print(f"\n--- Grid results: {label} ---")
    if timed_out:
        print(f"  (skipped {len(timed_out)} timed-out points)")
    print(f"{'Dist':<6}  {'max_dist':>8}  {'trash':>6}  {'L1':>8}  {'ok':>4}")
    print("-" * 40)
    for dist_name, md, tc, err, ok in valid:
        flag = "yes" if ok else "NO"
        print(f"{dist_name:<6}  {md:>8.3f}  {tc:>6.2f}  {err:8.4f}  {flag:>4}")
    print("-" * 40)
    print(f"Best: LINF  max_dist={best[1]}  trash={best[2]}  L1={best[3]:.4f}")
    return best[0], best[1], best[2]  # dist_name, md, tc


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Grid search — centroided")
    print("=" * 70)
    _, best_md_centr, best_tc_centr = _run_grid(
        "centroided", centroided.confs, desc="centr grid"
    )

    print("\n" + "=" * 70)
    print("Grid search — profile")
    print("=" * 70)
    _, best_md_prof, best_tc_prof = _run_grid(
        "profile",
        profile.confs,
        desc="prof grid",
        timeout=30,
    )

    # --- re-run with optimal params and print final summary -----------------
    print("\n" + "=" * 70)
    print("Final run with grid-optimal parameters")
    print("=" * 70)

    wnet_centr_opt, _ = run_wnetdeconv(
        empirical_wnet, DistanceMetric.LINF, best_md_centr, best_tc_centr
    )
    wnet_prof_opt, _ = run_wnetdeconv(
        emprical_wnet_profile, DistanceMetric.LINF, best_md_prof, best_tc_prof
    )

    cols_opt = [
        (f"wnet/centr\n(md={best_md_centr},tc={best_tc_centr})", wnet_centr_opt),
        (f"wnet/profile\n(md={best_md_prof},tc={best_tc_prof})", wnet_prof_opt),
        ("mass/centr", masser_centr_props),
        ("mass/profile", masser_prof_props),
    ]
    print(
        f"\n{'Spectrum':<{w}}  {'True':>8}"
        + "".join(f"  {c.split(chr(10))[0]:>14}" for c, _ in cols_opt)
    )
    print("-" * (w + 10 + 16 * len(cols_opt)))
    for i, lbl in enumerate(labels):
        row = f"{lbl:<{w}}  {proportions[i]:8.4f}"
        for _, props in cols_opt:
            row += f"  {props[i]:>14.4f}"
        print(row)
    print("-" * (w + 10 + 16 * len(cols_opt)))
    row = f"{'L1 distance':<{w}}  {'':>8}"
    for _, props in cols_opt:
        row += f"  {l1(proportions, props):>14.4f}"
    print(row)

    print(f"\nUpdate defaults at top of file:")
    print(f"  DIST_CENTR, MD_CENTR, TC_CENTR = LINF, {best_md_centr}, {best_tc_centr}")
    print(f"  DIST_PROF,  MD_PROF,  TC_PROF  = LINF, {best_md_prof},  {best_tc_prof}")
