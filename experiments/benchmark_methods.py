#!/usr/bin/env python3
"""Hemoglobin deconvolution — solver method/variant benchmark.

Generates the Masserstein hemoglobin example (HbA + HbB + myoglobin at
multiple charge states), then times every wnet min-cost flow variant through
a full SLSQP optimisation loop.  Variants that exceed n × (reference time)
are killed with SIGKILL so the C++ code is actually terminated.

Usage
-----
    python benchmark_methods.py [options]

    --n N         Kill variants > N × reference time  (default: 10)
    --maxiter N   Max SLSQP iterations per variant     (default: 200)
    --jobs N      Parallel subprocesses                (default: cpu_count)
    --seed N      RNG seed for spectrum simulation     (default: 42)

Run from the wnetdeconv repo root (or any directory that is NOT the
wnet_stuff directory) so the masserstein editable install is found correctly.
"""

import argparse
import dataclasses
import os
import signal
import sys
import time
from typing import Optional

import numpy as np
from scipy.optimize import minimize
import multiprocessing as mp

# ── wnet / wnetdeconv ─────────────────────────────────────────────────────────
from wnetdeconv import DeconvSolver, Spectrum_1D
from wnet.distances import DistanceMetric
from wnet.wnet_cpp import (
    NetworkSimplex,
    CostScaling,
    CycleCanceling,
    CapacityScaling,
    NSPivotRule,
    CSMethod,
    CCMethod,
)

# ── masserstein (spectrum generation only) ────────────────────────────────────
from masserstein import Spectrum as MasserSpectrum, peptides

# =============================================================================
# Problem constants
# =============================================================================

_DISTANCE = DistanceMetric.LINF
_MAX_DISTANCE = 0.025
_TRASH_COST = 0.1

# =============================================================================
# Solver variant registry
# =============================================================================


@dataclasses.dataclass
class SolverSpec:
    name: str
    kind: str  # "ns" | "cs" | "cc" | "caps"
    params: dict  # plain, picklable
    is_reference: bool = False

    def build(self):
        """Construct the solver config object."""
        if self.kind == "ns":
            s = NetworkSimplex()
            s.warm = self.params.get("warm", True)
            s.pivot = getattr(NSPivotRule, self.params.get("pivot", "BLOCK_SEARCH"))
            return s
        if self.kind == "cs":
            s = CostScaling()
            s.method = getattr(CSMethod, self.params.get("method", "PARTIAL_AUGMENT"))
            s.factor = self.params.get("factor", 16)
            return s
        if self.kind == "cc":
            s = CycleCanceling()
            s.method = getattr(
                CCMethod, self.params.get("method", "CANCEL_AND_TIGHTEN")
            )
            return s
        if self.kind == "caps":
            s = CapacityScaling()
            s.factor = self.params.get("factor", 4)
            return s
        raise ValueError(f"unknown kind {self.kind!r}")


def _all_variants() -> list[SolverSpec]:
    specs = []

    # NetworkSimplex — warm (reference = BLOCK_SEARCH warm)
    for pivot in (
        "BLOCK_SEARCH",
        "FIRST_ELIGIBLE",
        "BEST_ELIGIBLE",
        "CANDIDATE_LIST",
        "ALTERING_LIST",
    ):
        specs.append(
            SolverSpec(
                name=f"NS  warm  {pivot}",
                kind="ns",
                params={"warm": True, "pivot": pivot},
                is_reference=(pivot == "BLOCK_SEARCH"),
            )
        )

    # NetworkSimplex — cold
    for pivot in (
        "BLOCK_SEARCH",
        "FIRST_ELIGIBLE",
        "BEST_ELIGIBLE",
        "CANDIDATE_LIST",
        "ALTERING_LIST",
    ):
        specs.append(
            SolverSpec(
                name=f"NS  cold  {pivot}",
                kind="ns",
                params={"warm": False, "pivot": pivot},
            )
        )

    # CostScaling
    for method in ("PARTIAL_AUGMENT", "AUGMENT", "PUSH"):
        specs.append(
            SolverSpec(
                name=f"CostScaling  {method}  f=16",
                kind="cs",
                params={"method": method, "factor": 16},
            )
        )
    for factor in (4, 32):
        specs.append(
            SolverSpec(
                name=f"CostScaling  PARTIAL_AUGMENT  f={factor}",
                kind="cs",
                params={"method": "PARTIAL_AUGMENT", "factor": factor},
            )
        )

    # CycleCanceling
    for method in (
        "CANCEL_AND_TIGHTEN",
        "MINIMUM_MEAN_CYCLE_CANCELING",
        "SIMPLE_CYCLE_CANCELING",
    ):
        specs.append(
            SolverSpec(
                name=f"CycleCanceling  {method}",
                kind="cc",
                params={"method": method},
            )
        )

    # CapacityScaling
    for factor in (4, 16, 32):
        specs.append(
            SolverSpec(
                name=f"CapacityScaling  f={factor}",
                kind="caps",
                params={"factor": factor},
            )
        )

    return specs


VARIANTS = _all_variants()

# =============================================================================
# Spectrum generation (runs once in parent)
# =============================================================================

_MYOGLOBIN = (
    "GLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKFKHLKSEDEMKASE"
    "DLKKHGATVLTALGGILKKKGHHEAEIKPLAQSHATKHKIPVKYLEFISECIIQVLQSKH"
    "PGDFGADAQGAMNKALELFRKDMASNYKELGFQG"
)
_HBB = (
    "VHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPK"
    "VKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFG"
    "KEFTPPVQAAYQKVVAGVANALAHKYH"
)
_HBA = (
    "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHG"
    "KKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTP"
    "AVHASLDKFLASVSTVLTSKYR"
)


def build_problem(seed: int = 42) -> tuple[tuple, list[tuple]]:
    """Return (emp_data, thr_data) as plain numpy tuples for pickling.

    emp_data  = (mz_arr, intensity_arr)  for the centroided mixture
    thr_data  = list of (mz_arr, intensity_arr) for the 10 theoretical spectra
    """
    np.random.seed(seed)

    hA_f = peptides.get_protein_formula(_HBA)
    hB_f = peptides.get_protein_formula(_HBB)
    myo_f = peptides.get_protein_formula(_MYOGLOBIN)

    masser_spectra = [
        MasserSpectrum(hA_f, charge=19, adduct="H"),
        MasserSpectrum(hA_f, charge=20, adduct="H"),
        MasserSpectrum(hA_f, charge=21, adduct="H"),
        MasserSpectrum(hB_f, charge=20, adduct="H"),
        MasserSpectrum(hB_f, charge=21, adduct="H"),
        MasserSpectrum(hB_f, charge=22, adduct="H"),
        MasserSpectrum(myo_f, charge=21, adduct="H"),
        MasserSpectrum(myo_f, charge=22, adduct="H"),
        MasserSpectrum(myo_f, charge=23, adduct="H"),
        MasserSpectrum(myo_f, charge=24, adduct="H"),
    ]
    masser_spectra.sort(key=lambda s: s.confs[0][0])
    for s in masser_spectra:
        s.normalize()

    proportions_raw = [1, 2, 1.2, 0.5, 0.9, 0.6, 0.2, 0.3, 0.4, 0.0]
    total = sum(proportions_raw)
    proportions = [p / total for p in proportions_raw]

    convolved = MasserSpectrum(label="mix")
    for s, p in zip(masser_spectra, proportions):
        convolved += s * p
    convolved.add_chemical_noise(100, 0.1)
    convolved.gaussian_smoothing(0.01, 0.001)
    convolved.add_gaussian_noise(0.01)

    peaks, _ = convolved.centroid(peak_height_fraction=0.5, max_width=0.03)
    centroided = MasserSpectrum(confs=peaks)
    centroided.normalize()

    def _arr(s):
        mzs = np.array([mz for mz, _ in s.confs])
        ints = np.array([i for _, i in s.confs])
        return mzs, ints

    emp_data = _arr(centroided)
    thr_data = [_arr(s) for s in masser_spectra]
    return emp_data, thr_data


# =============================================================================
# Worker (runs inside child process)
# =============================================================================


def _bench_worker(
    result_queue: mp.Queue,
    emp_data: tuple,
    thr_data: list,
    spec_dict: dict,  # {"kind": ..., "params": {...}}
    maxiter: int,
    distance_name: str,
    max_distance: float,
    trash_cost: float,
):
    """Build solver, run SLSQP, put (elapsed, nit, success, fun) in queue."""
    # All imports inside the function for spawn-compatibility.
    import time as _time
    import numpy as _np
    from scipy.optimize import minimize as _minimize
    from wnetdeconv import DeconvSolver, Spectrum_1D
    from wnet.distances import DistanceMetric
    from wnet.wnet_cpp import (
        NetworkSimplex,
        CostScaling,
        CycleCanceling,
        CapacityScaling,
        NSPivotRule,
        CSMethod,
        CCMethod,
    )

    kind = spec_dict["kind"]
    params = spec_dict["params"]

    if kind == "ns":
        solver = NetworkSimplex()
        solver.warm = params.get("warm", True)
        solver.pivot = getattr(NSPivotRule, params.get("pivot", "BLOCK_SEARCH"))
    elif kind == "cs":
        solver = CostScaling()
        solver.method = getattr(CSMethod, params.get("method", "PARTIAL_AUGMENT"))
        solver.factor = params.get("factor", 16)
    elif kind == "cc":
        solver = CycleCanceling()
        solver.method = getattr(CCMethod, params.get("method", "CANCEL_AND_TIGHTEN"))
    elif kind == "caps":
        solver = CapacityScaling()
        solver.factor = params.get("factor", 4)
    else:
        result_queue.put(None)
        return

    emp_mzs, emp_ints = emp_data
    empirical = Spectrum_1D(emp_mzs, emp_ints)
    theoreticals = [Spectrum_1D(mzs, ints) for mzs, ints in thr_data]
    n = len(theoreticals)

    dist = getattr(DistanceMetric, distance_name)
    ds = DeconvSolver(
        empirical_spectrum=empirical,
        theoretical_spectra=theoreticals,
        distance=dist,
        max_distance=max_distance,
        trash_cost=trash_cost,
        solver=solver,
    )

    def cost_and_grad(point):
        ds.set_point(point)
        return ds.total_cost(), ds.gradient()

    t0 = _time.perf_counter()
    result = _minimize(
        cost_and_grad,
        x0=_np.ones(n) / n,
        jac=True,
        method="SLSQP",
        bounds=[(0.0, None)] * n,
        options={"maxiter": maxiter, "ftol": 1e-12},
    )
    elapsed = _time.perf_counter() - t0
    result_queue.put(
        (elapsed, int(result.nit), bool(result.success), float(result.fun))
    )


# =============================================================================
# Process launcher with SIGKILL timeout
# =============================================================================


def _launch(spec: SolverSpec, emp_data, thr_data, maxiter, dist_name, max_dist, trash):
    """Spawn a subprocess for this variant.  Returns (process, queue, t_start)."""
    q = mp.Queue()
    p = mp.Process(
        target=_bench_worker,
        args=(
            q,
            emp_data,
            thr_data,
            {"kind": spec.kind, "params": spec.params},
            maxiter,
            dist_name,
            max_dist,
            trash,
        ),
        daemon=True,
    )
    p.start()
    return p, q, time.monotonic()


def _reap(p: mp.Process, q: mp.Queue, t_start: float, timeout: float):
    """
    Wait up to ``timeout`` seconds for p, then SIGKILL if still alive.
    Returns (elapsed, nit, success, fun) or None on timeout/error.
    """
    remaining = timeout - (time.monotonic() - t_start)
    if remaining > 0:
        p.join(remaining)
    if p.is_alive():
        try:
            os.kill(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.join()
        return None
    if p.exitcode != 0 or q.empty():
        return None
    try:
        return q.get_nowait()
    except Exception:
        return None


# =============================================================================
# Table output
# =============================================================================

# Set in main() before print_table() is called.
args_n_for_fmt = 10.0


def print_table(rows: list, ref_time: float) -> None:
    col_w = 44
    header = (
        f"{'Solver':<{col_w}} {'Time':>9}  {'vs ref':>7}  "
        f"{'Iter':>5}  {'Conv':>4}  {'Cost':>14}"
    )
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)

    def _sort_key(row):
        name, res = row
        if res is None:
            return (2, 1e99, name)
        ok = res[2]
        return (0 if ok else 1, res[0], name)

    for name, res in sorted(rows, key=_sort_key):
        if res is None:
            t_s = " TIMEOUT"
            rat_s = f" >{args_n_for_fmt:.0f}x".rjust(7)
            nit_s = "    -"
            ok_s = "   -"
            fun_s = "             -"
        else:
            t, nit, ok, fun = res
            t_s = f"{t:8.3f}s"
            rat_s = f"{t / ref_time:6.2f}x"
            nit_s = f"{nit:5d}"
            ok_s = " yes" if ok else "  no"
            fun_s = f"{fun:14.8g}"
        print(f"{name:<{col_w}} {t_s}  {rat_s}  {nit_s}  {ok_s}  {fun_s}")

    print(sep)


# =============================================================================
# Main
# =============================================================================


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--n", type=float, default=10.0, help="kill threshold multiplier (default: 10)"
    )
    p.add_argument(
        "--maxiter",
        type=int,
        default=200,
        help="max SLSQP iterations per variant (default: 200)",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=max(1, os.cpu_count() or 2),
        help="parallel subprocesses (default: cpu_count)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for spectrum simulation (default: 42)",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    global args_n_for_fmt
    args_n_for_fmt = args.n

    mp.set_start_method("fork", force=True)

    dist_name = _DISTANCE.name

    # ── Build spectra ────────────────────────────────────────────────────────
    print("Building hemoglobin spectra...", flush=True)
    emp_data, thr_data = build_problem(seed=args.seed)
    n_emp = len(emp_data[0])
    n_thr = len(thr_data)
    n_theo_peaks = sum(len(t[0]) for t in thr_data)
    print(
        f"  empirical: {n_emp} peaks   "
        f"theoretical: {n_thr} spectra ({n_theo_peaks} peaks total)"
    )
    print(
        f"  distance={dist_name}  max_dist={_MAX_DISTANCE}  "
        f"trash={_TRASH_COST}  maxiter={args.maxiter}",
        flush=True,
    )

    ref_spec = next(s for s in VARIANTS if s.is_reference)
    other_specs = [s for s in VARIANTS if not s.is_reference]

    # ── Reference run ────────────────────────────────────────────────────────
    print(f"\nReference: {ref_spec.name} ...", end=" ", flush=True)
    p, q, t0 = _launch(
        ref_spec,
        emp_data,
        thr_data,
        args.maxiter,
        dist_name,
        _MAX_DISTANCE,
        _TRASH_COST,
    )
    ref_result = _reap(p, q, t0, timeout=3600.0)
    if ref_result is None:
        print("FAILED — cannot establish reference time; aborting.")
        sys.exit(1)

    ref_time = ref_result[0]
    timeout_sec = args.n * ref_time
    print(
        f"{ref_time:.3f}s   →  timeout = {timeout_sec:.1f}s  ({args.n:.0f}×)",
        flush=True,
    )

    results = [(ref_spec.name, ref_result)]

    # ── Remaining variants (parallel batches of --jobs) ───────────────────────
    total = len(other_specs)
    print(
        f"\nRunning {total} variants  "
        f"(jobs={args.jobs}  timeout={timeout_sec:.1f}s each) ...\n",
        flush=True,
    )

    for batch_start in range(0, total, args.jobs):
        batch = other_specs[batch_start : batch_start + args.jobs]

        # Launch entire batch
        running = []
        for spec in batch:
            proc, que, ts = _launch(
                spec,
                emp_data,
                thr_data,
                args.maxiter,
                dist_name,
                _MAX_DISTANCE,
                _TRASH_COST,
            )
            running.append((spec, proc, que, ts))

        # Reap in order (remaining timeout shrinks by wall time already spent)
        for spec, proc, que, ts in running:
            res = _reap(proc, que, ts, timeout=timeout_sec)
            tag = (
                "TIMEOUT"
                if res is None
                else (f"{res[0]:7.3f}s {'ok' if res[2] else 'no-conv':7s}")
            )
            print(f"  {tag}  {spec.name}", flush=True)
            results.append((spec.name, res))

    # ── Table ────────────────────────────────────────────────────────────────
    print()
    print_table(results, ref_time)


if __name__ == "__main__":
    main()
