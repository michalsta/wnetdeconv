"""
bench_warm_start.py
-------------------
Warm-start vs cold-start benchmark across all data_loader datasets.

4 graph configurations (factory × trash):
  factory: chain (O(m+n))  vs  dense (O(m·n), uses {dense_scale}× less data)
  trash:   simple           vs  asymmetric (separate experimental/theoretical)

For each (dataset, config) pair:
  * identical re-solve speedup  (DualRatio warm vs NONE cold)
  * per-solve timing for each WarmMode over an L-BFGS-B run
  * gradient variants: solve+exact, solve+fast_approx, cached

Parallelism is at the (dataset × config × warm_mode) level; each warm mode
runs as an independent worker task.  Results are printed per (dataset, config)
group once all its pieces arrive.  --serial disables the pool.

Usage:
  python bench_warm_start.py                               # all datasets/configs
  python bench_warm_start.py --serial
  python bench_warm_start.py --datasets pinene_benzyl hemoglobin
  python bench_warm_start.py --configs chain_simple dense_asym
  python bench_warm_start.py --approx-runtime 0.05        # ~0.05 s/step target
  python bench_warm_start.py --workers 4
  python bench_warm_start.py --timeout 60                 # per-mode timeout in s
"""

import argparse
import contextlib
import io
import multiprocessing
import os
import signal
import sys
import time
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from wnet.wnet_cpp import NetworkSimplex, WarmMode

# ─── constants ────────────────────────────────────────────────────────────────

ALL_DATASETS = [
    "pinene_benzyl",
    "overlapping_intensity",
    "perfumes",
    "shim",
    "hemoglobin",
    "pbttt_p1",
    "pbttt_p2",
    "pbttt_p3",
    "pbttt_p3_7p",
]

ALL_CONFIGS = ["chain_simple", "chain_asym", "dense_simple", "dense_asym"]

_WARM_MODES = [
    ("none",   WarmMode.NONE),
    ("simpl",  WarmMode.Simple),
    ("dual",   WarmMode.Dual),
    ("primal", WarmMode.Primal),
    ("dualR",  WarmMode.DualRatio),
    ("dualG",  WarmMode.DualGreedy),
    ("lct",    WarmMode.LinkCut),
]
_MODE_LABELS = [label for label, _ in _WARM_MODES]

_LBFGS_OPTS  = {"ftol": 1e-15, "gtol": 1e-10, "maxiter": 500}
_DENSE_SCALE = 0.2   # approx_runtime multiplier for dense factory (~5× fewer peaks)
_N_REPROBE   = 10    # identical re-solve repetitions (default)
_N_GRAD      = 150   # gradient-variant timing repetitions
_TIMEOUT_S   = 180   # per-mode timeout (seconds)

# Tasks per (dataset, config) group: 1 probe + N modes + 1 grad
_TASKS_PER_GROUP = 1 + len(_WARM_MODES) + 1

# Populated by main() before pool creation; inherited by all workers via fork.
_PRELOADED: dict = {}


# ─── shared helpers ───────────────────────────────────────────────────────────

def _load_dataset(name, approx_runtime):
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import data_loader as dl
    return getattr(dl, f"load_{name}")(approx_runtime=approx_runtime)


def _make_solver(emp, theo, factory, trash, mode, kappa):
    from wnetdeconv import DeconvSolver
    from wnet.distances import DistanceMetric
    ns = NetworkSimplex()
    ns.warm = mode
    kw = dict(solver=ns, force_dense_1d=(factory == "dense"))
    if trash == "simple":
        kw["trash_cost"] = kappa
    else:
        kw["experimental_trash_cost"] = kappa
        kw["theoretical_trash_cost"] = kappa
    return DeconvSolver(emp, theo, DistanceMetric.L1, kappa, **kw)


def _load_emp_theo(ds_name, factory, approx_rt):
    eff_rt = approx_rt * (_DENSE_SCALE if factory == "dense" else 1.0)
    ds = _load_dataset(ds_name, eff_rt)
    return ds.experimental, ds.theoretical, ds.suggested_max_transport_distance, eff_rt


def _get_dataset(ds_name, factory, approx_rt):
    """Look up pre-loaded data; fall back to loading if not cached."""
    key = (ds_name, factory, approx_rt)
    cached = _PRELOADED.get(key)
    if cached is None:
        return _load_emp_theo(ds_name, factory, approx_rt)
    if isinstance(cached, Exception):
        raise cached
    return cached


# ─── task workers ─────────────────────────────────────────────────────────────

def _task_probe(args):
    """Graph dimensions + identical re-solve speedup."""
    ds_name, factory, trash, approx_rt, n_reprobe = args
    try:
        emp, theo, kappa, eff_rt = _get_dataset(ds_name, factory, approx_rt)
    except Exception as exc:
        return {"kind": "probe", "dataset": ds_name, "factory": factory,
                "trash": trash, "error": str(exc)}

    n = len(theo)
    pt = np.ones(n) / n

    s_probe = _make_solver(emp, theo, factory, trash, WarmMode.DualRatio, kappa)
    nsub    = s_probe.graph.no_subgraphs()
    chain_e = s_probe.graph.wnet.count_chain_edges()
    tot_n   = sum(s_probe.graph.get_subgraph(i).no_nodes() for i in range(nsub))
    tot_e   = sum(s_probe.graph.get_subgraph(i).no_edges() for i in range(nsub))

    s_warm = s_probe
    s_cold = _make_solver(emp, theo, factory, trash, WarmMode.NONE, kappa)
    s_warm.graph.solve(pt)
    tw, tc = [], []
    for _ in range(n_reprobe):
        t0 = time.perf_counter(); s_warm.graph.solve(pt); tw.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); s_cold.graph.solve(pt); tc.append(time.perf_counter() - t0)

    return {
        "kind": "probe", "dataset": ds_name, "factory": factory, "trash": trash,
        "error": None, "eff_rt": eff_rt, "kappa": kappa,
        "n_theo": n, "nsub": nsub, "nodes": tot_n, "edges": tot_e, "chain_e": chain_e,
        "cold_ms": float(np.median(tc)) * 1e3,
        "warm_ms": float(np.median(tw)) * 1e3,
    }


def _mode_subprocess(conn, ds_name, factory, trash, approx_rt, label, mode):
    """Grandchild process: run one L-BFGS-B mode, send result dict through conn."""
    try:
        emp, theo, kappa, _eff = _get_dataset(ds_name, factory, approx_rt)
    except Exception as exc:
        conn.send({"error": str(exc)})
        return

    n = len(theo)
    pt = np.ones(n) / n
    s = _make_solver(emp, theo, factory, trash, mode, kappa)
    solve_times = []
    n_calls = [0]
    n_iters = [0]

    def cost_and_grad(w):
        t0 = time.perf_counter()
        s.set_point(np.asarray(w))
        solve_times.append(time.perf_counter() - t0)
        n_calls[0] += 1
        return float(s.total_cost()), np.array(s.gradient(), dtype=float)

    def callback(x):
        n_iters[0] += 1

    t0 = time.perf_counter()
    try:
        minimize(cost_and_grad, pt.copy(), jac=True, method="L-BFGS-B",
                 bounds=[(0.0, None)] * n, options=_LBFGS_OPTS, callback=callback)
    except Exception:
        pass
    t_total = time.perf_counter() - t0

    t_arr = np.array(solve_times) * 1e3
    conn.send({
        "error":   None,
        "nit":     n_iters[0],
        "n_calls": n_calls[0],
        "warm":    s.graph.wnet.warm_start_count(),
        "dual":    s.graph.wnet.dual_repair_count(),
        "primal":  s.graph.wnet.primal_repair_count(),
        "cold":    s.graph.wnet.cold_start_count(),
        "t_total": t_total,
        "med_ms":  float(np.median(t_arr))         if len(t_arr) else 0.0,
        "p95_ms":  float(np.percentile(t_arr, 95)) if len(t_arr) else 0.0,
    })


def _task_mode(args):
    """One L-BFGS-B run for a single WarmMode with hard subprocess timeout.

    Uses os.fork() directly so this works even inside daemon pool workers
    (multiprocessing.Process is forbidden in daemon processes on Python 3.12+).
    """
    ds_name, factory, trash, approx_rt, timeout, label, mode = args

    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    pid = os.fork()
    if pid == 0:
        # grandchild: run mode, send result, exit
        parent_conn.close()
        try:
            _mode_subprocess(child_conn, ds_name, factory, trash, approx_rt, label, mode)
        finally:
            child_conn.close()
            os._exit(0)

    child_conn.close()
    t0 = time.perf_counter()
    try:
        completed = parent_conn.poll(timeout)
        if completed:
            data = parent_conn.recv()
        else:
            os.kill(pid, signal.SIGTERM)
            data = None
    finally:
        os.waitpid(pid, 0)
    t_total = time.perf_counter() - t0

    if data is not None and data.get("error"):
        return {"kind": "mode", "dataset": ds_name, "factory": factory,
                "trash": trash, "label": label, "error": data["error"],
                "timeout": timeout}

    return {
        "kind": "mode", "dataset": ds_name, "factory": factory, "trash": trash,
        "error": None, "label": label,
        "nit":     data["nit"]     if data else 0,
        "n_calls": data["n_calls"] if data else 0,
        "warm":    data["warm"]    if data else 0,
        "dual":    data["dual"]    if data else 0,
        "primal":  data["primal"]  if data else 0,
        "cold":    data["cold"]    if data else 0,
        "t_total": t_total,
        "med_ms":  data["med_ms"]  if data else 0.0,
        "p95_ms":  data["p95_ms"]  if data else 0.0,
        "timed_out": not completed,
        "timeout": timeout,
    }


def _task_grad(args):
    """Gradient-variant timing (exact, fast_approx, cached)."""
    ds_name, factory, trash, approx_rt = args
    try:
        emp, theo, kappa, eff_rt = _get_dataset(ds_name, factory, approx_rt)
    except Exception as exc:
        return {"kind": "grad", "dataset": ds_name, "factory": factory,
                "trash": trash, "error": str(exc)}

    n = len(theo)
    pt = np.ones(n) / n
    sg = _make_solver(emp, theo, factory, trash, WarmMode.Dual, kappa)
    sg.set_point(pt)
    N = _N_GRAD

    t0 = time.perf_counter()
    for _ in range(N): sg.set_point(pt); sg.gradient()
    grad_exact_ms = (time.perf_counter() - t0) / N * 1e3

    t0 = time.perf_counter()
    for _ in range(N): sg.set_point(pt); sg.gradient_fast_approx()
    grad_fast_ms = (time.perf_counter() - t0) / N * 1e3

    sg.set_point(pt)
    t0 = time.perf_counter()
    for _ in range(N): sg.gradient()
    grad_cache_ms = (time.perf_counter() - t0) / N * 1e3

    return {
        "kind": "grad", "dataset": ds_name, "factory": factory, "trash": trash,
        "error": None,
        "grad_exact_ms": grad_exact_ms,
        "grad_fast_ms":  grad_fast_ms,
        "grad_cache_ms": grad_cache_ms,
    }


def _run_task(args):
    kind = args[0]
    if kind == "probe":
        return _task_probe(args[1:])
    elif kind == "mode":
        return _task_mode(args[1:])
    else:
        return _task_grad(args[1:])


# ─── output ───────────────────────────────────────────────────────────────────

def _print_group(group):
    probe = group.get("probe", {})
    modes = group.get("modes", {})
    grad  = group.get("grad",  {})

    ds      = probe.get("dataset") or next(
        (v.get("dataset") for v in modes.values()), "?")
    factory = probe.get("factory", "?")
    trash   = probe.get("trash",   "?")

    if probe.get("error"):
        print(f"\n{'='*80}")
        print(f"ERROR (probe)  dataset={ds}  factory={factory}  trash={trash}")
        print(f"  {probe['error']}")
        sys.stdout.flush()
        return

    n_to = sum(1 for m in modes.values() if m.get("timed_out"))
    timeout = next((m.get("timeout") for m in modes.values() if m.get("timeout")), 0)
    timeout_tag = (f"  [{n_to}/{len(modes)} modes timed out at {timeout:.0f}s]"
                   if n_to else "")

    print(f"\n{'='*80}")
    print(f"dataset={ds:22s}  factory={factory:6s}  trash={trash:6s}"
          f"  eff_rt={probe.get('eff_rt', 0):.3f}s{timeout_tag}")
    print(f"{'='*80}")
    print(f"  n_theo={probe.get('n_theo','?')}  subgraphs={probe.get('nsub','?')}  "
          f"nodes={probe.get('nodes','?')}  edges={probe.get('edges','?')}  "
          f"chain_e={probe.get('chain_e','?')}")
    cold_ms = probe.get("cold_ms", 0)
    warm_ms = probe.get("warm_ms", 0)
    speedup = cold_ms / max(warm_ms, 1e-9)
    print(f"  re-solve:  cold={cold_ms:<8.2f}ms  warm={warm_ms:<8.2f}ms  "
          f"speedup={speedup:<5.1f}x")

    for label in _MODE_LABELS:
        m = modes.get(label)
        if m is None:
            print(f"  [{label:6s}] (missing)")
            continue
        if m.get("error"):
            print(f"  [{label:6s}] ERROR: {m['error']}")
            continue
        mode_tag = "  [TIMEOUT]" if m.get("timed_out") else ""
        print(f"  [{m['label']:6s}] iters={m['nit']:<3d} calls={m['n_calls']:<3d} "
              f"warm={m['warm']:<3d} dual={m['dual']:<3d} primal={m['primal']:<3d} "
              f"cold={m['cold']:<3d}  total={m['t_total']:<7.2f}s  "
              f"med={m['med_ms']:<8.2f}ms  p95={m['p95_ms']:<8.2f}ms{mode_tag}")

    if grad.get("error"):
        print(f"  gradient:  ERROR: {grad['error']}")
    elif grad:
        ratio = grad["grad_exact_ms"] / max(grad["grad_cache_ms"], 1e-9)
        print(f"  gradient:  exact={grad['grad_exact_ms']:<8.3f}ms  "
              f"fast_approx={grad['grad_fast_ms']:<8.3f}ms  "
              f"cached={grad['grad_cache_ms']:<8.3f}ms  ({ratio:.0f}x speedup cached)")
    sys.stdout.flush()


# ─── main ─────────────────────────────────────────────────────────────────────

def _group_key(r):
    return (r["dataset"], r["factory"], r["trash"])


def _absorb(groups, counts, r):
    """Merge one result into the groups dict; return key if group is now complete."""
    key = _group_key(r)
    g = groups[key]
    kind = r["kind"]
    if kind == "probe":
        g["probe"] = r
    elif kind == "mode":
        g.setdefault("modes", {})[r["label"]] = r
    else:
        g["grad"] = r
    counts[key] += 1
    if counts[key] == _TASKS_PER_GROUP:
        return key
    return None


def main():
    # Inherit _PRELOADED via fork; Python 3.14's default "forkserver" forks
    # from a process spawned before _PRELOADED is populated, forcing every
    # worker to reload datasets from disk.
    multiprocessing.set_start_method("fork", force=True)

    p = argparse.ArgumentParser(
        description=__doc__.format(dense_scale=_DENSE_SCALE),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--serial", action="store_true",
                   help="Run single-process instead of parallel")
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                   choices=ALL_DATASETS, metavar="DS",
                   help="Datasets to benchmark (default: all). "
                        "Choices: " + " ".join(ALL_DATASETS))
    p.add_argument("--configs", nargs="+", default=ALL_CONFIGS,
                   choices=ALL_CONFIGS, metavar="CFG",
                   help="Graph configs to benchmark (default: all). "
                        "Choices: " + " ".join(ALL_CONFIGS))
    p.add_argument("--approx-runtime", type=float, default=0.02, metavar="S",
                   help=f"Target seconds per gradient step for chain configs; "
                        f"dense uses {_DENSE_SCALE}× (default: 0.02)")
    p.add_argument("--reps", type=int, default=_N_REPROBE, metavar="N",
                   help=f"Identical re-solve repetitions (default: {_N_REPROBE})")
    p.add_argument("--workers", type=int, default=None, metavar="N",
                   help="Parallel worker count (default: cpu_count)")
    p.add_argument("--timeout", type=float, default=_TIMEOUT_S, metavar="S",
                   help=f"Per-mode timeout in seconds; partial results are printed "
                        f"(default: {_TIMEOUT_S})")
    args = p.parse_args()

    tasks = []
    for ds in args.datasets:
        for cfg in args.configs:
            factory, trash = cfg.split("_")
            tasks.append(("probe", ds, factory, trash, args.approx_runtime, args.reps))
            for label, mode in _WARM_MODES:
                tasks.append(("mode", ds, factory, trash, args.approx_runtime,
                               args.timeout, label, mode))
            tasks.append(("grad", ds, factory, trash, args.approx_runtime))

    n_groups  = len(args.datasets) * len(args.configs)
    n_workers = args.workers or multiprocessing.cpu_count()
    print(f"bench_warm_start: {len(tasks)} tasks / {n_groups} groups  "
          f"({'serial' if args.serial else f'{n_workers} workers'})")
    print(f"datasets : {args.datasets}")
    print(f"configs  : {args.configs}")
    print(f"approx_runtime={args.approx_runtime}s  "
          f"(dense: {args.approx_runtime * _DENSE_SCALE:.3f}s)  "
          f"timeout={args.timeout:.0f}s")

    unique_loads = {
        (ds, cfg.split("_")[0], args.approx_runtime)
        for ds in args.datasets
        for cfg in args.configs
    }
    print(f"\nPre-loading {len(unique_loads)} dataset/factory combination(s)...")
    for ds_name, factory, approx_rt in sorted(unique_loads):
        print(f"  {ds_name:30s} factory={factory} ...", end=" ", flush=True)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                data = _load_emp_theo(ds_name, factory, approx_rt)
            _PRELOADED[(ds_name, factory, approx_rt)] = data
            _, theo, _, eff_rt = data
            print(f"ok  n_theo={len(theo)}  eff_rt={eff_rt:.3f}s")
        except Exception as exc:
            _PRELOADED[(ds_name, factory, approx_rt)] = exc
            print(f"ERROR: {exc}")
    print()

    groups = defaultdict(dict)
    counts = defaultdict(int)

    def process(r):
        key = _absorb(groups, counts, r)
        if key is not None:
            _print_group(groups.pop(key))
            del counts[key]

    if args.serial:
        for t in tasks:
            process(_run_task(t))
    else:
        with multiprocessing.Pool(processes=n_workers) as pool:
            for r in pool.imap_unordered(_run_task, tasks):
                process(r)

    # print any groups that never completed (shouldn't happen)
    for key in list(groups):
        _print_group(groups[key])


if __name__ == "__main__":
    main()
