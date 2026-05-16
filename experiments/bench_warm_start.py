"""
bench_warm_start.py
-------------------
Warm-start vs cold-start effects on the magnetstein 2-component estimation
example (preprocessed CSVs).

4 graph configurations (2x2):
  * trash:  simple        vs  asymmetric (experimental + theoretical)
  * 1D fac: chain (O(m+n)) vs  forced dense (O(m*n))

The dense 1D factory is O(m*n) in edges; at the chain threshold (1% of max
intensity, ~6700 nodes) a dense solve is ~1.5 s (~30x chain).  So the dense
cases use a STRICTER intensity threshold (~5x fewer nodes) to stay
tractable; chain cases keep the 1% threshold (comparable to earlier runs).
Dense absolute timings are therefore NOT directly comparable to chain — they
show the dense-factory behaviour and the warm/cold ratio on a ~10x smaller
problem of the same data.

For each configuration:
  * identical re-solve speedup (warm vs cold)
  * per-solve timing across an L-BFGS-B run for each WarmMode
  * gradient derivation: value-exact cache + opt-in fast_approx (dual-pi)
"""

import os, time
import numpy as np
from scipy.optimize import minimize
from wnetdeconv import DeconvSolver, Spectrum_1D
from wnet.distances import DistanceMetric
from wnet.wnet_cpp import NetworkSimplex, WarmMode

DATA = "/home/mist/wnet_stuff/magnetstein/examples"
CHAIN_THRESH = 0.01   # 1% of max intensity  (~6700 nodes)
DENSE_THRESH = 0.12   # stricter -> ~5x fewer nodes (~1360) so dense is tractable
KAPPA = 0.25
X0 = np.array([0.5, 0.5])
PT = np.array([0.5, 0.5])
LBFGS_OPTS = {'ftol': 1e-15, 'gtol': 1e-10, 'maxiter': 500}

_FILES = ("preprocessed_mix.csv", "preprocessed_comp0.csv", "preprocessed_comp1.csv")
_cache = {}

def _load(fname, thresh):
    arr = np.loadtxt(os.path.join(DATA, fname), delimiter=',')
    pos, ints = arr[:, 0], arr[:, 1].copy()
    ints[ints < 0] = 0.0
    mask = ints >= ints.max() * thresh
    return Spectrum_1D(pos[mask], ints[mask]).normalized()

def _spectra(thresh):
    if thresh not in _cache:
        mix = _load(_FILES[0], thresh)
        comp = [_load(_FILES[1], thresh), _load(_FILES[2], thresh)]
        _cache[thresh] = (mix, comp)
    return _cache[thresh]

def make_solver(mode, trash, dense):
    """trash in {'simple','asym'}; dense=bool (force dense 1D factory)."""
    mix, comp = _spectra(DENSE_THRESH if dense else CHAIN_THRESH)
    ns = NetworkSimplex(); ns.warm = mode
    kw = dict(solver=ns, force_dense_1d=dense)
    if trash == "simple":
        kw["trash_cost"] = KAPPA
    else:  # asymmetric: separate experimental + theoretical trash
        kw["experimental_trash_cost"] = KAPPA
        kw["theoretical_trash_cost"] = KAPPA
    return DeconvSolver(mix, comp, DistanceMetric.L1, KAPPA, **kw)


def run_case(trash, dense):
    tag = f"trash={trash:6s} | {'DENSE (5x-stricter data)' if dense else 'chain'}"
    print("=" * 80)
    print(f"CASE: {tag}")
    print("=" * 80)

    s_probe = make_solver(WarmMode.Simple, trash, dense)
    nsub = s_probe.graph.no_subgraphs()
    chain_edges = s_probe.graph.wnet.count_chain_edges()
    tot_n = sum(s_probe.graph.get_subgraph(i).no_nodes() for i in range(nsub))
    tot_e = sum(s_probe.graph.get_subgraph(i).no_edges() for i in range(nsub))
    print(f"  thresh={DENSE_THRESH if dense else CHAIN_THRESH}  subgraphs={nsub}"
          f"  nodes={tot_n}  edges={tot_e}  chain_edges={chain_edges}")

    s_warm = make_solver(WarmMode.Dual, trash, dense)
    s_cold = make_solver(WarmMode.NONE, trash, dense)
    s_warm.graph.solve(PT)                       # prime warm basis
    tw, tc = [], []
    for _ in range(10):
        t0 = time.perf_counter(); s_warm.graph.solve(PT); tw.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); s_cold.graph.solve(PT); tc.append(time.perf_counter() - t0)
    twm, tcm = np.median(tw) * 1e3, np.median(tc) * 1e3
    print(f"  identical re-solve: cold={tcm:<8.2f}ms  warm={twm:<8.2f}ms  "
          f"speedup={tcm / max(twm, 1e-9):<6.1f}x")

    for label, mode in [("none", WarmMode.NONE), ("simpl", WarmMode.Simple),
                        ("dual", WarmMode.Dual), ("primal", WarmMode.Primal)]:
        s2 = make_solver(mode, trash, dense)
        solve_times = []; n_calls = [0]

        def cost_and_grad(w):
            t0 = time.perf_counter()
            s2.set_point(np.asarray(w))
            solve_times.append(time.perf_counter() - t0)
            n_calls[0] += 1
            return float(s2.total_cost()), np.array(s2.gradient(), dtype=float)

        t_total = time.perf_counter()
        res = minimize(cost_and_grad, X0, jac=True, method='L-BFGS-B',
                       bounds=[(0., 1.)] * 2, options=LBFGS_OPTS)
        t_total = time.perf_counter() - t_total

        wc = s2.graph.wnet.warm_start_count()
        cc = s2.graph.wnet.cold_start_count()
        dr = s2.graph.wnet.dual_repair_count()
        pr = s2.graph.wnet.primal_repair_count()
        t_arr = np.array(solve_times) * 1e3
        print(f"  [{label:6s}] iters={res.nit:<2d} calls={n_calls[0]:<3d} "
              f"warm={wc:<3d} dual={dr:<3d} primal={pr:<3d} cold={cc:<3d} "
              f"total={t_total:<7.2f}s "
              f"med={np.median(t_arr):<8.2f}ms p95={np.percentile(t_arr, 95):<8.2f}ms "
              f"x={np.array2string(res.x, precision=4)}")

    sg = make_solver(WarmMode.Dual, trash, dense)
    sg.set_point(PT)
    N = 150
    t0 = time.perf_counter()
    for _ in range(N): sg.set_point(PT); sg.gradient()
    t_re = (time.perf_counter() - t0) / N * 1e3
    t0 = time.perf_counter()
    for _ in range(N): sg.set_point(PT); sg.gradient_fast_approx()
    t_fa = (time.perf_counter() - t0) / N * 1e3
    sg.set_point(PT)
    t0 = time.perf_counter()
    for _ in range(N): sg.gradient()                 # cache hit, no re-solve
    t_ca = (time.perf_counter() - t0) / N * 1e3
    print(f"  gradient: solve+exact={t_re:<8.3f}ms  solve+fast_approx={t_fa:<8.3f}ms"
          f"  cached={t_ca:<8.3f}ms ({t_re / max(t_ca, 1e-9):<3.0f}x)")
    print()


if __name__ == "__main__":
    for dense in (False, True):
        for trash in ("simple", "asym"):
            run_case(trash, dense)
