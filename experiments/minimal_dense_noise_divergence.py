"""
Smallest example where the FIXED MassersteinSolver still diverges from
dualdeconv2 (the hemoglobin Part-2 phenomenon in miniature).

  Component C : peaks {100, 101} intensities {0.9, 0.1}   (skewed pattern)
  Experimental: {100: 0.6, 100.02: 0.333, 101: 0.0667}    (3 peaks)
                 |signal+noise|  near-isotope noise  |minor isotope|

dualdeconv2 keeps w ~ 0.667: the minor isotope's surplus can't transport
far in its LP (no theoretical abyss, real distance 1), so it stops there.

MassersteinSolver inflates w -> 1: the major isotope's surplus matches the
nearby noise peak at distance 0.02 (cheap), and the minor isotope's
surplus is cheaply discarded by theoretical-trash at 2*MTD instead of
being transported the real distance 1 (its LP cap is max_distance = MTD).

Cross-scoring w_wnet in masserstein's actual LP exposes the inflation as
the over-estimate it really is.

Run:  python experiments/minimal_dense_noise_divergence.py
"""
import numpy as np
import pulp as lp
from pulp.apis import LpSolverDefault

from masserstein import Spectrum as MasserSpectrum
from masserstein.deconv_simplex import dualdeconv2, intensity_generator

from wnetdeconv import MassersteinSolver, Spectrum_1D


def score_at_w(exp_sp, thr_sps, w, MTD, quiet=True):
    """LP value of masserstein's dualdeconv2 model at fixed proportions w."""
    exp_confs = exp_sp.confs
    thr_confs = [s.confs for s in thr_sps]
    axis = sorted({p for p, _ in exp_confs}
                  | {p for c in thr_confs for p, _ in c})
    n = len(axis)
    intervals = [axis[i+1] - axis[i] for i in range(n-1)]
    exp_vec = np.array(list(intensity_generator(exp_confs, axis)))
    thr_vecs = [np.array(list(intensity_generator(c, axis))) for c in thr_confs]
    mixture = sum((w[j] * t for j, t in enumerate(thr_vecs)), np.zeros(n))
    deficit = exp_vec - mixture
    prob = lp.LpProblem("score", lp.LpMinimize)
    a  = [lp.LpVariable(f"a{i}",  lowBound=0) for i in range(n)]
    bp = [lp.LpVariable(f"bp{i}", lowBound=0) for i in range(n-1)]
    bn = [lp.LpVariable(f"bn{i}", lowBound=0) for i in range(n-1)]
    prob += (lp.lpSum(intervals[i] * (bp[i] + bn[i]) for i in range(n-1))
             + MTD * lp.lpSum(a))
    for i in range(n):
        flow_i    = (bp[i]   - bn[i])   if i < n-1 else 0
        flow_prev = (bp[i-1] - bn[i-1]) if i > 0   else 0
        prob += a[i] + flow_i - flow_prev == deficit[i]
    LpSolverDefault.msg = not quiet
    prob.solve(solver=LpSolverDefault)
    return lp.value(prob.objective)

MTD = 0.05

# Component
comp = [(100.0, 0.9), (101.0, 0.1)]

# Experimental: signal at 100 and 101, plus ONE noise peak within MTD of 100.
exp_raw = [(100.00, 0.9 + 0.1), (100.02, 0.5), (101.00, 0.1)]
tot = sum(i for _, i in exp_raw)
exp_confs = sorted((m, i / tot) for m, i in exp_raw)
print("Experimental (normalized):")
for m, i in exp_confs:
    print(f"  m/z {m:7.4f}  int {i:.4f}")
print()
print("Component:")
print(f"  m/z {comp[0][0]:7.4f}  int {comp[0][1]:.4f}")
print(f"  m/z {comp[1][0]:7.4f}  int {comp[1][1]:.4f}")

# ---- dualdeconv2 ---------------------------------------------------------
em = MasserSpectrum(confs=exp_confs, label="E"); em.normalize()
cm = MasserSpectrum(confs=comp, label="C");      cm.normalize()
dd2 = dualdeconv2(em, [cm], penalty=MTD, quiet=True)
w_dd2 = dd2["probs"][0]
print(f"dualdeconv2       : w={w_dd2:.4f}  fun={dd2['fun']:.6f}  "
      f"trash={sum(dd2['trash']):.4f}")

# ---- MassersteinSolver (fixed) -------------------------------------------
ew = Spectrum_1D(np.array([m for m, _ in exp_confs]),
                  np.array([i for _, i in exp_confs]), label="E")
cw = Spectrum_1D(np.array([m for m, _ in comp]),
                  np.array([i for _, i in comp]), label="C")
print(f"Wnet experimental: {str(ew)}")
print(f"Wnet component   : {str(cw)}")
res = MassersteinSolver(ew, [cw], MTD=MTD).deconvolve()
w_wnet = res["probs"][0]
print(f"MassersteinSolver : w={w_wnet:.4f}  fun_wnet={res['fun']:.6f}  "
      f"(wnet's own cost model)")
print()

# ---- Cross-score both w's in masserstein's actual LP ---------------------
v_at_dd2  = score_at_w(em, [cm], [w_dd2],  MTD, quiet=True)
v_at_wnet = score_at_w(em, [cm], [w_wnet], MTD, quiet=True)
print(f"masserstein LP at w_dd2   = {v_at_dd2:.6f}  (sanity: dd2.fun = "
      f"{dd2['fun']:.6f})")
print(f"masserstein LP at w_wnet  = {v_at_wnet:.6f}  "
      f"<-- ratio vs dd2's optimum = {v_at_wnet / dd2['fun']:.2f}x worse")
