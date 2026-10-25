# Certified cutting planes

`optimize()` now uses cutting planes with `polish=False`. Select a
NetworkSimplex, SlopeDP or ConvexSweep configuration to use the certified dual-cut oracle:

```python
from wnet.distances import DistanceMetric
from wnet.wnet_cpp import NetworkSimplex
from wnetdeconv import DeconvSolver, Spectrum_1D

empirical = Spectrum_1D([0, 1], [0.7, 0.3])
components = [Spectrum_1D([0], [1]), Spectrum_1D([1], [1])]
solver = DeconvSolver(
    empirical, components, DistanceMetric.L1,
    max_distance=0.1,
    experimental_trash_cost=0.1,
    theoretical_trash_cost=0.03,
    solver=NetworkSimplex(),
)
result = solver.optimize(tol=1e-8)
print(result.x, result.success, result.gap)
```

Transport evaluations use the selected integer backend. The small
Kelley master LP only optimizes mixture weights and the objective epigraph.
Node potentials provide one common matching dual; it produces supporting cuts
for continuous supplies with the graph's fixed quantized cost coefficients.
For the construction and supply-error proof, see wnet's `docs/dual_cuts.md`.

Which point is evaluated next is set by `solver.cutting_plane_master`. The
default, `"in_out"`, evaluates `in_out_alpha * best + (1 - in_out_alpha) *
kelley` (alpha 0.5): halfway between the best evaluated point and the master
LP's minimizer, or the minimizer itself right after a cut fails to raise the
model there. `"kelley"` always evaluates the minimizer; `"level_l1"` /
`"level_linf"` take a level-bundle projection (one extra LP per iteration).
The lower bound and stopping test are the same under every master. On the
prod_bench deconvolution jobs in-out took 0.78x Kelley's time (geometric
mean, x86-64-v3; 0.79x on AVX-512), up to 4x less where Kelley ran into its
iteration cap, and at most 1.25x more on any job.

For 1D `p > 1`, explicitly pass `solver=ConvexSweep()` and the desired `p`.
ConvexSweep supplies a common feasible dual through templated DP traceback
and native potential recovery. Cutting planes automatically request
`with_duals=True`; ordinary cost evaluations omit traceback and recovery.
Certificates are validated eagerly and cached for cut queries. Traceback can
require quadratic temporary memory for wide windows, while potential recovery
scans profitable windows without storing their pair graph. Its discrete
marginal derivatives alone are not used as certified cuts. The selected
transport backend is preserved.

The result separates the rounded cost (`fun`) from its continuous-supply upper
bound (`upper_bound`). `rounding_error` bounds their difference, while `lb` is
a dual-model lower bound over the configured finite budget bounds and mass/face
constraints. `gap = upper_bound - lb`. These bounds refer to fixed quantized
cost coefficients, not exact original real cost coefficients. Floating-point
accumulation cushions and LP feasibility tolerances still apply.

The master lower bound is evaluated from nonnegative cut multipliers and
standing-constraint multipliers, minimizing the resulting affine Lagrangian
over the finite box. It does not trust a possibly inaccurate LP primal
objective as a lower bound. `bound_certified=True` identifies this path.

When rounding uncertainty prevents progress, the solver increases intensity
precision by ten, retaining the graph semantics and simplex variant. It
reselects a safe cost scale and discards the old basis and every upper bound.
Previous cuts bound the old quantized objective, so they are carried over
with wnet's point-independent transfer bounds (`dual_cut()`'s `transfer_*`
keys: relative round-up rho, relative round-down sigma and absolute coordinate
error A, with `(1 - sigma) R - A <= Q <= (1 + rho) R + A` against the real-cost
objective R):

    cut_new(w) = (1 - sigma_new) (cut_old(w) - A_old) / (1 + rho_old) - A_new

Real-cost cuts (`real_cost_certificate`) are kept unchanged. With a wnet that
does not report the transfer bounds every cut is discarded instead.
`n_cuts_transferred` counts carried cuts. Up to eight refinements are allowed within
the `max_iter` evaluation budget. `n_precision_refinements` reports rebuilds;
`precision_limit`, `stalled`, or `max_iter` do not imply success. The public
`solver.refine_intensity_precision(factor)` allows explicit precision changes.

Success requires a nonnegative certified gap no larger than the requested
positive absolute `tol` (default 1e-9). Polishing may improve the returned
rounded cost but cannot grant success without that gap certificate. The solver
is left at the returned point, so a fresh `total_cost()` agrees with `fun`.

Other backends retain the legacy marginal-cut search with
`bound_certified=False` and `success=False`; their model gap is diagnostic.
No backend is silently replaced. Existing descent APIs retain their behavior.

The original aromatic cases and an analytic mass-balance kink are reproduced
by the tests. `wnet_smietnik/nmr_deconv/benchmark_dual_oracles.py` benchmarks
fresh solve-and-oracle sequences on the actual aromatic data, standard and
LinkCut simplex, and default/100x initial supply precision. Each returned bound
is checked against a known feasible mixture, and each success against its gap.

`optimize_descent()` preserves the previous descent API. `optimize()` accepts
`x0`, custom `bounds`, `maxiter` (oracle budget), `tol`, `print_steps`, and
optional `polish=True`. Custom bounds and mass constraints also apply during
polishing. `optimize_cutting_plane()` retains its `max_iter` spelling and now
also defaults to no polishing. Masserstein `deconvolve()` uses the same default
cutting-plane optimizer and preserves its dictionary return format. Analytic
1D transport backends remain explicitly selectable. The default is SlopeDP
for 1D `p == 1` chains and NetworkSimplex otherwise.
