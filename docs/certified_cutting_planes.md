# Certified cutting planes

`optimize()` now uses cutting planes with `polish=False`. Select a
NetworkSimplex configuration to use the certified dual-cut oracle:

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

All transport evaluations use wnet/pylmcf integer network simplex. The small
Kelley master LP only optimizes mixture weights and the objective epigraph.
Node potentials provide one common matching dual; it produces supporting cuts
for continuous supplies with the graph's fixed quantized cost coefficients.
For the construction and supply-error proof, see wnet's `docs/dual_cuts.md`.

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
reselects a safe cost scale, discards the old basis and every previous cut and
bound, then starts the model again. Up to eight refinements are allowed within
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
1D transport backends remain explicitly selectable; NetworkSimplex is now the
default in all dimensions so default fits can certify their stopping bounds.
