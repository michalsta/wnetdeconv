# Replacing the magnetstein LP with graph-based transport

Status: algorithm/design record, saved 2026-10-19. The graph oracle and outer
optimizer described here are proposed work, not an implemented compatibility
backend. The equivalent continuous joint LP has been prototyped and checked
against the legacy package cores.

The objective is to reproduce the normalized, two-sided-noise
`magnetstein.deconv_simplex.dualdeconv4` and
`masserstein.deconv_simplex.dualdeconv4` formulations while using an ordinary
min-cost-flow/network-simplex solve for transport. This document targets 1-D
`p=1`, where transport has a sparse chain representation.

## 1. What theoretical trash already represents

Theoretical trash already removes unmatched signal from the weighted library.
The legacy library-noise scalar is not an additional independent noise profile.
It records total theoretical trash and also changes the normalization of the
empirical target. That normalization is the important difference from the
current independent-trash wrappers.

Use these conventions throughout:

- `e_i >= 0`: empirical spectrum, normalized so `sum_i e_i = 1`.
- `t_ij >= 0`: reference component `j` at common position `i`, with every
  reference column normalized so `sum_i t_ij = 1`.
- `w_j >= 0`: raw library coefficients; `W = sum_j w_j`.
- `(Tw)_i = sum_j t_ij w_j`: weighted library spectrum.
- `a_i >= 0`: empirical trash; `A = sum_i a_i`.
- `b_i >= 0`: theoretical trash; `B = sum_i b_i`.
- `q >= 0`: legacy library-noise fraction.
- `Df`: transport divergence on the common axis, with `sum_i (Df)_i = 0`.
- `Ce, Ct >= 0`: empirical and theoretical trash costs per unit mass.

Here `w` contains spectral-area coefficients. Any molar/proton conversion is a
separate reporting operation.

The current independent-trash balance is

```text
Tw + a - b + Df = e.
```

Summing gives `W + A - B = 1`. Consequently:

- Fixed mass, `W = 1`, forces `A = B`.
- Free mass, `W <= 1`, forces `A >= B`.

Neither choice by itself reproduces the legacy normalization convention.

## 2. The legacy primal formulation

Dualizing the legacy `dualdeconv4` core gives

```text
minimize   transport_cost(f) + Ce * A + Ct * q

subject to Tw + q * e + a - b + Df = e
           W + A = 1
           B = q
           0 <= q <= W
           w, a, b >= 0.
```

Equivalently, the spatial balance is

```text
Tw + a - b + Df = (1 - q) * e.
```

The empirical target is rescaled proportionally; `q` is not a freely shaped
baseline or background spectrum. Because `q = B`, the term `Ct * q` is already
the theoretical-trash charge. Do not add a second `Ct * B` charge.

The implied domain for the outer variables is

```text
w_j >= 0,       0 <= q <= W <= 1.
```

For example, suppose a normalized reference contains 90% useful signal and
10% unwanted library signal, while the empirical spectrum is clean. At `W=1`,
the fixed-mass independent-trash model must discard equal amounts on both
sides. The legacy model can instead use `q=0.1`, discard the unwanted 10% of
the reference, and compare its useful 90% against `0.9 * e`, with no empirical
trash. In multi-component mixtures this changes both the objective and the
preferred proportions.

## 3. Reduce fixed `(w, q)` to an ordinary transport problem

For fixed outer variables, the trash amounts and matched mass are known:

```text
A = 1 - W
B = q
M = W - q.
```

The available empirical mass is `1-q`; the available theoretical mass is `W`.
Matching exactly `M` leaves

```text
empirical remainder   = (1-q) - M = 1-W = A
theoretical remainder = W - M     = q   = B.
```

Thus the inner problem is minimum-cost partial transport of an **exactly
specified amount**, rather than transport with freely chosen trash amounts.
Its objective is augmented by the constant

```text
K(w, q) = Ce * (1-W) + Ct * q.
```

The domain guarantees `0 <= M <= min(1-q, W)`. With the complete transport
support, every outer-feasible point has an inner-feasible transport problem.

## 4. Network-simplex graph

The direct bipartite graph has these arcs:

| Arc | Capacity | Unit cost |
|---|---|---|
| Source -> empirical point `i` | `(1-q) * e_i` | 0 |
| Empirical `i` -> theoretical point `l` | Unlimited | `abs(x_i-y_l)` |
| Theoretical point `l` -> sink | `(Tw)_l` | 0 |

Give the source supply `M`, the sink demand `M`, and all other nodes zero
supply. Run network simplex, then return

```text
F(w, q) = minimum_transport_cost + K(w, q).
```

No explicit trash arcs are needed in this construction: the unused endpoint
capacities already have exactly the required totals. Their costs are in `K`.

The exact-flow requirement is essential. Merely replacing `e` by `(1-q)e` in
the current independent-trash oracle, while allowing both trash totals to vary,
solves a different optimization problem.

### Sparse chain for 1-D p=1

Let `z_0 < ... < z_(N-1)` be the sorted union of all positions. Aggregate
coincident empirical and theoretical mass at the same position. Use one graph
node per position and:

- Source -> position `i`, capacity `(1-q)e_i`, zero cost.
- Position `i` -> sink, capacity `(Tw)_i`, zero cost.
- Both directions between adjacent positions, unlimited capacity, cost
  `z_(i+1)-z_i` per unit.

The same fixed source/sink supply `M` applies. A path's cost is the sum of its
gaps, equal to the absolute endpoint distance. The graph has `O(N)` nodes and
arcs; reference-capacity updates also require processing the nonzeros of `T`.
This is a sparsity statement, not a claim of linear network-simplex runtime.

For integer costs, use differences of rounded coordinates, not independently
rounded gaps, as in the scaled p=1 coordinate fix in wnet. This preserves the
path-length approximation across densely sampled profile points.

Do not assume the existing analytic-trash SlopeDP/ConvexSweep entry points
already implement this oracle. They normally choose matched mass themselves.
Network simplex with explicit fixed flow is the first implementation target;
specialized fixed-mass DP support can be investigated later.

### Topology and decomposition restrictions

Start with the complete bipartite support or the full adjacent-position chain.
Do not reuse a match cutoff of `Ce+Ct`, or split/delete chain regions under the
ordinary independent-trash profitability argument, without a proof for this
different formulation. A particular fixed `(w,q)` may require long-distance
transport to satisfy its fixed `M`.

Likewise, do not solve existing geometric subgraphs independently with their
usual local trash budgets. The total matched mass is a global constraint.
Source and sink must coordinate allocation among regions, or an additional
allocation/master problem must enforce that coupling.

## 5. Outer optimization and dual cuts

The outer problem is

```text
minimize F(w, q)
subject to w >= 0,  q >= 0,  q <= sum(w),  sum(w) <= 1.
```

Only one variable is added to the existing mixture-weight optimization. For
fixed geometry, all inner capacities and the required flow are affine in
`(w,q)`. The exact value function is convex and piecewise linear, so Kelley
cutting planes remain applicable.

For clarity, consider the bipartite partial-transport LP:

```text
minimize sum_(i,l) d_il * f_il

subject to sum_l f_il <= (1-q)e_i
           sum_i f_il <= (Tw)_l
           sum_(i,l) f_il = W-q
           f_il >= 0.
```

Use dual variables `u_i <= 0` for empirical capacities, `v_l <= 0` for
theoretical capacities, and a free `lambda` for the flow equality. Dual
feasibility is

```text
u_i + v_l + lambda <= d_il.
```

Any feasible dual gives a global affine lower bound after adding the trash
constant. In particular, define

```text
intercept = Ce + sum_i u_i * e_i
g_w[j]    = sum_l v_l * t_lj + lambda - Ce
g_q       = -sum_i u_i * e_i - lambda + Ct.
```

Then

```text
F(w, q) >= intercept + g_w . w + g_q * q.
```

An optimal dual makes the bound tight at the evaluated point in exact
arithmetic. The `lambda` terms matter: changing either `w` or `q` changes the
required transported amount, not just capacities.

Extract an equivalent dual from the chain network, with signs checked against
the bipartite LP. Do not assume that raw node potentials alone equal the
capacity multipliers, or reuse the existing free-matched-mass trash gradient
unchanged. Capacity bounds and the source/sink supply derivative all contribute.

### Cutting-plane loop

1. Choose feasible `(w,q)`, for example uniform weights with `W=1, q=0`, or a
   feasible warm start from a previous fit.
2. Update graph capacities and source/sink supply; solve transport.
3. Record a feasible objective upper bound and obtain a valid dual cut.
4. Add the cut to a master LP over `(w,q,z)`, with the outer domain above and
   `z >= intercept_r + g_w_r . w + g_q_r * q` for every stored cut. The global
   bound `z >= 0` is valid for nonnegative transport/trash costs.
5. Solve the master; its optimum supplies the next point and a lower bound.
6. Stop only when the certified upper/lower gap satisfies the requested
   tolerance, including numerical allowances.

Keep the best feasible evaluated point. Rebuilding a quantized cost model
invalidates old cuts unless they are explicitly transferred with a proof.
Iteration limits or precision failures must remain distinguishable from
certified convergence.

## 6. Numerical contract

The mathematical equivalence is for continuous masses and specified costs.
The current native transport backend uses integer flows/costs, so numerical
compatibility requires more work than adding `q` to the parameter vector.

In particular, independently flooring endpoint capacities and rounding `M`
can make a feasible continuous point infeasible in the integer graph. Even
when `M` is rounded down, its value can exceed the sum of separately floored
capacities. Handle this deliberately; do not silently lower transported mass
and report the result as the original legacy objective.

A possible certified construction is to choose an integer transported amount
that fits the floored capacities, then repair the missing flow in the original
continuous graph to obtain an upper bound. On complete support/full chains,
remaining endpoint capacity exists by the domain constraints, and a maximum
path cost can bound the repair cost. Include endpoint-cost quantization error
and charge the original `K(w,q)`. Validate the construction independently.

For lower cuts, recover a common feasible dual and evaluate it against the
original continuous capacities and flow amount. Derive cost-grid allowances
for this fixed-flow model. The current independent-trash supply-error bounds
and dual clipping thresholds are not automatically valid: a forced unit of
transport can cost more than the ordinary trash escape price.

Position changes, cost rescaling, supply refinement, warm restarts, zero
capacities, coincident positions, and dual-cache invalidation all need tests.
For an initial correctness reference, use the continuous joint LP below.

## 7. Fully joint LP alternative and fallback

The full formulation in section 2 is itself a sparse LP. In 1-D p=1, put the
two directed flow variables for each adjacent gap, `w`, `a`, `b`, and `q` into
one LP and optimize everything together. Its balance matrix contains the
library columns `T` and the empirical column `e` multiplying `q`.

This is not one ordinary network-simplex problem: ordinary graph conservation
cannot enforce that one variable scales many capacities in prescribed ratios.
Mixture weights already create the same kind of coupling across reference
peaks. A general sparse LP solver can handle these columns directly.

The joint LP is useful as:

- An independent formulation/duality oracle for the native implementation.
- A production p=1 backend when its size is reasonable.
- A fallback when cutting planes stall or reach precision/iteration limits.
- A source of feasible warm-start weights and `q` for the outer algorithm.

The proposed fallback must explicitly choose the same noise model as the fit.
The fixed-mass and free-mass independent-trash joint LPs are useful alternatives,
but substituting one of them does not establish legacy equivalence.

A previous NIST experiment demonstrated that continuous joint chain LPs can be
practical for many-component data. It does not establish universal superiority
on very large NMR profiles. Benchmark graph size, memory, oracle/master time,
and convergence before selecting an automatic crossover policy.

## 8. Reporting and degenerate cases

Return raw `w`, `W`, `q`, empirical trash `1-W`, matched mass `W-q`, objective,
and convergence/certificate diagnostics. Report relative library proportions
as `w/W` only when `W > 0`, and retain the unnormalized quantities.

Do not introduce `q < 1` as an undocumented numerical restriction: the legacy
LP allows the boundary `q=W=1`. With `W-q=0`, no mass matches and the coefficient
split need not be identifiable. With `W=0`, normalized proportions are undefined.
Expose these cases rather than manufacture a meaningful composition.

Model equivalence concerns the normalized legacy core LP. Public legacy APIs
also perform filtering, chunking, normalization, rounding, and possibly
area-to-molar conversion. Validate those separately when promising API parity.
Under nonunique optima, equal optimal objectives and valid coefficients do not
require bitwise-identical coefficient splits.

## 9. Implementation and validation plan

1. Preserve separate, explicit model choices: fixed-mass independent trash,
   free-mass independent trash, and legacy-normalized two-sided noise.
2. Add a continuous joint-LP reference for the legacy formulation, including
   objective values, primal/dual feasibility, and raw coefficient outputs.
3. Implement the fixed-`(w,q)` graph oracle with a global matched-mass supply.
4. Derive and verify its dual cuts, then connect the extra variable and domain
   constraints to the cutting-plane master.
5. Add integer rounding/repair bounds and validate them against continuous
   problems before treating native results as certified.
6. Add the general joint-LP backend/fallback with explicit resource limits.
7. Correct compatibility documentation and benchmark the completed alternatives.

Validation should include:

- Random small problems checked against both legacy package cores and the
  independent joint LP. Check objective, feasibility, noise totals, matched
  mass, and coefficients where the optimum is unique.
- The 90%-signal/10%-library-noise example, asymmetric noise, and mixtures with
  noise predominantly on either side.
- All boundaries: `q=0`, `q=W`, `W=0`, `W=1`, and `q=W=1`.
- Global cut validity at points other than the one used for recovery,
  including probes with a different matched mass and different active bounds.
- Long required transport paths, separated spectral regions, coincident
  points, and zero-supply intermediate nodes.
- Quantization-induced infeasibility, repaired upper bounds, scale changes,
  updates, warm starts, and invalidation of cached duals/cuts.
- Deposited NMR mixtures, synthetic MS, and NIST; compare methods at identical
  preprocessing and penalties and separate converged from incomplete fits.

The investigation's derived continuous legacy primal was already compared with
60 random problems through each of the two package cores (120 comparisons),
with maximum raw-weight difference around `1e-12`. This supports the formulation;
it is not a claim that the proposed graph oracle or native rounding contract
has already been implemented and validated.
