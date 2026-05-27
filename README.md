# wnetdeconv

Spectral deconvolution via Wasserstein optimal transport.

Given an empirical spectrum and a library of theoretical component spectra,
`wnetdeconv` finds the mixture proportions that minimise the total Wasserstein
transport cost between the empirical signal and the weighted sum of components.
The inner problem at each set of proportions is solved exactly as a min-cost
flow (via [pylmcf](https://github.com/michalsta/pylmcf) / LEMON), giving an
exact piecewise-linear objective with exact gradients — suitable for gradient-
based outer optimisation with scipy.

Supports 1-D spectra (NMR chemical shift, m/z) and higher-dimensional data
(e.g. m/z + retention time).

## Installation

```bash
pip install wnetdeconv
```

Dependencies: `pylmcf`, `wnet`, `numpy`, `scipy`.
Optional: `pyopenms` for loading featureXML files.

## Concepts

### Spectra as distributions

A spectrum is a set of *(position, intensity)* pairs.  In 1-D (NMR chemical
shift, m/z) use `Spectrum_1D`; for higher-dimensional data (m/z + retention
time) use `Spectrum` with a `(d, n)` positions array.

```python
from wnetdeconv import Spectrum_1D

empirical = Spectrum_1D([1.0, 2.0, 3.0], [10.0, 25.0, 15.0])
component = Spectrum_1D([1.0, 2.0, 3.0], [1.0, 2.0, 1.0])
```

### Transport cost

Matching a unit of intensity from an empirical peak at position *p* to a
theoretical peak at position *q* costs `distance(p, q)`.  Peaks that cannot
be matched cheaply are instead routed to a *trash node* at a fixed penalty.

`max_distance` caps the farthest match considered; anything farther is cheaper
to trash.  `trash_cost` (or the asymmetric pair
`experimental_trash_cost` / `theoretical_trash_cost`) sets that penalty.

### Precision and scaling

Internally all intensities and costs are scaled to integers for the MCF solver.
The `precision` parameter (default `1e-3`) sets the desired relative accuracy
of the cost output: `precision=1e-3` gives ≈ 3 significant figures.  The same
value becomes the `ftol` stop criterion for scipy optimisers, so the outer loop
stops as soon as further improvement is below the resolution the integer network
can deliver.

## Solvers

### `DeconvSolver` — unconstrained baseline

Solves the network at a given point and exposes `total_cost()` and
`gradient()`.  Optimisation (via `optimize()`, L-BFGS-B) minimises cost with
only non-negativity bounds.

```python
from wnetdeconv import DeconvSolver, Spectrum_1D
from wnet.distances import DistanceMetric

emp  = Spectrum_1D([1.0, 100.0], [10.0, 30.0])
t1   = Spectrum_1D([1.0],        [2.0])   # optimal proportion: 5
t2   = Spectrum_1D([100.0],      [3.0])   # optimal proportion: 10

solver = DeconvSolver(
    empirical_spectrum=emp,
    theoretical_spectra=[t1, t2],
    distance=DistanceMetric.LINF,
    max_distance=10.0,
    trash_cost=100.0,
)

result = solver.optimize()
print(result.x)   # [5. 10.]
```

You can also drive the solver manually — useful when embedding it in your own
optimisation loop:

```python
solver.set_point([5.0, 10.0])
print(solver.total_cost())   # 0.0
print(solver.gradient())     # [0. 0.]  (at the optimum)
```

### `ConstrainedSolver` — total-mass equality

Adds the constraint `Σ wₛ · Iₛ = I_emp` so that the mixture exactly accounts
for all empirical intensity.  Uses SLSQP.  Drop-in replacement for
`DeconvSolver`; call `optimize()` the same way.

```python
from wnetdeconv import ConstrainedSolver

solver = ConstrainedSolver(
    empirical_spectrum=emp,
    theoretical_spectra=[t1, t2],
    distance=DistanceMetric.LINF,
    max_distance=10.0,
    trash_cost=100.0,
)
result = solver.optimize()
```

## Key parameters

| Parameter | Applies to | Description |
|---|---|---|
| `max_distance` | all | Maximum peak-to-peak match distance. Also sets the sparsity of the internal network in 1-D. |
| `trash_cost` | all | Symmetric penalty for unmatched peaks. |
| `experimental_trash_cost` | `DeconvSolver` | Per-unit penalty for discarding empirical mass. |
| `theoretical_trash_cost` | `DeconvSolver` | Per-unit penalty for discarding theoretical mass. |
| `precision` | all | Desired relative cost accuracy; drives `scale_factor` and `ftol` (default `1e-3`). |
| `scale_factor` | all | Override automatic scaling (bypasses `precision`). |

## Distance metrics

From `wnet.distances.DistanceMetric`:

- `L1` — sum of absolute coordinate differences (Manhattan / taxicab)
- `L2` — Euclidean distance
- `LINF` — maximum absolute coordinate difference (Chebyshev); dual of the W₁ earth-mover distance used by masserstein

## Loading MS data (featureXML)

```python
from wnetdeconv import Spectrum

emp = Spectrum.FromFeatureXML("sample.featureXML")   # requires pyopenms
```

## Architecture

```
wnetdeconv
├── Spectrum / Spectrum_1D   — data containers (extend wnet.Distribution)
├── DeconvSolver             — core: builds WassersteinNetwork, exposes cost + gradient
└── ConstrainedSolver        — adds total-mass equality, uses SLSQP
```

The underlying min-cost flow is provided by
[wnet](https://github.com/michalsta/wnet) (network construction) and
[pylmcf](https://github.com/michalsta/pylmcf) (LEMON-based MCF algorithms,
including warm-restart Network Simplex).

## License

MIT
