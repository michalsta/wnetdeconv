# Cost Map 2D - Configuration Guide

## Overview

The `cost_map_2d.py` script compares different optimization methods for 2D spectrum deconvolution. It can generate detailed visualizations and/or collect statistics across multiple runs with different random spectra.

## Configuration Constants

All configuration is done via constants at the top of the file. Modify these before running:

### Control Flags

```python
ENABLE_PLOTTING = True  # Set to False to disable all plot generation
NUM_RUNS = 1           # Number of runs with different random spectra
```

- **ENABLE_PLOTTING**: When `False`, skips all matplotlib operations and only outputs diagnostic logs
- **NUM_RUNS**: Run multiple times to collect statistics. When > 1, a summary table is printed at the end

### Optimization Settings

```python
START_POINT = [15.0, 20.0]          # Starting point for optimization
BOUNDS = [(0, 25), (0, 25)]         # Bounds for each dimension
MAX_ITERATIONS = 200                 # Maximum iterations for optimization
GLOBAL_SEARCH_STARTS = 20            # Number of random starts for global optimum search
```

### Grid Computation Settings

```python
GRID_RESOLUTION = 200                            # Number of points per axis for cost grid
ZOOM_LEVELS = [10, 100, 1000, 10000, 100000]    # Zoom levels to compute
```

- **GRID_RESOLUTION**: Higher values = more accurate but slower (200 is a good balance)
- **ZOOM_LEVELS**: Each zoom level creates a new row in the visualization

### Plotting Settings

```python
FIGURE_SIZE = (30, 30)    # Figure size in inches
DPI = 150                 # Resolution for saved plots
ARROW_SUBSAMPLE = 20      # Show every Nth arrow in gradient field
ARROW_ALPHA = 0.6         # Transparency of gradient arrows
```

These are only used if `ENABLE_PLOTTING = True`.

### Optimization Methods

```python
METHODS = [
    ('L-BFGS-B', 'Limited-memory BFGS with Bounds'),
    ('TNC', 'Truncated Newton Conjugate-Gradient'),
    ('SLSQP', 'Sequential Least Squares Quadratic Programming'),
    ('Nelder-Mead', 'Nelder-Mead Simplex'),
    ('Powell', 'Powell Direction Set'),
    ('COBYLA', 'Constrained Optimization BY Linear Approximation'),
]
```

Comment out methods you don't want to test.

### Deconvolution Solver Parameters

```python
DISTANCE_METRIC = DistanceMetric.LINF
MAX_DISTANCE = 10
TRASH_COST = 100
SCALE_FACTOR = 1000
```

## Usage Examples

### Example 1: Generate visualizations (default)

```bash
python cost_map_2d.py
```

This will:
- Run all 6 optimization methods
- Generate 6 PNG files (one per method)
- Each file shows 6 zoom levels × 5 columns
- Print diagnostic output to console

### Example 2: Statistics only (no plots)

Edit the file and change:
```python
ENABLE_PLOTTING = False
NUM_RUNS = 10
```

Then run:
```bash
python cost_map_2d.py
```

This will:
- Run 10 different random spectra
- Test all methods on each
- Print summary statistics at the end
- No plots generated (much faster)

### Example 3: Quick test with fewer zoom levels

Edit the file and change:
```python
ZOOM_LEVELS = [10, 100]  # Only 2 zoom levels
GRID_RESOLUTION = 100     # Lower resolution
```

This runs much faster for testing.

### Example 4: Test only gradient-based methods

Edit the file and change:
```python
METHODS = [
    ('L-BFGS-B', 'Limited-memory BFGS with Bounds'),
    ('TNC', 'Truncated Newton Conjugate-Gradient'),
    ('SLSQP', 'Sequential Least Squares Quadratic Programming'),
]
```

## Output Files

### When ENABLE_PLOTTING = True

For each method, a PNG file is generated:
- `cost_map_2d_L-BFGS-B.png`
- `cost_map_2d_TNC.png`
- `cost_map_2d_SLSQP.png`
- `cost_map_2d_Nelder-Mead.png`
- `cost_map_2d_Powell.png`
- `cost_map_2d_COBYLA.png`

If `NUM_RUNS > 1`, files include run number: `cost_map_2d_L-BFGS-B_run001.png`

### Console Output

Always printed:
- Configuration summary
- Per-method convergence information (iterations, final cost, success)
- Global optimum found via multi-start

When `NUM_RUNS > 1`:
- Statistics summary table showing mean/std/min/max for iterations and final cost
- Success rate for each method

## Understanding the Visualization

Each PNG file has:
- **6 rows**: Different zoom levels (1x, 10x, 100x, 1000x, 10000x, 100000x)
- **5 columns**:
  1. Cost landscape + numerical gradient walk
  2. Numerical gradient magnitude + arrows + numerical walk
  3. Cost landscape + analytical gradient walk
  4. Analytical gradient magnitude + arrows + analytical walk
  5. Difference between numerical and analytical gradients

### Markers

- **Red circle**: Starting point
- **Cyan X**: Scipy optimization endpoint
- **Magenta diamond**: Global optimum (from multi-start search)
- **Yellow square**: Grid minimum (best cost found on discretized grid)

### Trajectories

- Black-to-white gradient shows optimization path
- Black = start, White = end

## Performance Tips

1. **For quick testing**: Set `GRID_RESOLUTION = 50`, `ZOOM_LEVELS = [10]`
2. **For statistics**: Set `ENABLE_PLOTTING = False`, `NUM_RUNS = 20`
3. **For publication plots**: Set `DPI = 300`, `GRID_RESOLUTION = 400`
4. **To test specific method**: Comment out others in `METHODS` list

## Notes

- Gradient-free methods (Nelder-Mead, Powell, COBYLA) show the same trajectory in both "numerical" and "analytical" columns since they don't use gradients
- The "maxiter" warning is expected and doesn't affect results
- Grid computation is the slowest part - reducing `GRID_RESOLUTION` or `ZOOM_LEVELS` speeds up significantly
