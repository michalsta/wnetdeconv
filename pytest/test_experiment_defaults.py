"""Ordinary experiment helpers use the public cutting-plane optimizer."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from wnetdeconv import DeconvSolver
from test_dual_cutting_plane import make, TRUTH


def load_experiment(name):
    path = Path(__file__).resolve().parents[1] / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_experiment_records_cutting_plane_trajectory(monkeypatch):
    helpers = load_experiment("experiments_support")
    monkeypatch.setattr(helpers, "minimize", lambda *a, **kw: pytest.fail("Unexpected descent"))
    solver = make(DeconvSolver)
    original = solver.set_point
    result, trajectory = helpers.run_optimization(
        solver, np.ones(3), [(0, 2)] * 3, max_iterations=100,
    )
    assert result.success and result.bound_certified
    assert trajectory.ndim == 2 and trajectory.shape[1] == 3
    np.testing.assert_allclose(trajectory[-1], result.x)
    np.testing.assert_allclose(result.x, TRUTH, atol=1e-6)
    assert solver.set_point == original


@pytest.mark.parametrize("module", ["experiments_support", "convex"])
def test_global_reference_search_uses_cutting_planes(module):
    helpers = load_experiment(module)
    result = helpers.find_global_optimum(make(DeconvSolver), [(0, 2)] * 3, n_starts=1)
    assert result.success and result.bound_certified
    np.testing.assert_allclose(result.x, TRUTH, atol=1e-6)
