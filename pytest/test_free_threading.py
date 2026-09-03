"""The free-threaded build must actually stay free-threaded.

On CPython 3.15t the extension is built in split mode with nanobind's
FREE_THREADED option, which declares ``Py_MOD_GIL_NOT_USED``. If that option is
ever dropped, or the build quietly falls back to a linked mode, importing the
extension re-enables the GIL and the wheel is free-threaded in name only --
nothing else in the suite would notice, because every test still passes.

The whole module is skipped unless the GIL is genuinely off *after* importing
the extension. That is the point: on 3.14t the linked fallback is expected to
turn the GIL back on, so this correctly stays quiet there rather than failing.

wnetdeconv registers no classes of its own -- it drives wnet's networks from
Python -- so a solve here exercises two extensions and the shared nanobind
backend at once. The gradient is checked alongside the cost because it is the
part that caches state between calls.
"""

import sys
import threading

import numpy as np
import pytest

from wnet.distances import DistanceMetric
from wnetdeconv import DeconvSolver, Spectrum_1D

pytestmark = pytest.mark.skipif(
    not hasattr(sys, "_is_gil_enabled") or sys._is_gil_enabled(),
    reason="needs a free-threaded interpreter with the GIL still disabled after import",
)

N_POINTS = 40
N_THREADS = 8
N_ROUNDS = 25


def _problem():
    # Seeded, so every call below has the same right answer.
    rng = np.random.default_rng(0)
    e_pos = np.sort(rng.uniform(0.0, 50.0, N_POINTS))
    e_int = rng.integers(1, 20, N_POINTS).astype(float)
    theo = []
    for seed in (1, 2):
        r = np.random.default_rng(seed)
        idx = np.sort(r.choice(N_POINTS, N_POINTS // 2, replace=False))
        theo.append((e_pos[idx] + r.uniform(-0.2, 0.2, N_POINTS // 2),
                     r.integers(1, 5, N_POINTS // 2).astype(float)))
    return (e_pos, e_int), theo


def _solve_once(problem):
    # A fresh solver per call: the claim under test is that the module holds no
    # *global* state, not that one solver may be driven from two threads.
    (e_pos, e_int), theo = problem
    solver = DeconvSolver(
        empirical_spectrum=Spectrum_1D(e_pos.copy(), e_int.copy()),
        theoretical_spectra=[Spectrum_1D(p.copy(), i.copy()) for p, i in theo],
        distance=DistanceMetric.L1,
        max_distance=10,
        trash_cost=100,
        scale_factor=1e5,
    )
    solver.set_point([1.0, 1.0])
    return solver.total_cost(), tuple(solver.gradient())


def test_gil_stays_disabled_after_import():
    assert not sys._is_gil_enabled()


def test_concurrent_deconvolutions_agree_with_serial():
    problem = _problem()
    expected = _solve_once(problem)

    results = []
    errors = []

    def worker():
        try:
            for _ in range(N_ROUNDS):
                results.append(_solve_once(problem))
        except BaseException as exc:  # noqa: BLE001 - re-raised in the assert below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors[:3]
    assert len(results) == N_THREADS * N_ROUNDS
    assert all(r == expected for r in results)
    # If the GIL had been re-enabled behind our back, the run above proves nothing.
    assert not sys._is_gil_enabled()
