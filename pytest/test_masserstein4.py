"""MassersteinSolver4 (dualdeconv4 reproduction, independent trash).

Skipped when the installed wnet predates add_independent_asymmetric_trash
(wnet < 1.4.0). Expected values below were produced by masserstein's own
dualdeconv4 on the identical inputs (penalty=0.5, penalty_th=1.0):
probs = [0.8, 0.0], fun = 0.26.
"""

import numpy as np
import pytest

from wnetdeconv import Spectrum_1D
from wnetdeconv.solver import _wnet_supports_independent_trash

pytestmark = pytest.mark.skipif(
    not _wnet_supports_independent_trash(),
    reason="installed wnet lacks add_independent_asymmetric_trash (< 1.4.0)",
)


def test_matches_dualdeconv4_reference():
    from wnetdeconv import MassersteinSolver4

    emp = Spectrum_1D([100.0, 100.4, 107.0], [2.0, 2.0, 1.0])
    t1 = Spectrum_1D([100.0], [3.0])
    t2 = Spectrum_1D([104.0], [1.0])
    solver = MassersteinSolver4(emp, [t1, t2], MTD=0.5, MTD_th=1.0)
    res = solver.deconvolve()
    # w2 must be exactly zero: with independent trash, mass for the
    # unmatchable component costs the full theoretical penalty per unit
    # (the annihilating model would let it float for free against noise).
    assert res["probs"][1] == pytest.approx(0.0, abs=1e-6)
    assert res["probs"][0] == pytest.approx(0.8, abs=0.01)
    assert res["fun"] == pytest.approx(0.26, rel=1e-3)
