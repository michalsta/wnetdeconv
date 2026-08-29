"""MagnetsteinSolver trash-model semantics.

With MTD_th given, the default is independent two-sided trash (dualdeconv3/4:
no transport between the trash points); independent_trash=False restores the
annihilating asymmetric model.  A single peak pair at distance d with
min(MTD, MTD_th) < d < MTD + MTD_th separates the two (within one chain
component): the independent model matches the pair and pays d, while the
annihilating model dumps both sides at the min(MTD, MTD_th) discount.
"""

import pytest

from wnet.distances import DistanceMetric
from wnetdeconv import MagnetsteinSolver, Spectrum_1D

MTD = 0.3
MTD_TH = 0.5
D = 0.4  # min(MTD, MTD_TH) < D < MTD + MTD_TH


def _solver(**kwargs):
    emp = Spectrum_1D([0.0], [1.0])
    theo = Spectrum_1D([D], [1.0])
    return MagnetsteinSolver(
        emp, [theo], DistanceMetric.L1, MTD=MTD, MTD_th=MTD_TH, **kwargs
    )


def test_default_is_independent():
    s = _solver()
    s.set_point([1.0])
    assert s.total_cost() == pytest.approx(D, rel=1e-6)


def test_annihilating_opt_out():
    s = _solver(independent_trash=False)
    s.set_point([1.0])
    assert s.total_cost() == pytest.approx(min(MTD, MTD_TH), rel=1e-6)


def test_independent_chain_matches_dense():
    emp = Spectrum_1D([0.0, 0.1, 0.35, 2.0], [2.0, 1.0, 1.0, 0.5])
    theos = [
        Spectrum_1D([0.05, 0.4], [1.0, 1.0]),
        Spectrum_1D([1.9], [1.0]),
    ]
    chain = MagnetsteinSolver(emp, theos, DistanceMetric.L1, MTD=MTD, MTD_th=MTD_TH)
    dense = MagnetsteinSolver(
        emp, theos, DistanceMetric.L1, MTD=MTD, MTD_th=MTD_TH,
        method="cost_scaling",
    )
    for pt in ([0.5, 0.5], [0.8, 0.2], [0.25, 0.75]):
        chain.set_point(pt)
        dense.set_point(pt)
        assert chain.total_cost() == pytest.approx(dense.total_cost(), rel=1e-6)
