from wnetdeconv import DeconvSolver, Spectrum_1D
from wnet.distances import DistanceMetric
import numpy as np


def test_basic():

    E = Spectrum_1D(
        [1, 2, 3],
        [10, 20, 30],
    )

    T1 = Spectrum_1D([1, 2], [2, 2])

    T2 = Spectrum_1D([2, 3], [1, 3])

    solver = DeconvSolver(
        empirical_spectrum=E,
        theoretical_spectra=[T1, T2],
        distance=DistanceMetric.LINF,
        max_distance=10,
        trash_cost=100,
    )

    solver.set_point([1.0, 1.0])
    print(solver.gradient())


if __name__ == "__main__":
    test_basic()
