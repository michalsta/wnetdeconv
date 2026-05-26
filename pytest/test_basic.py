from wnetdeconv import DeconvSolver, Spectrum_1D
from wnet.distances import DistanceMetric
import numpy as np


def test_basic():
    # E sum=60, T1+T2 sum=4+4=8 at fractions [1,1].
    # All T positions (1,2,3) coincide with E positions → matching distance 0.
    # 52 units of E trashed at 100 each → cost 5200.
    # Gradient: adding ε more of T_i saves ε*4 trash units → dCost/df_i = -400.
    E = Spectrum_1D([1, 2, 3], [10, 20, 30])
    T1 = Spectrum_1D([1, 2], [2, 2])
    T2 = Spectrum_1D([2, 3], [1, 3])

    solver = DeconvSolver(
        empirical_spectrum=E,
        theoretical_spectra=[T1, T2],
        distance=DistanceMetric.LINF,
        max_distance=10,
        trash_cost=100,
        scale_factor=1e7,
    )

    solver.set_point([1.0, 1.0])
    cost = solver.total_cost()
    grad = solver.gradient()

    assert np.isclose(cost, 5200, rtol=1e-4), f"Expected cost ~5200, got {cost}"
    assert grad.shape == (2,)
    assert np.allclose(grad, [-400.0, -400.0], rtol=1e-4), f"Expected gradient ~[-400,-400], got {grad}"

    # Finite-difference verification
    eps = 1e-4
    for i in range(2):
        point_plus = [1.0 + (eps if j == i else 0.0) for j in range(2)]
        solver.set_point(point_plus)
        fd = (solver.total_cost() - cost) / eps
        assert np.isclose(fd, grad[i], rtol=1e-3), (
            f"FD gradient[{i}]: expected {grad[i]:.2f}, got {fd:.2f}"
        )


if __name__ == "__main__":
    test_basic()
