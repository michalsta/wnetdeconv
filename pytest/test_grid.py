import numpy as np
import pytest

from wnet.distances import DistanceMetric
from wnetdeconv import MagnetsteinSolver
from wnetdeconv.grid import ascent_labels, grid_to_spectrum, noise_sigma, remove_baseline


def gauss2d(axes, centre, width, height):
    g0, g1 = np.meshgrid(axes[0], axes[1], indexing="ij")
    return height * np.exp(-0.5 * (((g0 - centre[0]) / width[0]) ** 2 + ((g1 - centre[1]) / width[1]) ** 2))


AXES = [np.linspace(10, 60, 200), np.linspace(0.5, 5.5, 300)]


def component(peaks):
    return sum(gauss2d(AXES, c, (0.4, 0.02), h) for c, h in peaks)


COMP_A = component([((20, 1.2), 1.0), ((35, 3.4), 0.6)])
COMP_B = component([((30, 2.1), 0.8), ((50, 4.6), 1.0), ((22, 1.25), 0.3)])


def test_ascent_labels_two_peaks():
    img = gauss2d(AXES, (20, 1.2), (0.4, 0.02), 1.0) + gauss2d(AXES, (40, 4.0), (0.4, 0.02), 1.0)
    img[img < 1e-3] = 0
    lab, n = ascent_labels(img)
    assert n == 2
    assert set(np.unique(lab)) == {0, 1, 2}


def test_noise_sigma_matches_gaussian_noise():
    rng = np.random.default_rng(1)
    assert noise_sigma(rng.normal(0, 2.0, (300, 300))) == pytest.approx(2.0, rel=0.05)


def test_remove_baseline_rowcol_removes_ridge():
    img = np.zeros((50, 60))
    img[:, 7] += 3.0  # t1 ridge
    assert np.abs(remove_baseline(img, "rowcol")).max() == pytest.approx(0.0)


@pytest.mark.parametrize("peaks", [True, False])
def test_grid_to_spectrum_recovers_proportions(peaks):
    rng = np.random.default_rng(0)
    noise = 0.002
    mix = 0.3 * COMP_A + 0.7 * COMP_B + rng.normal(0, noise, COMP_A.shape)
    kw = dict(axis_scale=(20, 1), peaks=peaks, noise_threshold=5, binning=(1, 1) if peaks else (2, 2))
    emp = grid_to_spectrum(mix, AXES, **kw)
    refs = [grid_to_spectrum(c + rng.normal(0, noise, c.shape), AXES, **kw) for c in (COMP_A, COMP_B)]
    # proportions of unit-normalised references, mapped back to the mixing weights
    w = np.asarray(MagnetsteinSolver(emp, refs, DistanceMetric.L1, MTD=0.2, MTD_th=0.2).optimize().x)
    totals = np.array([COMP_A.sum(), COMP_B.sum()])
    est = w / totals
    est /= est.sum()
    assert est == pytest.approx([0.3, 0.7], abs=0.03)


def test_grid_to_spectrum_peaks_conserve_mass():
    img = COMP_A.copy()
    s = grid_to_spectrum(img, AXES, baseline=None, smooth=0, noise_threshold=None, relative_threshold=1e-3)
    kept = img[img > 1e-3 * img.max()].sum()
    assert s.sum_intensities == pytest.approx(kept)
    assert s.positions.shape[1] == 2  # two peaks


def test_exclude_box_removes_peak():
    img = COMP_A.copy()
    s = grid_to_spectrum(img, AXES, baseline=None, noise_threshold=None, relative_threshold=1e-3,
                         exclude=[((30, 40), (3.0, 3.8))])
    assert s.positions.shape[1] == 1


def test_axis_scale_divides_coordinates():
    s = grid_to_spectrum(COMP_A, AXES, baseline=None, noise_threshold=None, relative_threshold=1e-3,
                         axis_scale=(20, 1))
    assert s.positions[0].max() < 60 / 20 + 1e-9


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        grid_to_spectrum(np.zeros((3, 4)), [np.arange(4), np.arange(3)])


def test_smoothing_suppresses_isolated_noise_peaks():
    rng = np.random.default_rng(5)
    img = COMP_A + rng.normal(0, 0.01, COMP_A.shape)
    raw = grid_to_spectrum(img, AXES, smooth=0, noise_threshold=3)
    smoothed = grid_to_spectrum(img, AXES, smooth=0.7, noise_threshold=3)
    assert smoothed.positions.shape[1] < raw.positions.shape[1]
