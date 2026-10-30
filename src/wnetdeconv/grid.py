"""Turn regularly sampled 2D spectra (HSQC, TOCSY, DOSY, ...) into Spectrum objects.

On processed 2D NMR grids, how the grid is reduced to a point set decides the
accuracy of the deconvolution far more than the solver settings do (trash
model, metric, p).  ``grid_to_spectrum`` packages the reduction that worked
best on the public 2D benchmarks with known composition:

1. optional exclusion of solvent / standard regions,
2. baseline removal (global median, or row/column medians against t1 ridges),
3. optional Gaussian smoothing, then a noise threshold in units of the
   robust (MAD) noise level of the (smoothed) image,
4. optional block binning,
5. optional peak segmentation: every positive pixel is assigned to the local
   maximum its steepest ascent reaches, and each peak becomes one point at its
   intensity centroid carrying the peak volume,
6. per-axis scaling, so one unit of distance means the same along every axis.
"""
from typing import Optional, Sequence, Tuple

import numpy as np

from .spectrum import Spectrum


def noise_sigma(image: np.ndarray) -> float:
    """Robust noise level of a baseline-corrected (unclipped) image: 1.4826 * MAD."""
    image = np.asarray(image, dtype=float)
    med = np.median(image)
    return float(1.4826 * np.median(np.abs(image - med)))


def remove_baseline(image: np.ndarray, mode: Optional[str] = "rowcol") -> np.ndarray:
    """Subtract a baseline estimate; the result is not clipped.

    ``"median"`` subtracts the global median; ``"rowcol"`` subtracts the median
    of every column and then of every row (t1 ridges, streaks); ``None`` or
    ``"none"`` leaves the image unchanged.
    """
    image = np.asarray(image, dtype=float)
    if mode in (None, "none"):
        return image.copy()
    if mode == "median":
        return image - np.median(image)
    if mode == "rowcol":
        out = image - np.median(image, axis=0, keepdims=True)
        return out - np.median(out, axis=1, keepdims=True)
    raise ValueError(f"unknown baseline mode {mode!r}")


def ascent_labels(image: np.ndarray) -> Tuple[np.ndarray, int]:
    """Label positive pixels by the local maximum reached by steepest ascent.

    Each pixel points to the largest value in its 3x3 neighbourhood (itself
    included, ties broken by flat index); pointer jumping follows the chains
    to their end.  Equivalent in practice to a marker watershed seeded at
    the local maxima, without the scikit-image dependency.

    Returns ``(labels, n)``: labels are 1..n on positive pixels, 0 elsewhere.
    """
    image = np.asarray(image, dtype=float)
    h, w = image.shape
    pad = np.pad(image, 1, constant_values=-np.inf)
    flat = np.arange(h * w).reshape(h, w)
    best_val = image.copy()
    best_idx = flat.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            v = pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            ys = np.clip(np.arange(h) + dy, 0, h - 1)[:, None]
            xs = np.clip(np.arange(w) + dx, 0, w - 1)[None, :]
            idx = flat[ys, xs]
            better = (v > best_val) | ((v == best_val) & (idx > best_idx) & np.isfinite(v))
            best_val = np.where(better, v, best_val)
            best_idx = np.where(better, idx, best_idx)
    ptr = best_idx.ravel()
    while True:
        nxt = ptr[ptr]
        if np.array_equal(nxt, ptr):
            break
        ptr = nxt
    positive = image.ravel() > 0
    roots, lab = np.unique(ptr[positive], return_inverse=True)
    labels = np.zeros(h * w, dtype=np.int64)
    labels[positive] = lab + 1
    return labels.reshape(h, w), len(roots)


def _bin(image, axes, factors):
    fy, fx = factors
    if fy == 1 and fx == 1:
        return image, axes
    ny, nx = image.shape[0] // fy * fy, image.shape[1] // fx * fx
    image = image[:ny, :nx].reshape(ny // fy, fy, nx // fx, fx).sum(axis=(1, 3))
    return image, [axes[0][:ny].reshape(-1, fy).mean(axis=1),
                   axes[1][:nx].reshape(-1, fx).mean(axis=1)]


def grid_to_spectrum(
    image: np.ndarray,
    axes: Sequence[np.ndarray],
    *,
    baseline: Optional[str] = "median",
    smooth: float = 0.7,
    noise_threshold: Optional[float] = 4.0,
    relative_threshold: Optional[float] = None,
    binning: Tuple[int, int] = (1, 1),
    peaks: bool = True,
    exclude: Sequence[Sequence[Tuple[float, float]]] = (),
    axis_scale: Sequence[float] = (1.0, 1.0),
    label: Optional[str] = None,
) -> Spectrum:
    """Reduce a processed 2D grid spectrum to a :class:`Spectrum`.

    Parameters
    ----------
    image : (n0, n1) array
        Processed real spectrum; ``image[i, j]`` is the intensity at
        ``(axes[0][i], axes[1][j])``.
    axes : two 1-D arrays
        Coordinates (e.g. ppm) of the rows and columns.
    baseline : {"rowcol", "median", None}
        Baseline removal, see :func:`remove_baseline`.
    smooth : float
        Gaussian smoothing width in pixels, applied before thresholding (0
        disables).  Smoothing removes isolated noise maxima, which lets the
        threshold sit lower and keeps weak peaks.
    noise_threshold : float or None
        Zero every pixel at or below ``noise_threshold`` times the MAD noise
        level of the baseline-corrected (smoothed) image.  With
        ``smooth=0.7``, 4 was the best setting on HSQC (5 without
        smoothing); lower values let noise peaks in, higher ones cut weak
        compounds.
    relative_threshold : float or None
        Additionally zero pixels at or below this fraction of the maximum.
    binning : (int, int)
        Block sums along each axis, applied after thresholding.
    peaks : bool
        Collapse each peak (steepest-ascent basin) to one point at its
        intensity centroid carrying its volume.  Use it when the references
        are peak lists, or to shrink the transport network; keep the profile
        (``False``) when references are full spectra on the same grid.
    exclude : sequence of boxes
        Each box is ``((lo0, hi0), (lo1, hi1))`` in axis coordinates; pixels
        inside are set to the image median before baseline removal (solvent,
        chemical-shift standard).
    axis_scale : (float, float)
        Coordinates are divided by these factors, so that distances along
        both axes are commensurable (e.g. ``(20, 1)`` for 13C x 1H HSQC in
        ppm: one 1H ppm costs as much as twenty 13C ppm).
    label : str, optional
        Label of the returned spectrum.
    """
    image = np.array(image, dtype=float)
    axes = [np.asarray(a, dtype=float) for a in axes]
    if image.ndim != 2 or len(axes) != 2 or image.shape != (len(axes[0]), len(axes[1])):
        raise ValueError("image must be 2-D with one coordinate array per axis")
    if exclude:
        g0, g1 = np.meshgrid(axes[0], axes[1], indexing="ij")
        bad = np.zeros(image.shape, dtype=bool)
        for (lo0, hi0), (lo1, hi1) in exclude:
            bad |= (g0 >= lo0) & (g0 <= hi0) & (g1 >= lo1) & (g1 <= hi1)
        image[bad] = np.median(image)
    image = remove_baseline(image, baseline)
    if smooth:
        from scipy import ndimage

        image = ndimage.gaussian_filter(image, smooth)
    t = 0.0
    if noise_threshold:
        t = max(t, noise_threshold * noise_sigma(image))
    if relative_threshold:
        t = max(t, relative_threshold * image.max())
    image[image <= t] = 0.0
    image, axes = _bin(image, axes, binning)
    if peaks:
        from scipy import ndimage

        lab, n = ascent_labels(image)
        idx = np.arange(1, n + 1)
        vol = ndimage.sum(image, lab, idx)
        g0, g1 = np.meshgrid(axes[0], axes[1], indexing="ij")
        safe = np.where(vol > 0, vol, 1.0)
        pos = np.vstack([ndimage.sum(image * g0, lab, idx) / safe,
                         ndimage.sum(image * g1, lab, idx) / safe])
        keep = vol > 0
        pos, inten = pos[:, keep], vol[keep]
    else:
        nz = np.nonzero(image > 0)
        pos = np.vstack([axes[0][nz[0]], axes[1][nz[1]]])
        inten = image[nz]
    if inten.size == 0:
        raise ValueError("no signal left after thresholding")
    pos = pos / np.asarray(axis_scale, dtype=float)[:, None]
    return Spectrum(np.ascontiguousarray(pos), np.ascontiguousarray(inten), label=label)
