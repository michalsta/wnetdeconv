from typing import Optional
from functools import cached_property
from copy import deepcopy

import numpy as np

from wnet import Distribution


class Spectrum(Distribution):
    """
    A class representing NMR or MS spectrum data.

    A thin subclass of Distribution: positions and intensities are owned by the
    C++ distribution object (real float64), and all scaling/normalization
    helpers are inherited (Distribution's polymorphic constructor returns a
    Spectrum here).  The only addition is the MS-specific ``FromFeatureXML``.
    """

    @staticmethod
    def FromFeatureXML(path):
        """
        Parse a featureXML file and return a Spectrum object.
        """
        import pyopenms as oms

        # load the featureXML file
        featureXML = oms.FeatureXMLFile()
        features = oms.FeatureMap()
        featureXML.load(path, features)
        # load m/z, rt, and intensity values from the features
        mzs = []
        rts = []
        intensities = []
        for feature in features:
            mzs.append(feature.getMZ())
            rts.append(feature.getRT())
            intensities.append(feature.getIntensity())
        # create a Spectrum object
        spectrum = Spectrum(np.array([mzs, rts]), np.array(intensities))
        return spectrum

    def copy(self):
        """
        Return a (deep) copy of self
        """
        return deepcopy(self)
    
    def sort_signals(self):
        """
        Returns a copy with peaks sorted lexicographically by position
        (dimension 0 first, then 1, ...), intensities permuted along.

        Delegates to the C++-backed ``Distribution.sorted_by_positions``
        (which returns a ``Spectrum`` here, label preserved).  Peaks are not
        merged.
        """
        return self.sorted_by_positions()

    def merge_signals(self):
        """
        Returns a copy with peaks sharing an identical position merged
        (intensities summed) and the result sorted lexicographically.

        Performed in C++ (via the single-input linear combination, weight 1 —
        exact in double).  Unlike the historical Python implementation this
        does not require the peaks to be pre-sorted.
        """
        # positions is (dim, n_peaks): emptiness is shape[1], not len()
        # (which is the dimension count and never 0).
        if self.positions.shape[1] == 0:
            return self.copy()
        cpp = type(self.vecdist).linear_combination(
            [self.vecdist], np.ones(1, dtype=np.float64)
        )
        return type(self)(
            cpp.py_get_positions(), cpp.py_get_intensities(), label=self.label
        )

    def sort_positions_and_intensities(self):

        """
        Sorts positions and intensities using np.lexsort with the positions as keys. Returns sorted positions and intensities
        """

        order = np.lexsort(tuple(self.positions[i, :] for i in range(self.positions.shape[0]-1, -1, -1)))
        sorted_positions = self.positions[:, order]
        sorted_intensities = self.intensities[order]
        return sorted_positions, sorted_intensities


    # def set_signals(self, positions, intensities):
    #     if len(positions) == 0 or len(intensities) == 0:
    #         raise ValueError(
    #             "Empty signal positions or intensities"
    #         )
    #     if positions.shape[1] != intensities.shape[0]:
    #         raise ValueError(
    #             "Number of signal positions and intensities do not match."
    #         )
    #     self.positions = positions
    #     self.intensities = intensities
    #     self.sort_confs()
    #     self.merge_confs()
    
    # __add__ is inherited from Distribution: the C++-backed merge
    # (concatenate, merge identical positions, sort lexicographically) with
    # None-safe label combination, returning a Spectrum (polymorphic
    # constructor).  Identical semantics to the historical
    # hstack + sort_signals + merge_signals chain, without the per-peak
    # Python loop.

    def __mul__(self, number):
        res = Spectrum(
            positions = self.positions,
            intensities = number * self.intensities,
            label = self.label,
        )
        return res

    def smoothed(self, sigma: float) -> "Spectrum":
        """Return a Gaussian-smoothed copy (1-D profile spectra only).

        Convolves the intensities with a Gaussian kernel of standard
        deviation ``sigma`` (in position units), sampled on the spectrum's
        own grid with the median grid step.  Total intensity is preserved up
        to edge truncation.  Intended for evenly gridded profile data; on a
        strongly non-uniform grid (e.g. centroided MS) the kernel width in
        points no longer tracks ``sigma`` and the result is not meaningful.

        Parameters
        ----------
        sigma : float
            Gaussian standard deviation in position units.  ``sigma <= 0``
            returns an unmodified copy.
        """
        if self.dimension != 1:
            raise ValueError("smoothed() supports 1-D spectra only")
        if sigma <= 0:
            return self.copy()
        pos = np.asarray(self.positions[0], dtype=float)
        order = np.argsort(pos)
        pos_sorted = pos[order]
        ints_sorted = np.asarray(self.intensities, dtype=float)[order]
        steps = np.diff(pos_sorted)
        if steps.size == 0:
            return self.copy()
        step = float(np.median(steps))
        if not step > 0:
            raise ValueError("smoothed() requires distinct positions")
        half = int(np.ceil(4.0 * sigma / step))
        kernel = np.exp(
            -0.5 * ((np.arange(-half, half + 1) * step) / sigma) ** 2
        )
        kernel /= kernel.sum()
        smoothed = np.convolve(ints_sorted, kernel, mode="same")
        return Spectrum(
            positions=pos_sorted[np.newaxis, :],
            intensities=smoothed,
            label=self.label,
        )


def Spectrum_1D(
    positions: np.ndarray, intensities: np.ndarray, label: Optional[str] = None
) -> Spectrum:
    """
    Create a 1D Spectrum object.

    Parameters
    ----------
    positions : np.ndarray
        The spatial coordinates of the spectrum (e.g., m/z for MS).
    intensities : np.ndarray
        The intensity values corresponding to the spatial coordinates.
    label : str, optional
        An optional label for the spectrum.

    Returns
    -------
    Spectrum
        A 1D Spectrum object.
    """
    if not isinstance(positions, np.ndarray):
        positions = np.array(positions)
    if not isinstance(intensities, np.ndarray):
        intensities = np.array(intensities)
    if positions.ndim != 1:
        raise ValueError(f"positions must be 1D, got shape {positions.shape}")
    if intensities.ndim != 1:
        raise ValueError(f"intensities must be 1D, got shape {intensities.shape}")
    if positions.shape[0] != intensities.shape[0]:
        raise ValueError(
            f"positions and intensities must have the same length, got {positions.shape[0]} and {intensities.shape[0]}"
        )
    return Spectrum(positions[np.newaxis, :], intensities, label=label)
