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
        Sorts positions and intensities using np.lexsort with the positions as keys.
        """

        order = np.lexsort(tuple(self.positions[i, :] for i in range(self.positions.shape[0]-1, -1, -1)))
        return Spectrum(
            positions = self.positions[:, order],
            intensities = self.intensities[order],
            label = self.label,
        )

    def merge_signals(self):
        """
        Merges signals with identical positions, summing their intensities.
        """
        if len(self.positions) > 0:
            cpos = self.positions[:, 0]
            csig = 0.0
            merged_pos = []
            merged_sig = []
            for pos, sig in zip(self.positions.T, self.intensities):
                if not np.all(pos == cpos):
                    merged_pos.append(cpos)
                    merged_sig.append(csig)
                    cpos = pos
                    csig = 0.0
                csig += sig
            merged_pos.append(cpos)
            merged_sig.append(csig)

            positions = np.array(merged_pos).T
            intensities = np.array(merged_sig)

            return Spectrum(
                positions = positions,
                intensities = intensities,
                label = self.label,
            )
        
    def sort_positions_and_intensities(self):

        """
        Sorts positions and intensities using np.lexsort with the positions as keys. Returns sorted positions and intensities
        """

        order = np.lexsort(tuple(self.positions[i, :] for i in range(self.positions.shape[0]-1, -1, -1)))
        sorted_positions = self.positions[:, order]
        sorted_intensities = self.positions[:, order]
        return sorted_positions, sorted_intensities

    
    def merge_positions_and_intensities(self):
        pass


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
    
    def __add__(self, other):
        
        res = Spectrum(
            positions = np.hstack((self.positions, other.positions)),
            intensities = np.hstack((self.intensities, other.intensities)),
            label = self.label + ' + ' + other.label,
        )
        # res.sort_signals()
        # res.merge_signals()
        res = res.sort_signals()
        res = res.merge_signals()
        return res

    def __mul__(self, number):
        res = Spectrum(
            positions = self.positions,
            intensities = number * self.intensities,
            label = self.label,
        )
        return res        


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
