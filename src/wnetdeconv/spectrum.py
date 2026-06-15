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

    def __init__(
        self,
        positions: np.ndarray,
        intensities: np.ndarray,
        label: Optional[str] = None,
    ):
        """
        Initialize a Spectrum object.

        Parameters
        ----------
        positions : np.ndarray
            The spatial coordinates of the spectrum (e.g., m/z and RT for MS).
        intensities : np.ndarray
            The intensity values corresponding to the spatial coordinates.
        """
        self.original_intensities = intensities
        super().__init__(positions, intensities, label=label)

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

    @cached_property
    def sum_intensities(self) -> float:
        """
        Return the sum of the original intensities.
        """
        return np.sum(self.original_intensities)

    def scaled(self, factor: float) -> "Spectrum":
        """
        Return a new Spectrum object with intensities scaled by the given factor.

        Parameters
        ----------
        factor : float
            The scaling factor to apply to the intensities.

        Returns
        -------
        Spectrum
            A new Spectrum object with scaled intensities.
        """
        return Spectrum(
            self.positions, self.original_intensities * factor, label=self.label
        )

    def positions_intensities_scaled(self, scale_factor: float) -> "Spectrum":
        """
        Return a new Spectrum with both positions and intensities scaled by the given factor.

        Parameters
        ----------
        scale_factor : float
            The scaling factor to apply to positions and intensities.

        Returns
        -------
        Spectrum
            A new Spectrum object with scaled positions and intensities.
        """
        new_positions = self.positions.astype(np.float64, copy=False) * scale_factor
        return Spectrum(new_positions, self.original_intensities * scale_factor, label=self.label)

    def normalized(self) -> "Spectrum":
        """
        Return a new Spectrum object with intensities normalized to sum to 1.

        Returns
        -------
        Spectrum
            A new Spectrum object with normalized intensities.
        """
        total = self.sum_intensities
        if total == 0:
            raise ValueError("Cannot normalize a spectrum with total intensity of 0.")
        return Spectrum(
            self.positions, self.original_intensities / total, label=self.label
        )

    def as_distribution(self) -> Distribution:
        """
        Convert the Spectrum object to a Distribution object.

        Returns
        -------
        Distribution
            A Distribution object with the same positions and intensities.
        """
        return Distribution(self.positions, self.intensities, label=self.label)
    
    def normalize_scaled(self) -> "Spectrum":
        """
        Return a new Spectrum object with intensities normalized to sum to 1. 
        Uses self.intensities not self.original_intensities for normalization in contrast to normalized method.

        Returns
        -------
        Spectrum
            A new Spectrum object with normalized intensities.
        """
        total = np.sum(self.intensities)
        if total == 0:
            raise ValueError("Cannot normalize a spectrum with total intensity of 0.")
        return Spectrum(
            self.positions, self.intensities / total, label=self.label
        )

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
        self.positions = self.positions[:, order]
        self.intensities = self.intensities[order]
        self.original_intensities = self.original_intensities[order]

    def merge_signals(self):
        """
        Merges signals with identical positions, summing their intensities.
        """
        if len(self.positions) > 0:
            cpos = self.positions[:, 0]
            csig = 0.0
            og_csig = 0.0
            merged_pos = []
            merged_sig = []
            merged_og_sig = []
            for pos, sig, og_sig in zip(self.positions.T, self.intensities, self.original_intensities):
                if not np.all(pos == cpos):
                    merged_pos.append(cpos)
                    merged_sig.append(csig)
                    merged_og_sig.append(og_csig)
                    cpos = pos
                    csig = 0.0
                    og_csig = 0.0
                csig += sig
                og_csig += og_sig
            merged_pos.append(cpos)
            merged_sig.append(csig)
            merged_og_sig.append(og_csig)

            self.positions = np.array(merged_pos).T
            self.intensities = np.array(merged_sig)
            self.original_intensities = np.array(merged_og_sig)

    def set_signals(self, positions, intensities):
        if len(positions) == 0 or len(intensities) == 0:
            raise ValueError(
                "Empty signal positions or intensities"
            )
        if positions.shape[1] != intensities.shape[0]:
            raise ValueError(
                "Number of signal positions and intensities do not match."
            )
        self.positions = positions
        self.intensities = intensities
        self.original_intensities = intensities
        self.sort_confs()
        self.merge_confs()
    
    def __add__(self, other):
        
        res = Spectrum(
            positions = np.hstack((self.positions, other.positions)),
            intensities = np.hstack((self.intensities, other.intensities)),
            label = self.label + ' + ' + other.label,
        )
        res.sort_signals()
        res.merge_signals()
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
