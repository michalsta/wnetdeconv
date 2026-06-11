from typing import Optional
from functools import cached_property

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
