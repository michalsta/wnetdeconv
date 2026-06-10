from typing import Optional

import numpy as np

from wnet import Distribution


class Spectrum(Distribution):
    """
    A class representing NMR or MS spectrum data.

    Compared to Distribution, this class retains the original (non-int)
    intensities in ``original_intensities`` for more precise scaling.  The
    scaling/normalization helpers (``scaled``, ``positions_intensities_scaled``,
    ``normalized``, ``as_distribution``, ``sum_intensities``) are inherited from
    Distribution, whose polymorphic constructor returns a Spectrum here.
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
