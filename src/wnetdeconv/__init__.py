#! /usr/bin/env python
# -*- coding: utf-8 -*-


from . import wnetdeconv_cpp
from .solver import (
    DeconvSolver,
    ConstrainedSolver,
    MagnetsteinSolver,
    MassersteinSolver,    # backwards-compat shim
    MassersteinSolver2,   # mimics dualdeconv2 (one-sided trash)
    MassersteinSolver4,   # mimics dualdeconv4 (two-sided trash)
)
from .spectrum import Spectrum, Spectrum_1D


def is_nanobind_split() -> bool:
    """True when wnetdeconv_cpp was built in nanobind split mode. See pylmcf.nanobind_mode."""
    from pylmcf.nanobind_mode import extension_is_split

    return extension_is_split(wnetdeconv_cpp)


def _check_nanobind_modes() -> None:
    # wnetdeconv drives wnet's networks from Python and hands wnet objects
    # around, so it must share wnet's nanobind runtime even though its own
    # extension registers no classes of its own.
    import pylmcf.pylmcf_cpp
    import wnet.wnet_cpp
    from pylmcf.nanobind_mode import check_consistent

    check_consistent(
        [
            ("pylmcf", pylmcf.pylmcf_cpp),
            ("wnet", wnet.wnet_cpp),
            ("wnetdeconv", wnetdeconv_cpp),
        ]
    )


_check_nanobind_modes()


def hello():
    print("Hello, world from wnetdeconv!")
