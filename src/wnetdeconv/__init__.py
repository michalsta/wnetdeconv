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


def hello():
    print("Hello, world from wnetdeconv!")
