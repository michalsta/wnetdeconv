#! /usr/bin/env python
# -*- coding: utf-8 -*-


from . import wnetdeconv_cpp
from .solver import DeconvSolver, ConstrainedSolver, MagnetsteinSolver, MassersteinSolver
from .spectrum import Spectrum, Spectrum_1D


def hello():
    print("Hello, world from wnetdeconv!")
