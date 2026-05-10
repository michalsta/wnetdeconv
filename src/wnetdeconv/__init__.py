#! /usr/bin/env python
# -*- coding: utf-8 -*-


from . import wnetdeconv_cpp
from .solver import DeconvSolver
from .spectrum import Spectrum, Spectrum_1D


def hello():
    print("Hello, world from wnetdeconv!")
