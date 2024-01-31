#!python
#cython: language_level=3
# This file is generated automatically by QuTiP.

import numpy as np
import scipy.special as spe
cimport numpy as np
cimport cython
from qutip.cy.math cimport erf, zerf
cdef double pi = 3.14159265358979323
include '/home/morgan/anaconda3/envs/spylind/lib/python3.8/site-packages/qutip/cy/complex_math.pxi'

@cython.boundscheck(False)
@cython.wraparound(False)
def f(double t, args):
    return 1
