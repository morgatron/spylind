import CyRK
import numpy as np
import numba
from numba import objmode

class MyRHS:
    def __call__(self, t, y):
        return -y

rhs_obj = MyRHS()

@numba.njit
def rhs_nb(t, y):
    with objmode(res='float64[:]'):
        res = rhs_obj(t, y)
    return res

y0 = np.array([1.0])
print("Testing nbsolve_ivp with objmode...")
try:
    # Use nbsolve_ivp which expects an njitted function
    res = CyRK.nbsolve_ivp(rhs_nb, (0.0, 1.0), y0)
    print("Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Failed: {e}")
