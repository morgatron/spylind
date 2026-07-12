import CyRK
import numpy as np
import numba
from numba import objmode

class MyRHS:
    def __call__(self, t, y):
        return -y

rhs_obj = MyRHS()

# For nbsolve2_ivp, the RHS must be njit and have a very specific signature
# It often needs to return the result by writing to an input array if it's the 'fast' path,
# or it might take (t, y) and return a new array.
# Looking at CyRK docs, nb_diffeq_addr usually expects (t, y) -> dy

@numba.njit
def rhs_nb(t, y):
    with objmode(res='float64[:]'):
        res = rhs_obj(t, y)
    return res

y0 = np.array([1.0])
print("Testing nbsolve2_ivp...")
try:
    # 1. Get address
    addr = CyRK.nb_diffeq_addr(rhs_nb)
    # 2. Solve
    res = CyRK.nbsolve2_ivp(addr, (0.0, 1.0), y0, method='RK45')
    print("Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Failed: {e}")
