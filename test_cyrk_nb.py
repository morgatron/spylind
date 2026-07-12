import CyRK
import numpy as np
import numba

def rhs(t, y):
    return -y

# Wrap in njit with object mode fallback
@numba.njit
def rhs_nb(t, y):
    # This is a dummy, but let's see if nbsolve_ivp likes it
    return -y

y0 = np.array([1.0])
print("Testing nbsolve_ivp...")
try:
    res = CyRK.nbsolve_ivp(rhs_nb, (0.0, 1.0), y0)
    print("nbsolve_ivp success!")
except Exception as e:
    print(f"nbsolve_ivp failed: {e}")
