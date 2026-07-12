import CyRK
import numpy as np
import sympy as sm
from spylind import spyIVP as so

# Define a simple system
x, v = sm.symbols("x, v", real=True)
dy_dtD = {x: v, v: -x}
ode_sys = so.ODESys(dy_dtD)
ode_sys.set_initial_state({x: 1.0, v: 0.0})
model = ode_sys.setup_model(backend='cyrk')

# Current wrapped RHS
def rhs_wrapped(t, y_flat):
    y = model._unflatten(y_flat)
    dy = model.d_dt_fast(t, y)
    return dy.reshape(-1)

print("Attempting cyjit...")
try:
    # CyRK.cyjit is used to compile the RHS
    rhs_jitted = CyRK.cyjit(rhs_wrapped)
    print("cyjit success!")
    
    # Test call
    y0 = np.array([1.0, 0.0])
    res = rhs_jitted(0.0, y0)
    print(f"Result: {res}")
except Exception as e:
    print(f"cyjit failed: {e}")

