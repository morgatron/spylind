import numpy as np
from spylind import spyIVP as so
import sympy as sm

def test_projectile_motion():
    xS, yS, vxS, vyS = sm.symbols("x, y, v_x, v_y", real=True)
    g= -10.0
    dy_dtD = {xS:vxS,
        yS:vyS,
        vxS:0,
        vyS:g }

    ode_s = so.ODESys(dy_dtD,  trans_dims={})
    ode_s.set_initial_state({xS:0,yS:0,vxS:10,vyS:10})
    model = ode_s.setup_model()
    tSteps = np.linspace(0,1,100)
    arr = model.integrate(tSteps)
    sim_res = dict(zip(["x", "y", "vx", "vy"], arr.T))

    sym_res = dict(
        x= 10 *tSteps,
        y = 10*tSteps -5*tSteps**2,
        vx = 10 * np.ones_like(tSteps),
        vy = 10 + g*tSteps
    )
    for name in ['x', 'y', 'vx', 'vy']:
        print('for sim: ', name)
        print(sym_res[name]-sim_res[name])
        assert(np.allclose(sym_res[name], sim_res[name]))

