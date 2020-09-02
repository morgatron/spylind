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

    ode_s = so.ODESolver(dy_dtD,  dims={}) 
    ode_s.set_initial_conditions({xS:0,yS:0,vxS:10,vyS:10}) 
    ode_s.setup()
    tSteps = np.linspace(0,1,100)
    arr =ode_s.integrate(tSteps, 1.0)
    sim_res = dict(zip(["x", "y", "vx", "vy"], arr.T))

    tEff = tSteps + (tSteps[1]-tSteps[0])/2
    sym_res = dict(
        x= 10 *tEff,
        y = 10*tEff -5*tEff**2,
        vx = 10,
        vy = 10 + g*tEff
    )
    for name in ['x', 'y', 'vx', 'vy']:
        print('for sim: ', name)
        print(sym_res[name]-sim_res[name])
        assert(np.allclose(sym_res[name], sim_res[name]))

def test_one_plus_one_is_two():
    "Check that one and one are indeed two."
    assert 1 + 1 == 2
