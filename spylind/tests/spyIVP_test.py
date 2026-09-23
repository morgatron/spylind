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
        assert np.allclose(sim_res[name], sym_res[name], atol=1e-5)

def test_late_binding_parameters():
    """Test updating tunable parameters without re-compiling."""
    x, v = sm.symbols("x, v", real=True)
    k = sm.symbols("k", real=True) # Parameter
    m = 1.0 # Constant
    
    dy_dtD = {x: v, v: -k/m * x}
    
    ode_sys = so.ODESys(dy_dtD, parameters=[k])
    ode_sys.set_initial_state({x: 1.0, v: 0.0})
    
    t_steps = np.linspace(0, 5.0, 50)
    
    for backend in ['numpy', 'diffrax']:
        if backend == 'diffrax' and so.ModelDiffrax is None:
            continue
            
        model = ode_sys.setup_model(backend=backend)
        
        # Test 1: k = 1.0 -> x(t) = cos(t)
        res1 = model.integrate(t_steps, params={k: 1.0})
        expected1 = np.cos(t_steps)
        assert np.allclose(res1[:, 0], expected1, atol=1e-4)
        
        # Test 2: k = 4.0 -> x(t) = cos(2t)
        # Should be fast as it uses late-binding
        res2 = model.integrate(t_steps, params={k: 4.0})
        expected2 = np.cos(2.0 * t_steps)
        assert np.allclose(res2[:, 0], expected2, atol=1e-4)

