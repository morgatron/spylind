import numpy as np
import sympy as sm
from spylind import spyIVP as so
import pytest

@pytest.mark.skipif(so.ModelCyRK is None, reason="CyRK not installed")
def test_projectile_motion_ensemble_cyrk():
    """Test projectile motion ensemble with different initial velocities using CyRK."""
    # Define symbols
    xS, yS, vxS, vyS = sm.symbols("x, y, v_x, v_y", real=True)
    v0S = sm.symbols("v_0", real=True) # Ensemble parameter
    g = -9.81
    
    # Equations of motion
    dy_dtD = {
        xS: vxS,
        yS: vyS,
        vxS: 0,
        vyS: g
    }
    
    # Define the ensemble dimension: initial velocities from 5 to 15 m/s
    v0_vals = np.linspace(5, 15, 5)
    trans_dims = {v0S: v0_vals}
    
    # Create the ODE system
    ode_sys = so.ODESys(dy_dtD, trans_dims=trans_dims)
    
    # Set initial conditions: all start at (0,0), launch angle 45 degrees
    # vx = v0 * cos(45), vy = v0 * sin(45)
    theta = np.pi / 4
    vx0 = v0_vals * np.cos(theta)
    vy0 = v0_vals * np.sin(theta)
    
    ode_sys.set_initial_state({
        xS: 0,
        yS: 0,
        vxS: vx0,
        vyS: vy0
    })
    
    # Setup and solve using CyRK
    model = ode_sys.setup_model(backend='cyrk')
    t_steps = np.linspace(0, 1.0, 50)
    # res shape: (n_t, n_vars, n_v0) = (50, 4, 5)
    res = model.integrate(t_steps)
    
    # Verify results for each v0 in the ensemble
    for i, v0 in enumerate(v0_vals):
        vx_expected = v0 * np.cos(theta)
        vy_expected = v0 * np.sin(theta) + g * t_steps
        x_expected = v0 * np.cos(theta) * t_steps
        y_expected = v0 * np.sin(theta) * t_steps + 0.5 * g * t_steps**2
        
        # res[:, 0, i] is x, [:, 1, i] is y, [:, 2, i] is vx, [:, 3, i] is vy
        assert np.allclose(res[:, 0, i], x_expected, atol=1e-4)
        assert np.allclose(res[:, 1, i], y_expected, atol=1e-4)
        assert np.allclose(res[:, 2, i], vx_expected, atol=1e-4)
        assert np.allclose(res[:, 3, i], vy_expected, atol=1e-4)

@pytest.mark.skipif(so.ModelCyRK is None, reason="CyRK not installed")
def test_harmonic_oscillator_cyrk():
    """Simple harmonic oscillator: d2x/dt2 = -k/m * x."""
    x, v = sm.symbols("x, v", real=True)
    k, m = 1.0, 1.0
    
    dy_dtD = {
        x: v,
        v: -(k/m) * x
    }
    
    ode_sys = so.ODESys(dy_dtD)
    ode_sys.set_initial_state({x: 1.0, v: 0.0})
    
    model = ode_sys.setup_model(backend='cyrk')
    t_steps = np.linspace(0, 10, 100)
    res = model.integrate(t_steps)
    
    # Analytical solution: x(t) = cos(t)
    expected_x = np.cos(t_steps)
    # res shape: (n_t, n_vars, *dims)
    # For no dims, it's (n_t, n_vars)
    if res.ndim == 3:
        assert np.allclose(res[:, 0, 0], expected_x, atol=1e-4)
    else:
        assert np.allclose(res[:, 0], expected_x, atol=1e-4)

@pytest.mark.skipif(so.ModelCyRK is None, reason="CyRK not installed")
def test_complex_oscillator_decomposed_cyrk():
    """Test a complex oscillator with decomposition using CyRK."""
    y = sm.symbols("y")
    omega = 1.0
    
    dy_dtD = {y: 1j * omega * y}
    
    # CyRK usually only supports real-valued ODEs, so decomposition is required
    ode_sys = so.ODESys(dy_dtD, bDecompose_to_re_im=True)
    ode_sys.set_initial_state({y: 1.0 + 0j})
    
    model = ode_sys.setup_model(backend='cyrk')
    t_steps = np.linspace(0, 2 * np.pi, 100)
    res = model.integrate(t_steps)
    
    expected = np.exp(1j * omega * t_steps)
    assert np.allclose(res[:, 0], expected, atol=1e-5)


@pytest.mark.skipif(so.ModelCyRK is None, reason="CyRK not installed")
def test_cyrk_fast_path_vs_fallback():
    """Verify that fast_path produces identical results to fallback pysolve_ivp."""
    x, v = sm.symbols("x, v", real=True)
    gamma = sm.symbols("gamma", real=True)
    dy_dtD = {
        x: v,
        v: -x - gamma * v
    }
    ode_sys = so.ODESys(dy_dtD, parameters={gamma: 0.2})
    ode_sys.set_initial_state({x: 1.0, v: 0.0})

    model_fast = ode_sys.setup_model(backend='cyrk', solver_opts={'fast_path': True})
    model_slow = ode_sys.setup_model(backend='cyrk', solver_opts={'fast_path': False})

    assert model_fast.fast_callback_info is not None
    assert model_slow.fast_callback_info is None

    t_steps = np.linspace(0, 10, 80)
    res_fast = model_fast.integrate(t_steps)
    res_slow = model_slow.integrate(t_steps)

    assert np.allclose(res_fast, res_slow, atol=1e-6)


@pytest.mark.skipif(so.ModelCyRK is None, reason="CyRK not installed")
def test_cyrk_parameter_update():
    """Verify that updating parameters works in the fast path args buffer."""
    x, v = sm.symbols("x, v", real=True)
    gamma = sm.symbols("gamma", real=True)
    dy_dtD = {
        x: v,
        v: -x - gamma * v
    }
    ode_sys = so.ODESys(dy_dtD, parameters={gamma: 0.0})
    ode_sys.set_initial_state({x: 1.0, v: 0.0})

    model = ode_sys.setup_model(backend='cyrk')
    assert model.fast_callback_info is not None

    t_steps = np.linspace(0, 10, 100)
    # Undamped run (gamma=0)
    res_undamped = model.integrate(t_steps)
    assert np.isclose(res_undamped[-1, 0]**2 + res_undamped[-1, 1]**2, 1.0, atol=1e-3)

    # Damped run (gamma=2.0) via params dict
    res_damped = model.integrate(t_steps, params={gamma: 2.0})
    # Energy should decay significantly
    energy_damped = res_damped[-1, 0]**2 + res_damped[-1, 1]**2
    assert energy_damped < 0.01

    # Also verify slow model with same params produces identical result
    model_slow = ode_sys.setup_model(backend='cyrk', solver_opts={'fast_path': False})
    res_slow_damped = model_slow.integrate(t_steps, params={gamma: 2.0})
    assert np.allclose(res_damped, res_slow_damped, atol=1e-5)


@pytest.mark.skipif(so.ModelCyRK is None, reason="CyRK not installed")
def test_cyrk_unjittable_driving_fallback():
    """Verify graceful fallback when driving function contains un-jittable Python code."""
    x = sm.symbols("x", real=True)
    f_sym = sm.symbols("f_sym", real=True)

    class CustomState:
        def __init__(self):
            self.count = 0
        def step(self):
            self.count += 1
            return 1.0

    state_obj = CustomState()

    def unjittable_driving(t):
        # Using custom object method cannot be JIT-compiled by Numba
        return state_obj.step()

    dy_dtD = {x: f_sym}
    ode_sys = so.ODESys(dy_dtD, driving_syms=[f_sym])
    ode_sys.set_driving({f_sym: unjittable_driving})
    ode_sys.set_initial_state({x: 0.0})

    model = ode_sys.setup_model(backend='cyrk')
    # Should fall back cleanly without raising
    assert model.fast_callback_info is None

    t_steps = np.linspace(0, 2, 20)
    res = model.integrate(t_steps)
    # dx/dt = 1 -> x(t) = t
    assert np.allclose(res[:, 0], t_steps, atol=1e-3)

