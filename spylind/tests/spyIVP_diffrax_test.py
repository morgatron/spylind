import numpy as np
import sympy as sm
from spylind import spyIVP as so
import pytest

@pytest.mark.skipif(so.ModelDiffrax is None, reason="Diffrax not installed")
def test_projectile_motion_ensemble_diffrax():
    """Test projectile motion ensemble with different initial velocities using Diffrax."""
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
    
    # Setup and solve using Diffrax
    model = ode_sys.setup_model(backend='diffrax')
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

try:
    import jax.numpy as jnp
except ImportError:
    jnp = np

@pytest.mark.skipif(so.ModelDiffrax is None, reason="Diffrax not installed")
def test_projectile_motion_wind_ensemble_diffrax():
    """Test projectile motion ensemble with time-dependent wind force using Diffrax."""
    xS, yS, vxS, vyS = sm.symbols("x, y, v_x, v_y", real=True)
    v0S = sm.symbols("v_0", real=True)
    windS = sm.symbols("F_wind", real=True) # Driving symbol
    g = -9.81
    
    # Equations of motion: dvx/dt = F_wind(t)
    dy_dtD = {
        xS: vxS,
        yS: vyS,
        vxS: windS,
        vyS: g
    }
    
    v0_vals = np.array([10.0, 20.0])
    trans_dims = {v0S: v0_vals}
    
    ode_sys = so.ODESys(dy_dtD, trans_dims=trans_dims, driving_syms=[windS])
    
    # Wind force: F_wind(t) = sin(t)
    # Use jnp for compatibility with Diffrax backend
    ode_sys.set_driving({windS: lambda t: jnp.sin(t)})
    
    theta = np.pi / 4
    vx0 = v0_vals * np.cos(theta)
    vy0 = v0_vals * np.sin(theta)
    
    ode_sys.set_initial_state({
        xS: 0,
        yS: 0,
        vxS: vx0,
        vyS: vy0
    })
    
    model = ode_sys.setup_model(backend='diffrax')
    t_steps = np.linspace(0, 2.0, 50)
    res = model.integrate(t_steps)
    
    # Analytical solution for x:
    # dvx/dt = sin(t) => vx(t) = vx0 + (1 - cos(t))
    # dx/dt = vx0 + 1 - cos(t) => x(t) = (vx0 + 1)*t - sin(t)
    
    for i, v0 in enumerate(v0_vals):
        vx0_i = v0 * np.cos(theta)
        vy0_i = v0 * np.sin(theta)
        
        x_expected = (vx0_i + 1) * t_steps - np.sin(t_steps)
        y_expected = vy0_i * t_steps + 0.5 * g * t_steps**2
        
        assert np.allclose(res[:, 0, i], x_expected, atol=1e-4)
        assert np.allclose(res[:, 1, i], y_expected, atol=1e-4)

@pytest.mark.skipif(so.ModelDiffrax is None, reason="Diffrax not installed")
def test_harmonic_oscillator_diffrax():
    """Simple harmonic oscillator: d2x/dt2 = -k/m * x."""
    x, v = sm.symbols("x, v", real=True)
    k, m = 1.0, 1.0
    
    dy_dtD = {
        x: v,
        v: -(k/m) * x
    }
    
    ode_sys = so.ODESys(dy_dtD)
    ode_sys.set_initial_state({x: 1.0, v: 0.0})
    
    model = ode_sys.setup_model(backend='diffrax')
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
