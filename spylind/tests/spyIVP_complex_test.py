import numpy as np
import sympy as sm
from spylind import spyIVP as so
import pytest

def test_complex_oscillator_numpy():
    """Test a simple complex oscillator: dy/dt = i * omega * y, y(0) = 1."""
    y = sm.symbols("y") # complex by default
    omega = 1.0
    t = sm.symbols("t")
    
    dy_dtD = {y: 1j * omega * y}
    
    ode_sys = so.ODESys(dy_dtD)
    ode_sys.set_initial_state({y: 1.0 + 0j})
    
    model = ode_sys.setup_model(backend='numpy')
    t_steps = np.linspace(0, 2 * np.pi, 100)
    res = model.integrate(t_steps)
    
    # Expected solution: y(t) = e^{i * omega * t}
    expected = np.exp(1j * omega * t_steps)
    
    # res is (100, 1) or (1, 100) depending on implementation, 
    # but integrate usually returns (n_steps, n_vars)
    assert np.allclose(res[:, 0], expected, atol=1e-5)

def test_complex_oscillator_decomposed_numpy():
    """Test the same oscillator with real/imaginary decomposition."""
    y = sm.symbols("y")
    omega = 1.0
    
    dy_dtD = {y: 1j * omega * y}
    
    # Enable decomposition
    ode_sys = so.ODESys(dy_dtD, bDecompose_to_re_im=True)
    ode_sys.set_initial_state({y: 1.0 + 0j})
    
    model = ode_sys.setup_model(backend='numpy')
    t_steps = np.linspace(0, 2 * np.pi, 100)
    res = model.integrate(t_steps)
    
    # Expected solution: y(t) = e^{i * omega * t}
    expected = np.exp(1j * omega * t_steps)
    
    # Even if decomposed internally, the output should be complex y
    assert np.allclose(res[:, 0], expected, atol=1e-5)

@pytest.mark.skipif(so.ModelTensorflow is None, reason="Tensorflow not installed")
def test_complex_oscillator_tensorflow():
    """Test the same oscillator with tensorflow backend."""
    y = sm.symbols("y")
    omega = 1.0
    
    dy_dtD = {y: 1j * omega * y}
    
    # Tensorflow backend usually requires decomposition for complex ODEs 
    # if it uses real-only solvers internally.
    ode_sys = so.ODESys(dy_dtD, bDecompose_to_re_im=True)
    ode_sys.set_initial_state({y: 1.0 + 0j})
    
    model = ode_sys.setup_model(backend='tensorflow')
    t_steps = np.linspace(0, 2 * np.pi, 100).astype(np.float32)
    res = model.integrate(t_steps)
    
    expected = np.exp(1j * omega * t_steps)
    assert np.allclose(res[:, 0], expected, atol=1e-3)

@pytest.mark.skipif(so.ModelDiffrax is None, reason="Diffrax not installed")
def test_complex_oscillator_diffrax():
    """Test the same oscillator with diffrax backend."""
    y = sm.symbols("y")
    omega = 1.0
    
    dy_dtD = {y: 1j * omega * y}
    
    ode_sys = so.ODESys(dy_dtD)
    ode_sys.set_initial_state({y: 1.0 + 0j})
    
    model = ode_sys.setup_model(backend='diffrax')
    t_steps = np.linspace(0, 2 * np.pi, 100)
    res = model.integrate(t_steps)
    
    expected = np.exp(1j * omega * t_steps)
    assert np.allclose(res[:, 0], expected, atol=1e-5)

@pytest.mark.skipif(so.ModelDiffrax is None, reason="Diffrax not installed")
def test_complex_oscillator_decomposed_diffrax():
    """Test the same oscillator with diffrax backend and real/imag decomposition."""
    y = sm.symbols("y")
    omega = 1.0
    
    dy_dtD = {y: 1j * omega * y}
    
    ode_sys = so.ODESys(dy_dtD, bDecompose_to_re_im=True)
    ode_sys.set_initial_state({y: 1.0 + 0j})
    
    model = ode_sys.setup_model(backend='diffrax')
    t_steps = np.linspace(0, 2 * np.pi, 100)
    res = model.integrate(t_steps)
    
    expected = np.exp(1j * omega * t_steps)
    assert np.allclose(res[:, 0], expected, atol=1e-5)

def test_complex_oscillator_multidim():
    """Test a complex oscillator with spatial dimensions."""
    y = sm.symbols("y")
    z = sm.symbols("z")
    omega = 1.0
    k_prop = 1.0
    
    # dy/dt = i * omega * y, but phase depends on z
    dy_dtD = {y: 1j * omega * y}
    
    z_axes = np.linspace(0, 10, 5)
    ode_sys = so.ODESys(dy_dtD, trans_dims={z: z_axes}, bDecompose_to_re_im=True)
    
    # Initial state: y(z, t=0) = exp(i * k_prop * z)
    y0 = np.exp(1j * k_prop * z_axes)
    ode_sys.set_initial_state({y: y0})
    
    model = ode_sys.setup_model(backend='numpy')
    t_steps = np.linspace(0, 1.0, 10)
    res = model.integrate(t_steps)
    
    # Expected: y(z, t) = exp(i * k_prop * z) * exp(i * omega * t)
    # res shape: (n_t, n_vars, n_z) = (10, 1, 5)
    for i, t in enumerate(t_steps):
        expected = np.exp(1j * k_prop * z_axes) * np.exp(1j * omega * t)
        assert np.allclose(res[i, 0, :], expected, atol=1e-5)
