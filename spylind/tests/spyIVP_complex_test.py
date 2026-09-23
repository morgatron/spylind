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

def test_complex_state_dependence():
    """Test state-dependent feedback in a complex system with real/imag decomposition."""
    y = sm.symbols("y")
    E_sd = sm.symbols("E_sd")
    
    # dy/dt = i * y + E_sd, where E_sd = 0.1 * y
    # Effective equation: dy/dt = (0.1 + 1j) * y
    ode_sys = so.ODESys({y: 1j * y + E_sd}, state_dep_syms=[E_sd], bDecompose_to_re_im=True)

    def calc_E(t, dimAxes, state, driving, *args):
        return 0.1 * state[0]

    ode_sys.set_state_dependence({E_sd: calc_E})
    ode_sys.set_initial_state({y: 1.0 + 0j})

    model = ode_sys.setup_model(backend='numpy')
    t_steps = np.linspace(0, 2.0, 50)
    res = model.integrate(t_steps)

    expected = np.exp((0.1 + 1j) * t_steps)
    assert np.allclose(res[:, 0], expected, atol=1e-4)

def test_complex_output_expressions():
    """Test output_exprD in a system decomposed to real/imag parts."""
    y = sm.symbols("y")
    omega = 2.0
    
    # Population |y|^2 should be constant 1.0
    ode_sys = so.ODESys(
        {y: 1j * omega * y}, 
        output_exprD={'pop': y * sm.conjugate(y)},
        bDecompose_to_re_im=True
    )
    ode_sys.set_initial_state({y: 1.0 + 0j})
    model = ode_sys.setup_model(backend='numpy')
    t_steps = np.linspace(0, 1.0, 20)
    res = model.integrate(t_steps)

    assert np.allclose(res[:, 0], 1.0, atol=1e-5)

def test_expand_list_to_re_im():
    """Test utils.expand_list_to_re_im for mixed real/complex symbols."""
    from spylind import utils as ut
    x = sm.symbols("x", real=True)
    z = sm.symbols("z") # complex
    
    expanded = ut.expand_list_to_re_im([x, z])
    assert len(expanded) == 3
    assert expanded[0] == x
    assert str(expanded[1]) == "z_re"
    assert str(expanded[2]) == "z_im"

