import numpy as np
from numpy import pi
import qutip as q
from spylind import spylind as spl
import sympy as sm
import pytest

def test_mesolve_backends_consistency():
    """Verify that numpy and diffrax backends produce same results for Rabi oscillations."""
    H0 = 0.1 * pi * q.sigmaz()
    Omega = 2.0 * pi
    H1 = [[sm.symbols('Omega')/2, q.sigmax()]]
    rho0 = q.basis(2, 0)
    tlist = np.linspace(0, 1.0, 51)
    
    # Run with numpy
    res_np = spl.mesolve(
        [H0] + H1, rho0, tlist, 
        t_dep_fL={'Omega': lambda t: Omega}, 
        e_ops=[q.sigmaz()],
        backend='numpy'
    )
    
    # Run with diffrax
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        res_dx = spl.mesolve(
            [H0] + H1, rho0, tlist, 
            t_dep_fL={'Omega': lambda t: Omega}, 
            e_ops=[q.sigmaz()],
            backend='diffrax'
        )
        assert np.allclose(res_np, res_dx, atol=1e-5)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("JAX/Diffrax not installed")

def test_mesolve_gaussian_pulse():
    """Test time-dependent driving with a Gaussian pulse."""
    def gauss(t, width, center):
        return np.exp(-(t-center)**2 / (2 * width**2))

    tlist = np.linspace(0, 10, 101)
    H0 = 0 * q.sigmaz()
    H1 = [[sm.symbols('Omega')/2, q.sigmax()]]
    rho0 = q.basis(2, 0)
    
    # Total pulse area should be pi
    width = 1.0
    center = 5.0
    # Integral of exp(-t^2/2sigma^2) is sqrt(2pi) * sigma
    # We want area = pi, so height * sqrt(2pi) * width = pi
    height = pi / (np.sqrt(2 * pi) * width)
    
    res = spl.mesolve(
        [H0] + H1, rho0, tlist, 
        t_dep_fL={'Omega': lambda t: height * gauss(t, width, center)}, 
        e_ops=[q.sigmaz()],
        backend='numpy'
    )
    
    # Start at 1 (ground), end at -1 (excited) if pulse area is pi
    assert np.allclose(res[0], 1.0, atol=1e-5)
    assert np.allclose(res[-1], -1.0, atol=1e-3)

def test_mesolve_ensemble_backends():
    """Test ensemble simulation consistency between backends."""
    DeltaS = sm.symbols('Delta')
    H = [[DeltaS/2, q.sigmaz()], [2*pi/2, q.sigmax()]]
    rho0 = q.basis(2, 0)
    tlist = np.linspace(0, 0.5, 21)
    
    deltas = np.linspace(-2, 2, 5)
    dims = {DeltaS: deltas}
    
    res_np = spl.mesolve(H, rho0, tlist, dims=dims, e_ops=[q.sigmaz()], backend='numpy')
    
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        res_dx = spl.mesolve(H, rho0, tlist, dims=dims, e_ops=[q.sigmaz()], backend='diffrax')
        assert np.allclose(res_np, res_dx, atol=1e-5)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("JAX/Diffrax not installed")
