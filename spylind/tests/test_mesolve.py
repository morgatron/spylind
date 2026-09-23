import numpy as np
from numpy import pi
import qutip as q
from spylind import spylind as spl
import sympy as sm
import pytest

def test_makeMESymb_basic():
    """Test the basic symbolic Lindblad equation generation."""
    H0 = pi * q.sigmaz()
    H1 = [[sm.symbols('E'), q.sigmax()]]
    c_ops = [[sm.sqrt(sm.symbols('gamma')), q.destroy(2)]]
    
    evoD, e_op_outL = spl.makeMESymb([H0] + H1, c_opL=c_ops, e_opL=[q.sigmaz()])
    
    # Check that we have the expected number of equations
    # For 2-level system, 2 diagonal elements + 1 coherence (complex)
    assert len(evoD) == 3
    assert len(e_op_outL) == 1

def test_mesolve_basic():
    """Test mesolve for a single qubit and compare with analytical Rabi oscillations."""
    H0 = 0 * q.sigmaz()
    Omega = 2 * pi
    H1 = [[sm.symbols('Omega')/2, q.sigmax()]]
    rho0 = q.basis(2, 0)
    tlist = np.linspace(0, 1, 101)
    
    res = spl.mesolve([H0] + H1, rho0, tlist, t_dep_fL={'Omega': lambda t: Omega}, e_ops=[q.sigmaz()])
    
    # Analytics for Rabi oscillations: <sigmaz> = cos(Omega * t)
    expected = np.cos(Omega * tlist)
    assert np.allclose(res.squeeze(), expected, atol=1e-5)

def test_mesolve_uncoupled_ensemble():
    """Test the simulation of many systems in parallel ('uncoupled')."""
    H0 = 0 * q.sigmaz()
    DeltaS = sm.symbols('Delta')
    H_ensemble = [[DeltaS/2, q.sigmaz()], [sm.symbols('Omega')/2, q.sigmax()]]
    
    rho0 = q.basis(2, 0)
    tlist = np.linspace(0, 1, 101)
    
    # Ensemble of 10 detunings
    deltas = np.linspace(-1, 1, 10)
    dims = {DeltaS: deltas}
    
    res = spl.mesolve([H0] + H_ensemble, rho0, tlist, 
                      dims=dims, 
                      t_dep_fL={'Omega': lambda t: 2*pi}, 
                      e_ops=[q.sigmaz()])
    
    # Shape should be (len(tlist), len(deltas))
    assert res.squeeze().shape == (101, 10)
    
    # For Delta=0, should be basic Rabi
    dims = {DeltaS: np.array([0.0])}
    res0 = spl.mesolve([H0] + H_ensemble, rho0, tlist, 
                      dims=dims, 
                      t_dep_fL={'Omega': lambda t: 2*pi}, 
                      e_ops=[q.sigmaz()])
    assert np.allclose(res0.squeeze(), np.cos(2*pi*tlist), atol=1e-5)

def test_mesolve_backends_consistency():
    """Verify that numpy, cyrk, and diffrax backends produce same results for Rabi oscillations."""
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
    
    # Run with CyRK
    try:
        import CyRK
        res_cyrk = spl.mesolve(
            [H0] + H1, rho0, tlist, 
            t_dep_fL={'Omega': lambda t: Omega}, 
            e_ops=[q.sigmaz()],
            backend='cyrk'
        )
        assert np.allclose(res_np, res_cyrk, atol=1e-4)
    except (ImportError, ModuleNotFoundError):
        pass

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

        # Also test decomposed diffrax
        res_dx_decomp = spl.mesolve(
            [H0] + H1, rho0, tlist, 
            t_dep_fL={'Omega': lambda t: Omega}, 
            e_ops=[q.sigmaz()],
            backend='diffrax',
            bDecompose_to_re_im=True
        )
        assert np.allclose(res_np, res_dx_decomp, atol=1e-5)
    except (ImportError, ModuleNotFoundError):
        pass

def test_mesolve_full_state_backends():
    """Verify full density matrix output consistency across backends."""
    H0 = 0.1 * pi * q.sigmaz()
    Omega = 2.0 * pi
    H1 = [[sm.symbols('Omega')/2, q.sigmax()]]
    rho0 = q.basis(2, 0)
    tlist = np.linspace(0, 0.5, 25)
    
    res_np = spl.mesolve(
        [H0] + H1, rho0, tlist, 
        t_dep_fL={'Omega': lambda t: Omega}, 
        backend='numpy'
    )
    
    try:
        import CyRK
        res_cyrk = spl.mesolve(
            [H0] + H1, rho0, tlist, 
            t_dep_fL={'Omega': lambda t: Omega}, 
            backend='cyrk'
        )
        assert np.allclose(res_np, res_cyrk, atol=1e-4)
    except (ImportError, ModuleNotFoundError):
        pass

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
