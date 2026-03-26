import numpy as np
from numpy import pi
import qutip as q
from spylind import spyIVP as so
from spylind import spylind as spl
import sympy as sm

def test_mesolve_single_atom():
    """ Test simulation of a single atom
    """
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


def test_makeMESymb_basic():
    """ Test the basic symbolic Lindblad equation generation.
    """
    H0 = pi * q.sigmaz()
    H1 = [[sm.symbols('E'), q.sigmax()]]
    c_ops = [[sm.sqrt(sm.symbols('gamma')), q.destroy(2)]]
    
    evoD, e_op_outL = spl.makeMESymb([H0] + H1, c_opL=c_ops, e_opL=[q.sigmaz()])
    
    # Check that we have the expected number of equations
    # For 2-level system, 2 diagonal elements + 1 coherence (complex)
    assert len(evoD) == 3
    assert len(e_op_outL) == 1

def test_mesolve_basic():
    """ Test mesolve for a single qubit and compare with QuTiP if possible.
    """
    H0 = 0 * q.sigmaz()
    Omega = 2 * pi
    H1 = [[sm.symbols('Omega')/2, q.sigmax()]]
    rho0 = q.basis(2, 0)
    tlist = np.linspace(0, 1, 101)
    
    # Spylind mesolve
    # returns expect values if e_ops given, else states
    res = spl.mesolve([H0] + H1, rho0, tlist, t_dep_fL={'Omega': lambda t: Omega}, e_ops=[q.sigmaz()])
    
    # Analytics for Rabi oscillations: <sigmaz> = cos(Omega * t)
    expected = np.cos(Omega * tlist)
    assert np.allclose(res.squeeze(), expected, atol=1e-5)

def test_mesolve_uncoupled_ensemble():
    """ Test the simulation of many systems in paralell ("uncoupled")
    """
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
    
    # For Delta=0 (index 4 or 5 depending on linspace), should be basic Rabi
    # Since we have 10 points from -1 to 1, 0 is not exactly in there.
    # Let's use a specific Delta=0 point
    dims = {DeltaS: np.array([0.0])}
    res0 = spl.mesolve([H0] + H_ensemble, rho0, tlist, 
                      dims=dims, 
                      t_dep_fL={'Omega': lambda t: 2*pi}, 
                      e_ops=[q.sigmaz()])
    assert np.allclose(res0.squeeze(), np.cos(2*pi*tlist), atol=1e-5)
