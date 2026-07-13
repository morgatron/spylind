from numpy import pi
import numpy as np
from spylind import spyIVP as so
import sympy as sm;
from box import Box
from jax import numpy as jnp


def make_repeating_pulse_func(period,rel_width, pulse_area, offset_frac = 0, scl = 1):
    amp0 = pulse_area/(rel_width*period)*scl
    def f(t):
        t_ = t/(period) - offset_frac
        T = t_% 1
        amp = amp0 if T<rel_width else 0
     
        return amp
    return f

#avoids branching to be Jittable by jax
def make_repeating_pulse_func_jax(period,rel_width, pulse_area, offset_frac = 0, scl = 1):
    amp0 = pulse_area/(rel_width*period)*scl
    def f(t):
        t_ = t/(period) - offset_frac
        T = t_% 1
        #amp = amp0 if T<rel_width else 0
        amp = jnp.where(T<rel_width, amp0, 0)
     
        return amp
    return f
    
#avoids branching to be Jittable by jax
def make_alternating_pulse_func_jax(period,rel_width, pulse_area, gamma=1, offset_frac = 0 ):
    amp0 = pulse_area/gamma/(rel_width*period)
    def f(t):
        t_ = t/(period) - offset_frac
        T = t_% 1
        #amp = amp0 if T<rel_width else 0
        amp = jnp.where(T<rel_width, amp0, 0)
        amp *= jnp.where(t_%2<1,1,-1)
        #if amp:
        #    pdb.set_trace()
        #amp *= 1 if t_%2<1 else -1 #alternate sign
        return amp
    return f 
def make_alternating_pulse_func(period,rel_width, pulse_area, gamma=1, offset_frac = 0 ):
    amp0 = pulse_area/gamma/(rel_width*period)
    #@numba.jit
    def f(t):
        t_ = t/(period) - offset_frac
        T = t_% 1
        amp = amp0 if T<rel_width else 0
        #if amp:
        #    pdb.set_trace()
        amp *= 1 if t_%2<1 else -1 #alternate sign
        return amp
    return f
    
# Symbols
rS = lambda label: sm.symbols(label, real=True)
s = Box(dict(
        Om_v = sm.Matrix(rS(r'Omega_x:z')),
        B_v = sm.Matrix(rS('B_x:z')),
        anom_e_v = sm.Matrix(rS('beta^e_x:z')),
        anom_n_v = sm.Matrix(rS('beta_n_x:z')),
        lam = rS("lambda"),
        Mn = rS("M_n"),
        Me = rS("M_e"),
        R_n_dec= rS("R^n_{dec}"),

        y_e= rS(r"gamma_e"),
        R_en_se = rS("R^{en}_{se}"),
        R_e_sd = rS("R^e_{sd}"),
        Rp =rS('R_p'),  
        Rm = rS('R_m'), # pumping rate for p and m,
        sp_v = sm.Matrix(rS('s^p_x:z')),
        sm_v = sm.Matrix(rS('s^m_x:z')),

        y_n= rS(r"gamma_n"),
        R_ne_se = rS("R^{ne}_{se}"),
        R_n_sd = rS("R^n_{sd}"),
        Bbias = rS("B_{bias}"),
        Pn_v = sm.Matrix(rS('P^n_x:z')),
        Pe_v = sm.Matrix(rS('P^e_x:z')),

        t = rS('t'),     
))  

def get_d_dtDs():
    decoh_n = s.R_n_dec*sm.Matrix([-s.Pn_v[0], -s.Pn_v[1], 0])

    #q = (3 + 1)/(2 - 4/(3 + (P_v[0]**2 + P_v[1]**2 + P_v[2]**2)));

    dPe_dt = s.y_e*(s.B_v + s.lam*s.Mn*s.Pn_v + s.anom_e_v).cross(s.Pe_v) + s.Om_v.cross(s.Pe_v) + \
                  (s.Rp*s.sp_v + s.Rm*s.sm_v + s.R_en_se*s.Pn_v - (s.R_e_sd + s.Rp + s.R_en_se)*s.Pe_v)
    dPn_dt = s.y_n*(s.B_v + s.lam*s.Me*s.Pe_v + s.anom_n_v).cross(s.Pn_v) + s.Om_v.cross(s.Pn_v) + \
                    s.R_ne_se*s.Pe_v - (s.R_n_sd+ s.R_ne_se)*s.Pn_v + decoh_n
    evoDs = {
        'alk': (s.Pe_v, dPe_dt),
        'nob': (s.Pn_v, dPn_dt),
        }
    return evoDs
    
def alkali_only_d_dt():
    subsD = {
        s.Mn : 0,
        s.R_en_se: 0,
    }
    state_syms, d_dt = get_d_dtDs()['alk']
    return state_syms, d_dt.subs(subsD)

def noble_only_d_dt():
    subsD = { # Zero out the alkali interaction parts
        s.Me: 0,
        s.R_ne_se: 0,
    } 
    state_syms, d_dt = get_d_dtDs()['nob']
    return state_syms, d_dt.subs(subsD)


    #def simple_pulsed_aklali(tau = 10, duty_cycle = 0.1, Re_sd =100, y_e = 7, B_v = [None,None,None], Om_v = [0,0,0], 
    #sp_v = [0,0,1], sm_v = [0,0,0], Rp = 10000, Rm = 0, setup_model_kwargs = {}):
# A simple static CW alkali sim
def simple_cw_alkali(Rp = 100, Re_sd =100, y_e = 2*pi*7, B_v = [None,None,None], Om_v = [0,0,0], 
              sp_v = [0,0,1], sm_v = [0,0,0], Rm = 0, setup_model_kwargs = {}):
    """ 
    Any parameter set to 'None' will need to specified (probably as a driving parameter) in the resultant model

    Leave setup_model_kwargs alone unless you're sure.
    """
    
    # CREATE subD BASED ON INPUT PARAMETERS---------------------------
    subD =  dict( zip(s.Om_v, Om_v) )  \
            |dict( zip(s.anom_e_v, [0,0,0]) ) \
            |dict( zip(s.B_v, B_v) ) \
            |dict( zip(s.sp_v, sp_v) ) \
            |dict( zip(s.sm_v, sm_v) )
            
    constantsD = {
        s.y_e : y_e, #Hz/nT
        s.R_e_sd : Re_sd, #/s
        s.Rm : Rm,
    } 
    subD = subD | constantsD
    # now get rid of any symbols that are "None", as this indicates they should be left in the model
    undefinedSyms = [sym for sym,val in subD.items() if val is None]
    subD = {sym:val for sym,val in subD.items() if val is not None}
    #===============================================----------------------
    
    # ACTUAL SIMULATION DEFINITION---------------------------------------
    state_syms, d_dt =alkali_only_d_dt()
    evoD = dict( zip(state_sims, d_dt.subs(subD)))
    ode_s = so.ODESys(evoD, tSym = s.t, trans_dims={}, driving_syms=undefinedSyms)
    model = ode_s.setup_model(**setup_model_kwargs)
    #==================================================================
    #if undefinedSyms:
    print(f"Driving symbols: {undefinedSyms}") 
    
    return model
    
    
def pulsed_alkali():
    pass

def necom_comag():
    pass

def pulsed_comag():
    pass

def basic_cw_comag(B, Rp, sp = [0,0,1], state0= 6*[0]): 
    ode_s = so.ODESys(evoD,  trans_dims={}, driving_syms=[*B_v, Rp, anom_n_v[2]])
    ode_s.set_initial_state([0,0.,0., 0,0.5,.5]) 
    #ode_s.set_constants({T1:10, T2:20, Rx:0, Ry:0})
    ode_s.set_driving({B[0]:lambda t: 100 if (t>20) & (t<20.01) else 0., 
                       B[1]:0.2, 
                       B[2]:10 , 
                       Rp:make_repeating_pulse_func(0.01,0.05, 10),
                       anom_n_v[2]:make_alternating_pulse_func(0.01, 0.05, pulse_area = pi/2,gamma=constantsD[y_n]), 
                      })
    model = ode_s.setup_model(max_step = 0.0005)
