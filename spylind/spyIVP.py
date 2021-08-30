""" Kind of an NDSolve replacement for sympy

Plan: a class called ODESolver. Takes a symbollic dy_dt as input, as well as as
python or symbollic functions for indep variables
It will/can:
    1. Do some preprocessing to simplify and eliminate unchanging state
        variables
    2. Will prepare a fast(ish) function that evaluates the derivative, and
        only evolves the necessary bits
    3. Gives an interface to solvers to actually do the integration, given initial variables
    4. Can process the output back into the original format.
    5. Optionally takes a list of actually desired outputs, so that others potentially don't have to be calculated.


# Notes for various builtins

# Decomposition into real and imaginary parameters
In general, the equations we use(especially master equations) will will have
both real and complex state variables and paramters. The simplest way to deal
with this is to make everything complex, and use an ODE solver that supports complex arithmetic. This is the default approach, but is obviously not generally optimal. However by setting
bDecompose_to_re_im, we will instead attempt to decompose all equations into
real and imaginary parts, resulting in a larger system of only real equations.

This means the following changes should be attempted by code:
* when __init__ is called, the type of symbols will be used to determine the final
    call signature. The equation dict(dy_dtD) will be decomposed.
* when set_driving is called, symbols that aren't exiplicitly set as 'real' will have
    their functions re - wrapped into two: to take the real and imaginary parts.
* when set_state_dep_funcs is called, any functions will be re - wrapped to take the new
 call signature(of pure real values)
* when set_inititial_values is called, complex(non - real) parameters will be expanded
"""
import sympy as sm
import numpy as np
from numpy import linspace, arange
import pdb
from munch import Munch, munchify
from scipy import integrate
import tensorflow as tf
import tensorflow_probability as tfp
from .tfp_fixed_step import FixedStep
from .import utils as ut

tfp.math.ode.FixedStep = FixedStep
#import tfdiffeq
from itertools import chain
from collections.abc import Mapping

TF_DTYPE= 'float64'
tf.keras.backend.set_floatx(TF_DTYPE) # Presumably can get more speed, especially on a GPU, 

#tf = None

# These objects are the actual solver, or Model. The plan is they are usually made by the ODEDefinition object. The Model's main job is to be an optimised solver of the problem at hand. It shouldn't have to know anything about the problem dimension. It will often however have access to a function which converts from the model's result space into whatever output is required. This function may be called 'online', ie during the calcuation, or may be used to process the results as a block once it's finished.

from abc import ABC, abstractmethod
class Model(ABC):
    """A wrapper around the actual integrator code, and it's results, to give some common interface. 
    Keep as thin as possible. Shouldn't have to (directly) know anything about the problem dimension.

    """
    tSteps_last_run = None
    state_at_t0 = None
    online_process_func = None

    def __init__(self, d_dt_fast):
        pass
    @abstractmethod
    def integrate(self, tSteps):
        raise NotImplementedError

    def clearOutput(self):
        self.outputL = []

    def continue_to(self, tEnd, dt):
        tCur = self.scipy_integrator.t
        if tEnd <= tCur:
            raise ValueError(f"Already integrated past {tEnd} (tCur is {tCur}")
        tSteps = np.arange(tCur+dt, tEnd, dt)#tSteps_last_run[-1]
        return self.integrate(tSteps)
    def integrate_to(self, tEnd, dt):
        tSteps = np.arange(0, tEnd, dt)
        return self.integrate(tSteps)


class ModelNumpy(Model):# Should really be called Scipy, probably!
    outputL = []
    def __init__(self, d_dt, state_at_t0, online_processing_func,  name =None, method='adams',atol = 1e-12, rtol=1e-6, max_step=0.1, **kwargs):
        """this is usually called by setup in ODESys"""

        if name is None:
            name = 'vode'
            if np.iscomplexobj(state_at_t0):
                name = 'zvode'
        self.d_dt = d_dt
        self._online_process_func =  online_processing_func
        self.initial_state = state_at_t0
        int_obj = integrate.ode(d_dt)
        #if self.bDecompose_to_re_im or self.default_dtype not in (np.complex64, np.complex128):
        int_obj.set_integrator(name, max_step=max_step, method = method,
                             **kwargs)  # ,order=5) # or adams
        int_obj.set_initial_value(state_at_t0, 0)
        self.scipy_integrator = int_obj


    def integrate(self, tSteps, **kwargs):
        I = self.scipy_integrator # useful to keep this for debugging
        outputL = []
        if tSteps[0]==I.t: # handle initial. It's a bit messy here. Hopefully can get rid of it
            tSteps= tSteps[1:]
            outputL.append(self._online_process_func(I.y))
            #self._online_process_func(cur_state_flat)
            print("Not integrating first step (it's just the initial state)")

        # Integrate
        for k, tNext in enumerate(tSteps):
            if not I.successful():
                self.lastGoodInd = k - 1
                print("Integration failed at sim time {}".format(tNext))
                break
            #print(I.t, I.y)
            cur_state_flat = I.integrate(tNext, **kwargs)
            # probably reshaping, calculating fields etc
            outputL.append(self._online_process_func(cur_state_flat))
        return np.array(outputL)

class D_Dt_Fast_TF:
    """ 
    attributes:
    * dimAxes
    * state_shape
    * Npars
    * flatten
    * unflatten
    * d_dt
    """
    def __init__(self, tSym, dimSyms, dimAxes, eqD, t_dep_FD, state_dep_FD, bForceStateDimensions = False, dtype=tf.float64):
        state_dep_syms = list(state_dep_FD.keys())
        t_dep_syms = list(t_dep_FD.keys())
        indep_syms = t_dep_syms + state_dep_syms
        state_syms = list(eqD.keys())
        rhs = list(eqD.values())

        dim_shape = tuple([len(ax) for ax in dimAxes])
        state_shape = tuple([len(eqD), *dim_shape])

        input_syms = [tSym] + dimSyms + \
            state_syms + t_dep_syms + state_dep_syms
        self.eq = Munch(lhs=input_syms, rhs=rhs)
        d_dt_lam = sm.lambdify(input_syms, rhs, modules=[
                            "tensorflow", {'conjugate': tf.math.conj}]) #Should probably not need the 'conjugate' here
        d_dt_lam = tf.function(d_dt_lam, experimental_compile=False) #<- experimental_compile is a guess

        self.d_dt_lam = d_dt_lam
        self.preShaped_dimAxes = [dAx.reshape(*(k * [1] + [dAx.size] + (len(dimAxes) - k - 1) * [1]))
                                for k, dAx in enumerate(dimAxes)]
        self.preShaped_dimAxes = tf.convert_to_tensor(self.preShaped_dimAxes, dtype = TF_DTYPE)
        self.dtype = dtype
        self.state_dep_f = list(state_dep_FD.values())
        #self.t_dep_f = list(t_dep_FD.values())
        self.t_dep_FD = t_dep_FD
        #self.state_shape = tf.convert_to_tensor(state_shape, dtype=tf.int32)
        self.state_shape = state_shape
        self.dim_shape = dim_shape
        #self.sim_size = np.product(state_shape)
        #super().__init__()

    #@tf.function
    def _d_dt(self, t, state, driving_vals, state_dep_vals):
        resL = self.d_dt_lam(t, *tf.unstack(self.preShaped_dimAxes), *tf.unstack(state), *tf.unstack(driving_vals), *tf.unstack(state_dep_vals) )
        resL = [tf.broadcast_to(tf.cast(res, TF_DTYPE), self.dim_shape) for res in resL]
        return tf.stack(resL);

    #@tf.function
    def _calc_driving_vals(self, t, **driving_params):
        if not driving_params:
            return tf.stack([f(t) for f in self.t_dep_FD.values() ]) 
        else:
            return tf.stack([f(t, *driving_params[str(sym) ]) for sym,f in self.t_dep_FD.items() ]) 

    #@tf.function
    def _calc_state_dep_vals(self, t, state, driving_vals):
        return tf.stack([f(t, self.preShaped_dimAxes, state, driving_vals) for f in self.state_dep_f ]) 
        
    @tf.function
    def __call__(self, t, cur_state_flat, **driving_params):
        if 'training' in driving_params:
            driving_params.pop('training')
        state = self._unflatten_state(cur_state_flat)
        driving_vals = self._calc_driving_vals(t, **driving_params)
        state_dep_vals = self._calc_state_dep_vals(t, state, driving_vals)
        result =self._d_dt(t, state, driving_vals, state_dep_vals)
        #return cur_state_flat
        return self._flatten_state(result)
        
    def _flatten_state(self, state): 
        #tf.print(state)
        return tf.reshape(state, (-1,) ) # .reshape(-1)

    def _unflatten_state(self, state):
        return tf.reshape(state, self.state_shape)

class D_Dt_Fast_numpy:
    """ 
    attributes:
    * dimAxes
    * state_shape
    * Npars
    * flatten
    * unflatten
    * d_dt
    """
    def __init__(self, tSym, dimSyms, dimAxes, eqD, t_dep_FD, state_dep_FD, bForceStateDimensions = False, dtype=np.complex128):
        state_dep_syms = list(state_dep_FD.keys())
        t_dep_syms = list(t_dep_FD.keys())
        indep_syms = t_dep_syms + state_dep_syms
        state_syms = list(eqD.keys())
        rhs = list(eqD.values())

        dim_shape = tuple([len(ax) for ax in dimAxes])
        state_shape = tuple([len(eqD), *dim_shape])

        input_syms = [tSym] + dimSyms + \
            state_syms + t_dep_syms + state_dep_syms
        self.eq = Munch(lhs=input_syms, rhs=rhs)

        d_dt_lam = sm.lambdify(input_syms, rhs, modules="numpy")
        if bForceStateDimensions: #Useful if the included functions don't always return a correctly shaped output
            d_dtF0 = d_dt_lam
            d_dt_lam = lambda *args: [np.broadcast_to(out, dim_shape)
                                    for out in d_dtF0(*args)]

        self.d_dt_lam = d_dt_lam
        self.d_dt_prealloc=np.zeros(
            state_shape, dtype= dtype)
        self.dimAxes = [dAx.reshape(*(k * [1] + [dAx.size] + (len(dimAxes) - k - 1) * [1]))
                                for k, dAx in enumerate(dimAxes)]
        self.dtype = dtype
        self.state_dep_f = list(state_dep_FD.values())
        self.t_dep_f = list(t_dep_FD.values())
        self.state_shape = state_shape

    def _d_dt(self, t, state, driving_vals, state_dep_vals):
        return self.d_dt_lam(t, *self.dimAxes, *state, *driving_vals, *state_dep_vals)

    def _calc_driving_vals(self, t):
        driving_vals = [f(t) for f in self.t_dep_f]
        return driving_vals
    def _calc_state_dep_vals(self, t, state, driving_vals):
        state_dep_vals = [f(t, self.dimAxes, state, driving_vals)
                            for f in self.state_dep_f]
        return state_dep_vals
        
    def __call__(self, t, cur_state_flat):
        state = self._unflatten_state(cur_state_flat)
        driving_vals = self._calc_driving_vals(t)
        state_dep_vals = self._calc_state_dep_vals(t, state, driving_vals)
        self.d_dt_prealloc[:] = self._d_dt(t, state, driving_vals, state_dep_vals)
        return self._flatten_state(self.d_dt_prealloc)

    def _flatten_state(self, state):
        return state.reshape(-1)
    def _unflatten_state(self, flat_state):
        return flat_state.reshape(*self.state_shape)
        

#NOTE: This class (ODESolver) is very bloated, and needs stripping back. Trying to put most of it's functionality into the above.
# The new ides is... (thought s in progress):
# * This is more of an ODEDefinition, or ODESys. or ODEinterface... or maybe Model??
# * The actual work will be done by a Model object (which could be called a Solver???)
# * The job of this thing is to handle the definitions, creating an optimised object for solving the system, and providing a convenient interface to get translate the output of it (maybe)
# * Typical usage:
#   sys = ODESys(dy_dtD, dims={})
#   sys.set_driving()
#   sys.set_initial_conditions()
#   sys.set_state_dependence()
#   sys.set_outputs()
#   model = sys.setup_model()
#   model.integrate_to()
class ODESys:
    """
    """
    #lambdify_modules = [{'conjugate':tf.math.conj}, 'tensorflow']
    lambdify_modules = ['numpy']
    #backend = 'tensorflow'
    backend = 'numpy'
    stateDepSubsD = {}
    stateDepFD = {}
    tDepSubsD = {}
    tDepFD = {}
    _online_process_func = None

    def __init__(self, dy_dtD, dims={}, tSym=sm.symbols('t'), driving_syms=[],
                 state_dep_syms=[],
                 bDecompose_to_re_im=False, backend='numpy', default_dtype=np.complex128):
        self.backend = backend

        self.dimAxes = list(dims.values())
        self.dimSyms = list(dims.keys())
        self.dim_shape = tuple([dim.size for dim in self.dimAxes])
        self.dims = preShaped_dimAxes = [dAx.reshape(*(k * [1] + [dAx.size] + (len(self.dimAxes) - k - 1) * [1]))
                                 for k, dAx in enumerate(self.dimAxes)]
        self.tSym = tSym
        # np.float64 if bDecompose_to_re_im else np.complex128
        self.default_dtype = default_dtype

        # Take care of any python data types
        dy_dtD = {sym: sm.sympify(ex) for sym, ex in dy_dtD.items()}
        if bDecompose_to_re_im:
            propagated_state_syms, stationary_state_syms, indep_syms = self.sort_symbols(
                dy_dtD)
            self.symsD_orig = munchify({'tSym': self.tSym,
                                        'dimSyms': self.dimSyms,
                                        'prop_state_syms': propagated_state_syms,
                                        'stationary_state_syms': stationary_state_syms,  # Not yet used
                                        'driving_syms': driving_syms,
                                        'state_dep_syms': ut.list_diff(indep_syms, driving_syms),
                                        })
            #input_syms = driving_syms + state_dep_syms
            lhs_syms_orig = list(dy_dtD.keys())
            dy_dtD, complex_subsD, symbol_mapD = ut.expand_to_re_im(dy_dtD)
            self.complex_subsD = complex_subsD
            self.symbol_mapD = symbol_mapD
            self.state_func_sig_orig = [self.tSym] + self.dimSyms + list(propagated_state_syms) + \
                list(stationary_state_syms) + driving_syms

        self.dy_dtD = dy_dtD
        self.sim_size = int(len(dy_dtD) * np.product(self.dim_shape))
        self.state_shape = tuple(
            [len(dy_dtD)] + [dim.size for dim in self.dimAxes])
        self.bDecompose_to_re_im = bDecompose_to_re_im

        propagated_state_syms, stationary_state_syms, indep_syms = self.sort_symbols(
            dy_dtD)
        self.symsD = munchify({'tSym': self.tSym,
                               'dimSyms': self.dimSyms,
                               'prop_state_syms': propagated_state_syms,
                               'stationary_state_syms': stationary_state_syms,  # Not yet used
                               'driving_syms': driving_syms,
                               'state_dep_syms': ut.list_diff(indep_syms, driving_syms),
                               })
        state_func_sig = [self.tSym] + self.dimSyms + list(propagated_state_syms) + \
            list(stationary_state_syms) + driving_syms
        if bDecompose_to_re_im:
            def ensure_list(el): return el if np.iterable(el) else [el]
            # expand_to_re_im(state_func_sig)
            state_func_sig = [ensure_list(symbol_mapD[sym])
                              if sym in symbol_mapD else sym for sym in state_func_sig]
            state_func_sig = chain(*state_func_sig)
        self.state_func_sig = state_func_sig
        print(self.symsD)
        print("state dependent functions should have signature {}".format(
            self.state_func_sig))
        if self.backend == 'tensorflow':
            import tensorflow
            import tensorflow_probability as tfp
            global tf
            tf = tensorflow
            tf.keras.backend.set_floatx(TF_DTYPE) # Presumably can get more speed, especially on a GPU, 

    @staticmethod
    def sort_symbols(eqD):
        """ Sort symbols in a system of equations defined by eqD.

        What it's supposed to do:
        Take a dict defining symbols (LHS) and their derivatives (RHS), and sort out:
            1. Which LHS symbols don't change (have a zero derivative)
            2. Which RHS symbols need specifying, that is will need to be defined externally
            3. If there are any LHS symbols that don't appear elsewhere and thus
            potentially be ignored.


        returns:
        -------
        propagated_state_syms: LHS symnbols that need to be evolved
        stationary_state_syms: LHS symbols that don't change (RHS==0)
        indep_syms: symbols appearing on the RHS that aren't state syms, which
            will need to be specified externally.

        Notes:
        ------
        At the moment, this function is mostly aspirational
        """
        state_syms = list(eqD.keys())

        rhs = list(eqD.values())
        rhsSyms = list(sm.Matrix(rhs).free_symbols)
        indep_syms = ut.list_diff(rhsSyms, state_syms)
        stationary_state_syms = [lhs for lhs, rhsin in eqD.items() if rhs == 0]
        propagated_state_syms = ut.list_diff(state_syms, stationary_state_syms)
        return propagated_state_syms, stationary_state_syms, indep_syms

    def set_initial_conditions(self, par0={}, bRealified=False):
        """ Set the initial paramters. The main logic here is to account for different ways they might be expressed.
        Possibilities are:
            * A dictionary of symbol: initial state pairs
            * A list of initial states in the appropriate order
            * An array of initial states, appropriately sized according to all the dimensions
            * The input could be needing to be 'realified' to match the new real-only variables.

        Parameters:
        -----------
        par0: dictionary of {symbol:state} pairs, or an iterable of states (in the right order)
        bRealified: whether it's being specified in the 'realified' format or
            original (potentially) complex format. Relevant only if we're using automatic
            decomposition to real-imaginary parts.
        """
        # Do something to make sure dimensions, order of symbols, are right
        #missing_symbols = set(par0.keys()).difference(self.symsD.prop_state_syms)
        if self.bDecompose_to_re_im and not bRealified:  # map complex to real
            # Map from position in self.state_syms to self.dy_dD
            # Somehow do the mapping based on a dictionary
            # par0:{p1:, p2:, p3:} -> par0{p1r:, p1i:, p2i, p3r:, p3i:}
            # for k in range(len(par0)):
            #    par = par0[k]
            #    realified_par0.append(par.real)
            # pass
            if not isinstance(par0, Mapping): # if it's a list, not a dictionary 
                par0 = dict(zip(self.symsD_orig.prop_state_syms, par0))
            par0, _, _ = ut.expand_to_re_im(par0, bNumericalRHS=True)
            #realified_par0 = []

        self.Npars = len(par0)  # Should make sure this matches dy_dtD
        if isinstance(par0, Mapping):
            par0 = np.array([par0[key]
                             for key in self.symsD.prop_state_syms], dtype=self.default_dtype)
        else:  # If not a dictionary, assume it's a list/array with correct order
            par0 = np.array(par0)

        # by now par0 should an iterable of paramters in the required order. Now we'll broadcast it to
        # the other dimensions.
        # Assume a uniform initial state
        if par0.ndim == 1 and len(self.dimAxes) > 0:
            par0 = par0.reshape(-1, *len(self.dimAxes) * [1])
            newShape = (len(self.symsD['prop_state_syms']),
                        *[np.size(ax) for ax in self.dimAxes])
            par0 = np.broadcast_to(par0, newShape)

        self.par0 = par0

    def set_driving(self, tDepFD={}):
        """ Functions that depend only on current value of t

        If they're not callables assume they can be substituted in and sorted out by sympy.
        """
        if not isinstance(tDepFD, Mapping):  # Then we'll assume it's an iterable
            tDepFD = {sym: el for sym, el in zip(
                self.symsD.driving_syms, tDepFD)}
        subsD = {}
        for sym in list(tDepFD.keys()):
            val = tDepFD[sym]
            #if is_instance(F, RT): #RT_arg
            #    pass
            if not callable(val): #Then maybe it's a sympy expression?
                F= sm.lambdify(self.tSym, val, modules="numpy")
                tDepFD[sym] = F

        if self.bDecompose_to_re_im:  # Split into real and imagiunary bits
            tDepFD_ri = {}
            for sym in tDepFD:
                if sym.is_real:
                    tDepFD_ri[sym] = tDepFD[sym]
                else: #This is inefficient, especially for numpy
                    sym_r, sym_i = self.complex_subsD[sym].as_real_imag()
                    f = tDepFD[sym]
                    tDepFD_ri[sym_r] = lambda t: f(t).real
                    tDepFD_ri[sym_i] = lambda t: f(t).imag
            tDepFD = tDepFD_ri
        self.tDepSubsD = subsD
        self.tDepFD = tDepFD
        print(subsD, tDepFD)

    def set_state_dependence(self, stateDepFD={}):
        """Functions that depend on state and time,
        but not history.

        E.g. E_int = E_in*exp(1j*k*z) + cumsum( P(z) )>..
        If they're not callables assume they can be substituted in and sorted out by sympy.
        """
        subsD = {}
        for sym, F in list(stateDepFD.items()):
            if not callable(F):
                subsD[sym] = F
                stateDepFD.pop(sym)
        # DECOMPOSE TO REAL/IMAGINARY PARTS-----------------------------------
        if self.bDecompose_to_re_im:  # Split into real and imagiunary bits
            nDims = len(self.state_shape) - 1
            nStates = len(self.symsD_orig.prop_state_syms)
            nDriving = len(self.symsD_orig.driving_syms)

            which_complex_in = [0] + nDims * [0] + \
                [0 if sym.is_real else 1 for sym in self.symsD_orig['prop_state_syms']] + \
                [0 if sym.is_real else 1 for sym in self.symsD_orig['driving_syms']]
            stateDepFD_ri = {}
            for sym, F in stateDepFD.items():
                # F_flat takes flattened complex inputs
                F_flat = ut.flatten_inputs(F, nDims, nStates, nDriving)
                which_complex_out = [0 if sym.is_real else 1]
                # F_ri_flat takes flattened real inputs
                F_ri_flat = ut.realify_np(
                    F_flat, which_complex_in, which_complex_out)

                def F_ri(t, dimAxes, states, driving): return F_ri_flat(
                    t, *dimAxes, *states, *driving)

                # def f_r(*args):
                # return F_ri(*args).real
                if sym.is_real:
                    stateDepFD_ri[sym] = lambda *args: F_ri(*args)[0]
                else:
                    sym_r, sym_i = self.complex_subsD[sym].as_real_imag()
                    stateDepFD_ri[sym_r] = lambda *args: F_ri(*args)[0]
                    stateDepFD_ri[sym_i] = lambda *args: F_ri(*args)[1]

            stateDepFD = stateDepFD_ri
        #FINISHED DECOMPOSITION=============================================

        self.stateDepSubsD = subsD
        self.stateDepFD = stateDepFD

    def set_outputs(self, f=None):
        """ This is the function called at every output step. If not given, it'll just output the system state variables.
        """
        if f is None:
            self.outputL = []
            def f2(sim_state):
                res = self._unpack(sim_state)
                return res
                #self.outputL.append(res)
        else:
            def f2(sim_state):
                state = self._unpack(sim_state)
                res = f(state)
                return res
                #self.outputL.append(f(state) )

        self._online_process_func = f2

    def setup_model(self, bTryToSimplify=False, bForceStateDimensions=True, subsD = {}, **kwargs):
        """ Do some of the expensive steps required before things can be integrated. Return a model.

        Steps:
            1. Decide which variables are necessary to integrate,
            based on which ones change and which ones are desired as output
            2. Optionally look at initial conditions to see if anything changes based on this.
            3. Compile a dy_dt_fast() function to do the actual integration as fast as possible
            4. Create a function to recreate the desired outputs based on the output of dy_dt_fast

        TODO: this would ideally jsut return an object, all compiled, which can be used for integrating. 
        The interface of the object should be just integrate_to(tEnd), integrate(tSteps), continue(tNext), reset_initial()
        """

        # DO SUBSTITUTIONS
        dy_dtD = self.dy_dtD
        dy_dtD = {sym: ex.subs(self.stateDepSubsD).subs(self.tDepSubsD).subs(subsD)
                  for sym, ex in self.dy_dtD.items()}
        if bTryToSimplify:
            dy_dtD = {sym: ex.simplify() for sym, ex in dy_dtD.items()}
        ##
        if self.bDecompose_to_re_im:
            # Do something to dy_dtD: this should all have been done already
            pass

        # NOTE: This may have some keys/values removed, if they're stationary.
        # Maybe I should just print a warning about this for the moment, rather
        # than trying to optimise automatically? YES, I SHOULD DO THAT FOR THE MOMENT.
        # (At least until I work out if it's actually faster to optimise away)
        # So for now: We can assume that all variables stay in the same order:
        # t, *dims, *state, *driving
        #to_simD = {sym:dy_dtD[sym] for sym in propagated_state_syms}
        #self.d_dt_fast = self.make_d_dt_fast(dy_dtD, self.tDepFD,
                                             #self.stateDepFD, bForceStateDimensions=bForceStateDimensions)
        if self.backend=="numpy":
            D_Dt_Fast = D_Dt_Fast_numpy
        else:
            D_Dt_Fast = D_Dt_Fast_TF
        d_dt_fast = D_Dt_Fast(self.tSym, self.dimSyms, self.dimAxes, dy_dtD, self.tDepFD, self.stateDepFD, bForceStateDimensions=bForceStateDimensions, dtype = self.default_dtype)

        if self._online_process_func is None:
            self.set_outputs()

        self.d_dt_fast = d_dt_fast
        initial_state_flat = self.flatten_state(self.par0)
        if self.backend=="numpy":
            return ModelNumpy(d_dt_fast, initial_state_flat, self._online_process_func, **kwargs)

        def revert(vals):
            """ Add back in the initial conditions for stationary variables

            This will only be useful once we start doing automatic variable removal.
            """
            raise NotImplementedError()
        # if ut.list_diff(state_dep_syms+t_dep_syms, indep_syms) != []
        #    raise ValueError("there are undefined symbols")

    # REDUNDANT BELOW HERE (HOPEFULLY)
    def make_d_dt_fast(self, eqD, t_dep_FD, state_dep_FD, bForceStateDimensions=False):
        state_dep_syms = list(state_dep_FD.keys())
        t_dep_syms = list(t_dep_FD.keys())
        indep_syms = t_dep_syms + state_dep_syms

        state_dep_f = list(state_dep_FD.values())
        t_dep_f = list(t_dep_FD.values())

        state_syms = list(eqD.keys())
        rhs = list(eqD.values())
        input_syms = [self.tSym] + self.dimSyms + \
            state_syms + t_dep_syms + state_dep_syms
        self.eq = Munch(lhs=input_syms, rhs=rhs)
        #dummy = np.zeros( (*self.state_shape), dtype='c16').reshape(-1)

        # Try to avoid closures here by using lots of default variobles
        if self.backend == 'numpy':
            preShaped_dimAxes = [dAx.reshape(*(k * [1] + [dAx.size] + (len(self.dimAxes) - k - 1) * [1]))
                                 for k, dAx in enumerate(self.dimAxes)]
            d_dtF = sm.lambdify(input_syms, rhs, modules="numpy")
            if bForceStateDimensions: #Useful if the included functions don't always return a correctly shaped output
                d_dtF0 = d_dtF
                d_dtF = lambda *args: [np.broadcast_to(out, self.dim_shape)
                                       for out in d_dtF0(*args)]

            def flatten_state(state): return state.reshape(-1)

            def unflatten_state(
                state, shape=self.state_shape): return state.reshape(*shape)

            def dy_dt(t, cur_state_flat,
                      dimAxes=preShaped_dimAxes,
                      Npars=self.Npars,
                      state_shape=self.state_shape,
                      d_dt_prealloc=np.zeros(
                          self.state_shape, dtype=self.default_dtype),
                      d_dtF=d_dtF,
                      flatten=flatten_state, unflatten=unflatten_state,
                      state_dep_f=tuple(state_dep_f), t_dep_f=tuple(t_dep_f)):
                state = unflatten(cur_state_flat)
                driving_vals = [f(t) for f in t_dep_f]
                state_dep_vals = [f(t, dimAxes, state, driving_vals)
                                  for f in state_dep_f]
                d_dt_prealloc[:] = d_dtF(
                    t, *dimAxes, *state, *driving_vals, *state_dep_vals)
                return flatten(d_dt_prealloc)
        else:  # Tensorflow it...
            #preShaped_dimAxes = tf.constant([dAx.reshape(*(k * [1] + [dAx.size] + (len(self.dimAxes) - k - 1) * [1]))
            #                                 for k, dAx in enumerate(self.dimAxes)], dtype=tf.complex128)
            preShaped_dimAxes = tf.constant([dAx.reshape(*(k * [1] + [dAx.size] + (len(self.dimAxes) - k - 1) * [1]))
                                             for k, dAx in enumerate(self.dimAxes)], dtype=TF_DTYPE)
            d_dtF = sm.lambdify(input_syms, rhs, modules=[
                                "tensorflow", {'conjugate': tf.math.conj}]) #Should probably not need the 'conjugate' here
            d_dtF = tf.function(d_dtF, experimental_compile=True,
                                experimental_relax_shapes=True)

            def flatten_state(state): return tf.reshape(
                state, [self.sim_size])  # .reshape(-1)

            def unflatten_state(
                state, shape=self.state_shape): return tf.reshape(state, shape)
            # def call_d_dtF(t, dimAxes, state, driving_vals, state_dep_vals):
            #    return d_dtF(t, *dimAxes, *tf.unstack(state), *driving_vals, *state_dep_vals)

            @tf.function
            def tf_driving_F(t):
                return tf.stack([f(t) for f in t_dep_f ]) 
            @tf.function
            def tf_state_dep_F(t, state, driving_vals):
                return tf.stack([f(t, preShaped_dimAxes, state, driving_vals) for f in state_dep_f ]) 

            @tf.function(experimental_compile=True, experimental_relax_shapes=True)
            def dy_dt(
                t,
                cur_state_flat,
                dimAxes=preShaped_dimAxes,
                d_dtF=d_dtF,
                flatten=flatten_state,
                unflatten=unflatten_state,
                state_dep_f= tf_state_dep_F,
                driving_f = tf_driving_F,
                                ):
                #driving_vals = [f(t) for f in t_dep_f]
                #state = tf.reshape(cur_state_flat, state_shape)
                state = unflatten_state(cur_state_flat)
                driving_vals = driving_f(t)
                state_dep_vals = state_dep_f(t, state, driving_vals)
                d_dt_prealloc[:] = d_dtF(
                    t, *tf.unstack(dimAxes), *tf.unstack(state), *tf.unstack(driving_vals), *tf.unstack(state_dep_vals) )


                #d_dt_prealloc = d_dtF(tf.cast(t, dtype=tf.complex128),
                #                      *tf.unstack(dimAxes), *tf.unstack(state), E)
                return flatten_state(d_dt_prealloc)
                #return tf.reshape(d_dt_prealloc, [sim_size])

        self._d_dtF = d_dtF
        self.d_dt_fast = dy_dt
        return dy_dt

    def flatten_state(self, state):
        return state.reshape(-1)  # reshape(self.Npars, -1)

    def unflatten_state(self, flat_state):
        return flat_state.reshape(*self.state_shape)

    @tf.function(experimental_compile=False)
    def integrate_tf(self, tSteps, yInit, **kwargs):
        if 1: #tfp method
            if 1: #adaptibe step
                results_obj =tfp.math.ode.DormandPrince(**kwargs).solve(self.d_dt_fast.__call__, tSteps[0], yInit, solution_times=tSteps)
            else: #fixed step
                if 'step_size' not in kwargs:
                    kwargs['step_size']= tSteps[1]-tSteps[0]
                results_obj =tfp.math.ode.FixedStep(**kwargs).solve(self.d_dt_fast.__call__, tSteps[0], yInit, solution_times=tSteps)
            return results_obj 
        else: # The old tfdiffeq method
            class TFdy_dt(tf.keras.Model):
                def __init__(self, f):
                    self.Nevals = tf.Variable(0)#tf.convert_to_tensor(0, dtype=tf.int64)
                    self.f = tf.function(f)
                    super().__init__()
                @tf.function
                def call(self, t, state_flat):
                    self.Nevals.assign_add(1)
                    return self.f(t,state_flat)
            tf_model = TFdy_dt(f=self.d_dt_fast.__call__)
            sol = tfdiffeq.odeint(self, y_init, tSteps_T, **kwargs)
            #method=method, atol=atol, rtol=rtol, options=solver_options)
            return sol


    def integrate(self, tSteps, max_step_size=.1, atol=1e-12, rtol=1e-6, method = 'dopri5', solver_options={}, grad_pair = [], **kwargs):
        if self.backend == 'numpy':
            r = integrate.ode(self.d_dt_fast)
            if self.bDecompose_to_re_im or self.default_dtype not in (np.complex64, np.complex128):
                r.set_integrator('vode', method='adams', max_step=max_step_size,
                                 atol=atol, rtol=rtol)  # ,order=5) # or adams
                #r.set_integrator('dopri5', max_step=max_step_size,
                #                atol=atol, rtol=rtol)  # ,order=5) # or adams
            else:
                r.set_integrator('zvode', method='adams', max_step=max_step_size,
                                 atol=atol, rtol=rtol)  # ,order=5) # or adams
            # r.set_integrator('zvode', method='bdf',max_step=max_step_size, atol=1e-12, rtol=1e-6)#,order=5) # or adams
            #par0=self._pack(self.params.initial.P, self.params.initial.w, self.params.initial.pop)
            self.clearOutput()
            self.integrateObject = r # useful to keep this for debugging
            if tSteps[0]==0:
                tSteps= tSteps[1:]
                cur_state_flat = self.flatten_state(self.par0)
                r.set_initial_value(cur_state_flat, 0)
                self._online_process_func(cur_state_flat)
                print("Not integrating first step (it's just the initial state)")
            self.tSteps = tSteps #+ (tSteps[3] - tSteps[2]) / 2
            # Integrate
            for k, tNext in enumerate(self.tSteps):
                #print(r.t, r.y)
                if not r.successful():
                    self.lastGoodInd = k - 1
                    print("Integration failed at sim time {}".format(tNext))
                    break
                cur_state_flat = r.integrate(tNext)
                # probably reshaping, calculating fields etc
                self._online_process_func(cur_state_flat)
            return np.array(self.outputL)
        elif self.backend == 'tensorflow':
            y_init = tf.convert_to_tensor(self.flatten_state(
                self.par0), dtype=TF_DTYPE)
            tSteps = tf.convert_to_tensor(self.flatten_state(
                tSteps), dtype=TF_DTYPE)
            if grad_pair:
                if len (grad_pair) != 2:
                    raise ValueError("grad_pair must be length 2: a target (loss funciton) and a list of variables to differentiaite w.r.t")
                loss = grad_pair[0]
                with tf.GradientTape(watch_accessed_variables=False, persistent=False) as tape:
                    for var in grad_pair[1]:
                        tape.watch(var)
                    results_obj = self.integrate_tf(tSteps, y_init, **kwargs)
                    states = results_obj.states
                    states_unflat= tf.reshape(results_obj.states, (states.shape[0], *self.state_shape))
                    loss = grad_pair[0](states_unflat)
                gradients = tape.gradient(loss, grad_pair[1])

                return states_unflat, loss, gradients
            else:
                results_obj = self.integrate_tf(tSteps, y_init,**kwargs)
                print("num function evaluations: {}".format(results_obj.diagnostics.num_ode_fn_evaluations))
                states_unflat= tf.reshape(results_obj.states, (results_obj.states.shape[0], *self.state_shape))
                return states_unflat#, results_obj
            ##dt = self.d_dt_fast
            ##dt(0.1, y_init)
            #tf_model = TFdy_dt(f=self.d_dt_fast.__call__)
            #sol = tfdiffeq.odeint(tf_model, y_init, tSteps_T, method=method, atol=atol, rtol=rtol, options=solver_options)
            #self.integration_diag = results_obj.diagnostics
            #print("Nevals: {}".format(tf_model.Nevals))
            #dErr_dIn = g.gradient(err_func, params )
            #res = tfp.math.ode.DormandPrince().solve(self.d_dt_fast, 0, y_init,
            #                                         solution_times=tSteps)
        else:
            print("Unrecognised backend:", self.backend)

    def _pack(self, state):
        """Convert from output format into internal sim format
        """
        return self.flatten_state(state)

    def _unpack(self, sim_state):
        """Convert from internal sim format to output format
        """
        return self.unflatten_state(sim_state)

    def clearOutput(self):
        self.outputL = []



if __name__ == "__main__":
    def test_ODESolver():
        p1S, p2S, zS, tS, F, F2, G, DeltaS = sm.symbols(
            'x, y, z, t, F, F2, G Delta')
        d_dtD = {p1S: p1S**2, p2S: zS - 2 + F2}

        ode_s = ODESolver(d_dtD, dims=[zS, DeltaS])
        ode_s.setIndepFuncs(tDepF={F: sm.cos(10 * t), F2: lambda t: t**2},
                            state_dep_funcs={G: lambda p1, p2: DeltaS * (p1**2 + p2)})
        ode_s.set_initial_conditions({p1S: arange(15).reshape(3, 5),
                                      p2S: arange(15).reshape(3, 5)})
        ode_s.prepare_to_ingegrate()
        return ode_s
