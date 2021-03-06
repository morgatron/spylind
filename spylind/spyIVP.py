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
import tfdiffeq
from itertools import chain
from collections.abc import Mapping

TF_DTYPE= 'float64'
tf.keras.backend.set_floatx(TF_DTYPE) # Presumably can get more speed, especially on a GPU, 


#tf = None

def list_diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]


def expand_list_to_re_im(sym_list):
    re_im_out_L = []
    for sym in sym_list:
        if sym.is_real:
            re_im_out_L.append(sym)
        else:
            re_im_out_L.append()
    # all complex symbols
    complex_syms = [sym for sym in sym_list if not sym.is_real]


def expand_to_re_im(eqD, bNumericalRHS=False):  # , indep_syms):
    """ Take a system of potentially complex symbol:expression pairs and return
    a similar object with all complex symbols/expressions replaced with their
    real and imaginary parts.

    The pairs are assumed to be in a dictionary, eqD. Any complex LHS symbols,
    will result in an extra pair (for real and imaginary parts). It also means replacing
    any complex RHS symbols by their real and imaginary components.

    Parameters
    ----------
    eqD : dict
        A mapping of symbol:expression pairs
    bNumericalRHS : bool
        If True, will assume that the expressions are numbers/arrays
        of sympy objects.
    Returns
    -------
    real_eqD : the realified map
    complex_subsD : a mapping from any original complex symbols to their newly
        defined componentes (e.g. x: x_r + 1j*x_i). The purpose is to be used in
        substitutions.
    symbol_mapD : a mapping from old symbols to new ones.
    """
    lhsL = list(eqD.keys())
    # display(lhsL)
    rhsL = list(eqD.values())
    if bNumericalRHS:
        rhsSyms = []
    else:
        rhsSyms = list(sm.Matrix(rhsL).free_symbols)
    indep_syms = list_diff(rhsSyms, lhsL)

    def expand_sym(sym):
        if sym.is_real:
            return sym
        else:
            return (sm.symbols(str(sym) + '^r', real=True), sm.symbols(str(sym) + '^i', real=True))
    symbol_mapD = {sym: expand_sym(sym) for sym in lhsL + indep_syms}
    complex_subsD = {sym: val[0] + sm.I * val[1]
                     for sym, val in symbol_mapD.items() if not sym.is_real}
    # display(complex_subsD) #Replacement map for symbols

    new_lhsL = []
    new_rhsL = []
    for sym, ex in zip(lhsL, rhsL):
        if bNumericalRHS:
            rhsRe = np.real(ex)
            rhsIm = np.imag(ex)
        else:
            rhsRe, rhsIm = ex.subs(complex_subsD).as_real_imag()
            rhsRe = rhsRe.simplify()
            rhsIm = rhsIm.simplify()
        new_rhsL.append(rhsRe)
        # display(sym)
        if sym.is_real:
            new_lhsL.append(sym)
        else:
            new_lhsL.extend(complex_subsD[sym].as_real_imag())
            new_rhsL.append(rhsIm)
    # lhs_sym_mapD
    return dict(zip(new_lhsL, new_rhsL)), complex_subsD, symbol_mapD


def flatten_inputs(f, nDims, nStates, nDriving):
    def wrapped(*args):
        t = args[0]
        k = 1

        dimAxs = args[k:k + nDims]
        k += nDims

        state = args[k:k + nStates]
        k += nStates

        driving = args[k:k + nDriving]

        return f(t, dimAxs, state, driving)
    return wrapped

def realify_np(f, which_complex_in=[], which_complex_out=[]):
    """ Take a function with a mixture of real and complex numbers and make
    it a 'real' function by expanding the complex arguments and complex outputs
    to real/imaginary components

    Parameters
    -----------
    f : base function to be 'realified'.
    which_complex_in : list of boolean values
        Has same length as number of input arguments to f. A false value denotes
        that argument position as a real number, a True value indicates complex.
    which_complex_out : list
        Same length as number of return values for f. Same format as which_complex_in

    Returns
    -------
    wrap_func: A function that takes and returns real numbers only

    Example:
    Consider a function of 4 parameters, with signature:
        f_in(r_p1, c_p2, c_p3, r_p4) -> c_o1, r_o2
    Here the first and fourth parameters
    are real, and the second and third are complex. It returns 2 numbers, the
    first of which is complex while the second is purely real.
    >>> def f_in( r_p1, c_p2, c_p3, r_p4):
            c_o1 = r_p1 + c_p2.real*c_p3
            r_o2 = r_p4 + c_p2.imag
            return c_o1, r_o2
    >>> f_in(1, 1+1j, 0.5+3j, 5)

    We call realify_np as
    >>> f_out = realify_np(f_in, which_complex_in = [0,1,1, 0],
                            which_complex_out = [1,0])
    >>> f_out(1, 1, 1, 0.5, 3, 5)
    That is, the resulting f_out has signature:
            f_out(r_arg1, c_arg2_r, c_arg2_i, c_arg3_r, c_arg3_i, r_arg4)
                                -> c_out1_r, c_out1_i, r_out2


    Notes:
    The current implentation is going to be relatively slow. Could maybe make it somewhat
    quicker using type hints and tuples instead of lists, but probably will need
    some kind of compilation step (e.g numba or tensorflow) to make it both general and
    fast.
    """
    N_orig_in = len(which_complex_in)  # len(real_sig.parameters)
    N_orig_out = len(which_complex_out)

    def wrap_func(*real_args):
        fin_args = []
        i = 0
        # Build up the arguments for the real function
        for k in range(N_orig_in):
            if which_complex_in[k]:
                fin_args.append(real_args[i] + 1j * real_args[i + 1])
                i += 1
            else:
                fin_args.append(real_args[i])
            i += 1

        # actually call the function
        output = f(*fin_args)
        if N_orig_out == 1:
            output = [output]

        # Same for output
        fin_output = []
        for k in range(N_orig_out):
            fin_output.append(output[k].real)
            # l+=1
            if which_complex_out[k]:
                fin_output.append(output[k].imag)
        return fin_output
    return wrap_func


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
        

class ODESolver(object):
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
        self.dims = dims
        self.dimAxes = list(dims.values())
        self.dimSyms = list(dims.keys())
        self.dim_shape = tuple([dim.size for dim in self.dimAxes])
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
                                        'state_dep_syms': list_diff(indep_syms, driving_syms),
                                        })
            #input_syms = driving_syms + state_dep_syms
            lhs_syms_orig = list(dy_dtD.keys())
            dy_dtD, complex_subsD, symbol_mapD = expand_to_re_im(dy_dtD)
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
                               'state_dep_syms': list_diff(indep_syms, driving_syms),
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
        indep_syms = list_diff(rhsSyms, state_syms)
        stationary_state_syms = [lhs for lhs, rhsin in eqD.items() if rhs == 0]
        propagated_state_syms = list_diff(state_syms, stationary_state_syms)
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
            if not isinstance(par0, Mapping):
                par0 = dict(zip(self.symsD_orig.prop_state_syms, par0))
            par0, _, _ = expand_to_re_im(par0, bNumericalRHS=True)
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
            F = tDepFD[sym]
            #if is_instance(F, RT): #RT_arg
            #    pass
            if not callable(F): #Then maybe it's a symbollic thing?
                subsD[sym] = F
                tDepFD.pop(sym)
        if self.bDecompose_to_re_im:  # Split into real and imagiunary bits
            tDepFD_ri = {}
            for sym in tDepFD:
                if sym.is_real:
                    tDepFD_ri[sym] = tDepFD[sym]
                else:
                    sym_r, sym_i = self.complex_subsD[sym].as_real_imag()
                    f = tDepFD[sym]
                    tDepFD_ri[sym_r] = lambda t: f(t).real
                    tDepFD_ri[sym_i] = lambda t: f(t).imag
            tDepFD = tDepFD_ri
        self.tDepSubsD = subsD
        self.tDepFD = tDepFD

    def set_state_dep_funcs(self, stateDepFD={}):
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
                F_flat = flatten_inputs(F, nDims, nStates, nDriving)
                which_complex_out = [0 if sym.is_real else 1]
                # F_ri_flat takes flattened real inputs
                F_ri_flat = realify_np(
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

        self.stateDepSubsD = subsD
        self.stateDepFD = stateDepFD

    def setup(self, bTryToSimplify=False, bForceStateDimensions=True):
        """ Do some of the expensive steps required before things can be integrated

        Steps:
            1. Decide which variables are necessary to integrate,
            based on which ones change and which ones are desired as output
            2. Optionally look at initial conditions to see if anything changes based on this.
            3. Compile a dy_dt_fast() function to do the actual integration as fast as possible
            4. Create a function to recreate the desired outputs based on the output of dy_dt_fast
        """

        # DO SUBSTITUTIONS
        dy_dtD = self.dy_dtD
        dy_dtD = {sym: ex.subs(self.stateDepSubsD).subs(self.tDepSubsD)
                  for sym, ex in self.dy_dtD.items()}
        if bTryToSimplify:
            dy_dtD = {sym: ex.simplify() for sym, ex in dy_dtD.items()}
        ##
        if self.bDecompose_to_re_im:
            # Do something to dy_dtD: this should all have been done already
            # dy_dtD  = dy_dtD.copy()
            # dy_dtD = self.split_RI(dy_dtD)
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
        self.d_dt_fast = D_Dt_Fast(self.tSym, self.dimSyms, self.dimAxes, self.dy_dtD, self.tDepFD, self.stateDepFD, bForceStateDimensions=bForceStateDimensions, dtype = self.default_dtype)

        if self._online_process_func is None:
            self.set_online_processing()

        def revert(vals):
            """ Add back in the initial conditions for stationary variables

            This will only be useful once we start doing automatic variable removal.
            """
            raise NotImplementedError()
        # if list_diff(state_dep_syms+t_dep_syms, indep_syms) != []
        #    raise ValueError("there are undefined symbols")

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
            results_obj =tfp.math.ode.DormandPrince(**kwargs).solve(self.d_dt_fast.__call__, tSteps[0], yInit, solution_times=tSteps)
            return results_obj 

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

    def set_online_processing(self, f=None):
        """ This is the function called at every output step. If not given, it'll just output the system state variables.
        """
        if f is None:
            self.outputL = []
            def f2(sim_state):
                self.outputL.append(self._unpack(sim_state))
        else:
            def f2(sim_state):
                state = self._unpack(sim_state)
                self.outputL.append(f(state) )

        self._online_process_func = f2





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
