""" A class to ease the use of sympy expressions in ODEs

The main interface is an object representing a system of ODEs, ODESys. 
This class is initialised with a 'symbollic' description of the system. Which symbols are time dependent, which are state dependent etc. If plain numerical functions are required to evaluate some of these symbols, the requured signatures for these functions are shown. 

To actually solve the system, we make a 'Model'. To make the Model, we need to tell ODESys how to evaluate all unkown symbols and what backend it should use. We can also tell it what should be saved along the way. These are put into a ModelParameters object. The combination of a ModelParameters and an ODESys makes a Model, which can then be evaluated. 

========================
python or symbollic functions for indep variables
It will/can:
    1. Do some preprocessing to simplify and eliminate unchanging state
        variables
    2. Will prepare a fast(ish) function that evaluates the derivative, and
        only evolves the necessary bits
    3. Gives an interface to solvers to actually do the integration, given initial variables
    4. Can process the output back into the original format.
    5. Optionally takes a list of actually desired outputs, so that others potentially don't have to be calculated.



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
from box import Box
from .import utils as ut
from .backends import ModelNumpy, D_Dt_Fast_numpy 
#from .backends import ModelTensorflow, D_Dt_Fast_TF
#import tfdiffeq
from itertools import chain
from collections.abc import Mapping
from scipy import interpolate


#NOTE: This class (ODESys) is quite bloated, and needs stripping back. Trying to put most of it's functionality into the above.
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
    _output_func = None
    def __init__(self, dy_dtD, intermediate_calcs = {}, output_exprD = None, trans_dims={}, tSym=sm.symbols('t'), 
            driving_syms=[],state_dep_syms = [], constant_syms = [], bTryToSimplify=False,
                 bDecompose_to_re_im=False, sim_dtype=None):
        # Take care of any python data types on the RHS, convert them to sympy equivalents
        dy_dtD = {sym: sm.sympify(ex) for sym, ex in dy_dtD.items()}

        if bTryToSimplify:
            dy_dtD = {sym: ex.simplify() for sym, ex in dy_dtD.items()}
        N_dims = len(trans_dims)

        #dims = Munch(
        #    vals = list(dimsD.values()),
        #    symbols = list(dimsD.keys()),
        #    #shape = tuple([dim.size for dim in dimsD.values()]) if N_dims else 1,
        #    shape = tuple([dim.size for dim in dimsD.values()]),
        #    broadcastable = {sym:arr.reshape(*(k * [1] + [arr.size] + (N_dims - k - 1) * [1]))
        #                         for k, (sym, arr) in enumerate(dimsD.items()) }
        #    )
        dims = Box(
            axes = trans_dims,
            symbols = list(trans_dims.keys()),
            #shape = tuple([dim.size for dim in dimsD.values()]) if N_dims else 1,
            shape = tuple([dim.size for dim in trans_dims.values()]),
            axes_broadcastable = {sym:arr.reshape(*(k * [1] + [arr.size] + (N_dims - k - 1) * [1]))
                                 for k, (sym, arr) in enumerate(trans_dims.items()) }
            )


        propagated_state_syms, stationary_state_syms, free_syms = self.sort_symbols(
            dy_dtD)
        symsD = Box({'t': tSym,
                               'dimensions': dims.symbols,
                               'state': propagated_state_syms + stationary_state_syms,
                               'stationary_state': [],  # Not yet used
                               'intermediate': [sm.symbols('I_{}'.format(i)) for i in range(len(intermediate_calcs))],
                               'driving': driving_syms,
                               'state_dep': state_dep_syms,
                               'constants': constant_syms,
                               'unspecified': set(free_syms).difference( list(driving_syms)+list(state_dep_syms)+dims.symbols + constant_syms), 
                               })
        state_func_signature = (symsD.t, symsD.dimensions, symsD.state, \
            symsD.stationary_state, symsD.driving)

        if sim_dtype is None:
            bAll_real_state = all([sym.is_real==True for sym in symsD.state])
            sim_dtype = np.float64 if bDecompose_to_re_im or bAll_real_state else np.complex128
        
        self.symsD = symsD
        self.sim_dtype = sim_dtype
        self.dims = dims
        self.dy_dtD = dy_dtD
        self.intermediate_calcs = intermediate_calcs
        self.sim_size = int(len(dy_dtD) * np.prod(dims.shape))
        self.state_shape = tuple([len(dy_dtD), *dims.shape])
        self.state_func_signature = state_func_signature
        self.constantsD = {sym:None for sym in constant_syms} # TO CHECK
        self.output_exprD = output_exprD

        #print(self.symsD)
        #print(f"Signature for state_dep_funcs: {state_func_signature}")

        # REAL/IMAG decomposition
        # If we need to decompose, take the following steps:
        # * save original symbol breakdown
        # * save the substitutions and symbol mappings needed to get this
        # * save the original state_function_signature.
        # * Split complex symbollic expressions into two:
        #   * make a new dy_dtD with real and imaginary symbols seperated -#
        #   * make a new, expanded list of constant expressions
        #   * intermediate expressions... may be able to be kept as is, if they're just going into user code (which will be complex anyway)... but will need it's inputs to be real (outputs can be complex)
        #
        # * make a new state_dep_func signature with the real/imaginary components of arguments seperated
        # * state_dependent functions and driving functions are handled later in the relevant set_ methods
        if bDecompose_to_re_im:
            dy_dtD_new, complex_subsD, symbol_mapD = ut.expand_to_re_im(dy_dtD)
            if bTryToSimplify:
                dy_dtD_new = {sym: ex.simplify() for sym, ex in dy_dtD_new.items()}

            real_imag_conv = Box(
                symsD_orig = symsD,
                dy_dtD_orig = dy_dtD,
                state_func_signature_orig = state_func_signature,
                complex_subsD = complex_subsD,
                symbol_mapD = symbol_mapD,
                )

            propagated_state_syms, stationary_state_syms, free_syms = self.sort_symbols(
            dy_dtD_new)
            symsD.state = propagated_state_syms + stationary_state_syms
            #symsD.constants = 
            symsD = Box({'t': tSym,
                               'dimensions': dims.symbols,
                               'state': propagated_state_syms + stationary_state_syms,
                               'stationary_state': [],  # Not used yet
                               'driving': driving_syms,
                               'unspecified': ut.list_diff(free_syms, driving_syms), # TO CHECK
                               })

            #def ensure_list(el): return el if np.iterable(el) else [el]
            #state_func_signature_new = [ensure_list(symbol_mapD[sym])
                              #if sym in symbol_mapD else sym for sym in state_func_signature]
            #[[symbol_mapD[sym] if sym in symbol_mapD else sym for sym in ensure_list ]

            self.symsD =symsD
            self.state_func_signature = chain(*state_func_signature)
            self.dy_dtD = dy_dtD_new
            self.real_imag_conv = real_imag_conv
            print("realified state dependent functions signature: {}".format(
                state_func_signature))
        self.bDecompose_to_re_im = bDecompose_to_re_im


        self.state_dep_subs_D = {};
        self.state_dep_funcs_D = {};
        self.driving_funcs_D = {}

    def summary(self):
        from IPython.display import display, Latex, Markdown
        def ltx_repr_list(lst):
            lst=[obj._repr_latex_() for obj in lst]
            txt = '[ '+ ', '.join(lst) + ' ]'
            return txt

        self.show_signatures()
        display(Markdown("**Variables:**"))
        display(Latex(f"Dimensions: {ltx_repr_list(self.dims.symbols)} : {self.dims.shape}"))
        display(Latex(f"State variables: {ltx_repr_list(self.symsD.state)}") )
        display(Latex(f"Driving symbols: {ltx_repr_list(self.symsD.driving)}") )
        display(Latex(f"State-dep symbols: {ltx_repr_list(self.symsD.state_dep)}") )
        display(Latex(f"Constant symbols: {ltx_repr_list(self.symsD.constants)}") )
        display(Latex(f"Free symbols: {ltx_repr_list(self.symsD.unspecified)}") )
        state_size = np.prod([len(self.symsD.state), *self.dims.shape])
        display(Latex(f"State size: {state_size/1e3} k vars"))
        #Ideally test here wether running in an Ipython notebook
        # Plain text version
        #print(f"Dimensions: {list(self.dims.axes.keys())} ({self.dims.shape})")
        #print(f"State variables: {self.symsD.state}")
        #print(f"Driving symbols: {self.symsD.driving}")
        #print(f"State-dep symbols: {self.symsD.state_dep}")
        #print(f"Free symbols: {self.symsD.unspecified}")
        ##print(f"Intermediate calcs :{}")
        #state_size = np.product([len(self.symsD.state), *self.dims.shape])
        #print(f"State size: {state_size/1e3} k vars")

    def show_signatures(self):
        from IPython.display import display, Latex, Markdown
        def ltx_repr_list(lst):
            lst=[obj._repr_latex_()[1:-1] for obj in lst]
            txt = '[ '+ ', '.join(lst) + ' ]'
            return '$'+txt+'$'
        display(Markdown(("**Function signatures:**")))
        st_state = f"state_dep_f( {self.symsD.t._repr_latex_()}, dimAxes = {ltx_repr_list(self.dims.axes.keys())}, state = { ltx_repr_list(self.symsD.state) }, driving= { ltx_repr_list(self.symsD.driving) }, intermediate = {ltx_repr_list(self.symsD.intermediate)} )"    
        #st_state = f"state_dep_f( intermediate= { ltx_repr_list(self.symsD.intermediate) } )"    

        st_output = f"output_f( {self.symsD.t._repr_latex_()}, dimAxes = {ltx_repr_list(self.dims.axes.keys())}, state = {ltx_repr_list(self.symsD.state)}, driving= {ltx_repr_list(self.symsD.driving)}, state_dependent= {ltx_repr_list(self.symsD.state_dep)} , intermediate = {ltx_repr_list(self.symsD.intermediate)} )"    
        display(Markdown(st_state))
        display(Markdown(st_output))
        #print(st_state)
        #print(st_output)

    @staticmethod
    def sort_symbols(eqD):
        """ Sort symbols in a system of equations defined by eqD.

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
        """
        state_syms = list(eqD.keys())
        rhs = list(eqD.values())
        rhsSyms = list(sm.Matrix(rhs).free_symbols)
        free_syms = ut.list_diff(rhsSyms, state_syms)
        stationary_state_syms = [lhs for lhs, rhs in eqD.items() if rhs == 0]
        propagated_state_syms = ut.list_diff(state_syms, stationary_state_syms)
        return propagated_state_syms, stationary_state_syms, free_syms

    def set_initial_state(self, par0={}, bRealified=False):
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
        if self.bDecompose_to_re_im and not bRealified:  # We have a complex input, but we're
                                                #doing a real-only sim. Need to map complex to real/im
            # Map from position in self.state_syms to self.dy_dD
            # par0:{p1:, p2:, p3:} -> par0{p1r:, p1i:, p2r:, p2i:, p3r:, p3i:}
            if not isinstance(par0, Mapping): # if it's a list, not a dictionary, we'll make it a dict
                par0 = dict(zip(self.real_imag_conv.symsD_orig.state, par0))
            par0, _, _ = ut.expand_to_re_im(par0, bNumericalRHS=True)

        self.Npars = len(par0)  # Should make sure this matches dy_dtD
        if isinstance(par0, Mapping):
            par0 = [par0[key] for key in self.symsD.state] #order them correctly
                             
        def expand_shape(arr): 
            """Covers the case when @arr is a scalar, a full array with same size as the state space, or an array broadcastable to that."""
            if np.array(arr).shape != self.dims.shape:
                arr = arr*np.ones(self.dims.shape, dtype=self.sim_dtype)
            return arr

        par0 = np.array([expand_shape(arr) for arr in par0], dtype=self.sim_dtype)

        if 0:
            # by now par0 should an iterable of paramters in the required order. Now we'll broadcast it to
            # the other dimensions.
            # Assume a uniform initial state
            if par0.ndim == 1 and len(self.dimAxes) > 0:
                par0 = par0.reshape(-1, *len(self.dimAxes) * [1])
                newShape = (len(self.symsD['prop_state_syms']),
                            *[np.size(ax) for ax in self.dimAxes])
                par0 = np.broadcast_to(par0, newShape)

        self.par0 = par0

    def set_driving(self, driving_funcs_D={}, bAlreadyRealified=False):
        """ Set functions that depend only on current value of t

        If they're not callables assume they can be substituted in and sorted out by sympy.
        """
        for sym in list(driving_funcs_D.keys()):
            val = driving_funcs_D[sym]

            if not callable(val): #Then maybe it's a pair of arrays for interpolation?
                if np.iterable(val):
                    t_, y_ = val
                    F=interpolate.interp1d(t_,y_, bounds_error=False, fill_value=0)
                    driving_funcs_D[sym] = F
                else: # Assume it's a sympy expression
                    F= sm.lambdify(self.symsD.t, val, modules="numpy")
                    driving_funcs_D[sym] = F

        if self.bDecompose_to_re_im and not bAlreadyRealified:  # Split each F into real and imaginary outputs
            driving_funcs_D_ri = {} # This'll be the new driving_funcs_D
            for sym in driving_funcs_D:
                if sym.is_real:
                    driving_funcs_D_ri[sym] = driving_funcs_D[sym]
                else: #Ugh, this is pretty inefficient, at least for numpy
                    sym_r, sym_i = self.real_imag_conv.complex_subsD[sym].as_real_imag()
                    f = driving_funcs_D[sym]
                    driving_funcs_D_ri[sym_r] = lambda t: f(t).real
                    driving_funcs_D_ri[sym_i] = lambda t: f(t).imag
            driving_funcs_D = driving_funcs_D_ri

        self.driving_funcs_D = driving_funcs_D
        self.input_modifiers = ...
        #print(driving_funcs_D)

    def set_constants(self, constantsD):
        """Set multi-dimensional values that won't change during the simulation

        They should represent the symbols that were given in the constructor

        If the values are symbollic expressions, they will be evaluated first. Otherwise they are assumed to be numpy arrays.
        

        ... except we should probably actually not do this here...
        """
        constantsD_numerical = {}
        for key in constantsD: # loop through the entries and evaluate if they're expressions (depending on dimensions only at this point)
            in_syms = self.symsD.dimensions #+ symsD.unspecified
            if isinstance(constantsD[key], sm.Expr):
                
                evaluated  = sm.lambdify(in_syms, val, modules="numpy")(*self.dimensions.values())
                constantsD_numerical[key]=  evaluated
            else:
                constantsD_numerical[key]= constantsD[key]
            # Now it's a numerical value. It might be complex however
        if self.bDecompose_to_re_im:
            constantsD_numerical, subsD, symbol_mapD = ut.expand_to_re_im(constantsD, bNumericalRHS = True)

        self.constantsD = constantsD_numerical

        return

        

    def set_state_dependence(self, state_dep_funcs_D={}, bAlreadyRealified=False):
        """Functions that depend on state and time,
        but not history.

        E.g. E_int = E_in*exp(1j*k*z) + cumsum( P(z) )>..
        If they're not callables assume they can be substituted in and sorted out by sympy.
        """
        subsD = {}
        for sym, F in list(state_dep_funcs_D.items()): #if it's not callable, we'll just substitute it
            if not callable(F):
                subsD[sym] = F
                state_dep_funcs_D.pop(sym)

        # DECOMPOSE TO REAL/IMAGINARY PARTS-----------------------------------
        if self.bDecompose_to_re_im and not bAlreadyRealified:  # This is ugly, and ends up calling the function twice if it represents a complex value. Tensorflow (which is mainly why this exists) may sort it out on the compilation stage though.
            # To fix for numpy,  we'd need to allow a single function to return the value for two symbols.
            nDims = len(self.state_shape) - 1
            nStates = len(self.symsD_orig.prop_state_syms)
            nDriving = len(self.symsD_orig.driving_syms)

            symsD_orig = self.real_imag_conv.symsD_orig
            complex_subsD = self.real_imag_conv.complex_subsD

            which_complex_in = [0] + nDims*[0] + \
                [0 if sym.is_real else 1 for sym in symsD_orig['state']] + \
                [0 if sym.is_real else 1 for sym in symsD_orig['driving']]
            state_dep_funcs_D_ri = {}
            for sym, F in state_dep_funcs_D.items(): 
                # F_flat takes flattened complex inputs
                F_flat = ut.flatten_inputs(F, nDims, nStates, nDriving)
                which_complex_out = [0 if sym.is_real else 1]
                # F_ri_flat takes flattened real inputs
                F_ri_flat = ut.realify_np(
                    F_flat, which_complex_in, which_complex_out)

                def F_ri(t, dimAxes, states, driving): return F_ri_flat(
                    t, *dimAxes, *states, *driving)

                if sym.is_real:
                    state_dep_funcs_D_ri[sym] = lambda *args: F_ri(*args)[0]
                else:
                    sym_r, sym_i = complex_subsD[sym].as_real_imag()
                    state_dep_funcs_D_ri[sym_r] = lambda *args: F_ri(*args)[0]
                    state_dep_funcs_D_ri[sym_i] = lambda *args: F_ri(*args)[1]

            state_dep_funcs_D = state_dep_funcs_D_ri
        #FINISHED DECOMPOSITION=============================================

        self.state_dep_subs_D = subsD
        self.state_dep_funcs_D = state_dep_funcs_D

    def set_outputs(self, f=None):
        """ This is the function called at every output step. If not given, it'll just output the system state variables.
        """

        self.outputL=[]
        if f is None: # if we
            def f(t, sim_state, *args):
                res = sim_state
                return res
        #could check if it's a list of expressions and lambdify those?

        self._output_func = f

    def setup_model(self, substitutionsD = {}, bLastMinuteSimplify=False, backend = 'numpy', bForceStateDimensions=False, **model_kwargs):
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
        dy_dtD = {sym: ex.subs(self.state_dep_subs_D).subs(substitutionsD)
                  for sym, ex in self.dy_dtD.items()}
        if bLastMinuteSimplify:
            dy_dtD = {sym: ex.simplify() for sym, ex in dy_dtD.items()}

        intermediate_calcs_subbed = [expr.subs(substitutionsD) for expr in self.intermediate_calcs]
        ##
        if self.bDecompose_to_re_im:
            # Do something to dy_dtD: this should all have been done already in __init__
            pass

        if self._output_func is None:
            self.set_outputs()
        initial_state = self.par0
        # NOTE: If we're removing stationary variables, this may have some keys/values removed.
        # Currently we're not, so we'll ignore this.

        if backend=="numpy":
            d_dt_fast = D_Dt_Fast_numpy(self.symsD.t, self.symsD.dimensions, self.dims.axes.values(), dy_dtD, self.driving_funcs_D, intermediate_calcs_subbed, self.state_dep_funcs_D, constantsD =self.constantsD, bForceStateDimensions=bForceStateDimensions, dtype = self.sim_dtype)
            if self.output_exprD:
                output_func = d_dt_fast.getLambdified(self.output_exprD)
            else:
                output_func = self._output_func
            model= ModelNumpy(d_dt_fast, initial_state, output_func, **model_kwargs)

        elif backend == 'tensorflow':
            #D_Dt_Fast = D_Dt_Fast_TF
            #model = ModelTensorflow(...)
            # TODO: Need to make output expressions somehow
            d_dt_fast = D_Dt_Fast_TF(self.symsD.t, self.symsD.dimensions, self.dims.axes.values(), dy_dtD, self.driving_funcs_D, intermediate_calcs_subbed, self.state_dep_funcs_D, constantsD =self.constantsD, bForceStateDimensions=bForceStateDimensions, dtype = self.sim_dtype)
            model= ModelTensorflow(d_dt_fast, initial_state, self._output_func, **model_kwargs)
        else:
            raise ValueError(f"Don't understand the backend: {self.backend}")

        def revert(vals):
            """ Add back in the initial conditions for stationary variables, if they exist.

            Currently don't actually treat stationary variables, so this is not useful
            """
            raise NotImplementedError()
        return model







if __name__ == "__main__":
    def test_ODESys():
        pass
