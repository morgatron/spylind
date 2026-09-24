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

try:
    import jax
    import jax.numpy as jnp
except (ImportError, ModuleNotFoundError):
    jax = None
    jnp = None

from box import Box
from .import utils as ut
from .backends import ModelNumpy, D_Dt_Fast_numpy

try:
    from .backends import ModelDiffrax, D_Dt_Fast_Diffrax
except (ImportError, ModuleNotFoundError):
    ModelDiffrax = None
    D_Dt_Fast_Diffrax = None

try:
    from .backends import ModelCyRK, D_Dt_Fast_CyRK
except (ImportError, ModuleNotFoundError):
    ModelCyRK = None
    D_Dt_Fast_CyRK = None
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

    @property
    def _backend_registry(self):
        return {
            "numpy": self._setup_numpy_model,
            "diffrax": self._setup_diffrax_model,
            "cyrk": self._setup_cyrk_model,
        }
    def __init__(self, dy_dtD, intermediate_calcs = {}, output_exprD = None, trans_dims={}, tSym=sm.symbols('t'), 
            driving_syms=[],state_dep_syms = [], constant_syms = [], parameters=[], bTryToSimplify=False,
                 bDecompose_to_re_im=False, sim_dtype=None):
        # Take care of any python data types on the RHS, convert them to sympy equivalents
        dy_dtD = {sym: sm.sympify(ex) for sym, ex in dy_dtD.items()}

        if bTryToSimplify:
            dy_dtD = {sym: ex.simplify() for sym, ex in dy_dtD.items()}
        N_dims = len(trans_dims)

        dims = Box(
            axes = trans_dims,
            symbols = list(trans_dims.keys()),
            shape = tuple([dim.size for dim in trans_dims.values()]),
            axes_broadcastable = {sym:arr.reshape(*(k * [1] + [arr.size] + (N_dims - k - 1) * [1]))
                                 for k, (sym, arr) in enumerate(trans_dims.items()) }
            )


        propagated_state_syms, stationary_state_syms, free_syms = self.sort_symbols(
            dy_dtD)

        def to_symbols(sym_list):
            out = []
            for s in sym_list:
                if isinstance(s, str):
                    matched = None
                    for fs in set(free_syms) | set(dy_dtD.keys()):
                        if str(fs) == s:
                            matched = fs
                            break
                    if matched is None:
                        matched = sm.symbols(s)
                    out.append(matched)
                else:
                    out.append(s)
            return out

        driving_syms = to_symbols(driving_syms)
        state_dep_syms = to_symbols(state_dep_syms)
        constant_syms = to_symbols(constant_syms)
        parameters = to_symbols(parameters)

        symsD = Box({'t': tSym,
                               'dimensions': dims.symbols,
                               'state': propagated_state_syms + stationary_state_syms,
                               'stationary_state': [],  # Not yet used
                               'intermediate': [sm.symbols('I_{}'.format(i)) for i in range(len(intermediate_calcs))],
                               'driving': driving_syms,
                               'state_dep': state_dep_syms,
                               'constants': constant_syms,
                               'parameters': parameters,
                               'unspecified': set(free_syms).difference( list(driving_syms)+list(state_dep_syms)+dims.symbols + constant_syms + parameters), 
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
        self.parametersD = {sym: 0.0 for sym in parameters} # Default parameter values
        self.output_exprD = output_exprD

        self.bDecompose_to_re_im = bDecompose_to_re_im
        self.state_dep_subs_D = {}
        self.state_dep_funcs_D = {}
        self.driving_funcs_D = {}

        if bDecompose_to_re_im:
            self._decompose_to_re_im(bTryToSimplify)

    def set_parameters(self, paramsD):
        """Set default values for tunable parameters."""
        for sym, val in paramsD.items():
            if sym in self.parametersD:
                self.parametersD[sym] = val
            else:
                # If symbol given as string
                for psym in self.parametersD:
                    if str(psym) == str(sym):
                        self.parametersD[psym] = val
                        break

    def _decompose_to_re_im(self, bTryToSimplify=False):
        """Split complex symbolic expressions into real and imaginary parts."""
        dy_dtD_new, complex_subsD, symbol_mapD = ut.expand_to_re_im(self.dy_dtD)
        if bTryToSimplify:
            dy_dtD_new = {sym: ex.simplify() for sym, ex in dy_dtD_new.items()}

        real_imag_conv = Box(
            symsD_orig=self.symsD,
            dy_dtD_orig=self.dy_dtD,
            state_func_signature_orig=self.state_func_signature,
            complex_subsD=complex_subsD,
            symbol_mapD=symbol_mapD,
        )

        propagated_state_syms, stationary_state_syms, free_syms = self.sort_symbols(
            dy_dtD_new
        )
        
        # Expand driving syms
        new_driving = []
        for sym in self.symsD.driving:
            if sym in symbol_mapD:
                mapped = symbol_mapD[sym]
                if isinstance(mapped, tuple):
                    new_driving.extend(mapped)
                else:
                    new_driving.append(mapped)
            elif not getattr(sym, 'is_real', False):
                new_driving.extend([
                    sm.symbols(f"{sym}_re", real=True),
                    sm.symbols(f"{sym}_im", real=True)
                ])
            else:
                new_driving.append(sym)

        # Expand state_dep syms
        new_state_dep = []
        for sym in self.symsD.state_dep:
            if sym in symbol_mapD:
                mapped = symbol_mapD[sym]
                if isinstance(mapped, tuple):
                    new_state_dep.extend(mapped)
                else:
                    new_state_dep.append(mapped)
            elif not getattr(sym, 'is_real', False):
                new_state_dep.extend([
                    sm.symbols(f"{sym}_re", real=True),
                    sm.symbols(f"{sym}_im", real=True)
                ])
            else:
                new_state_dep.append(sym)

        # Create new symsD for the real-only system
        new_symsD = Box({
            't': self.symsD.t,
            'dimensions': self.dims.symbols,
            'state': propagated_state_syms + stationary_state_syms,
            'stationary_state': [],
            'intermediate': self.symsD.intermediate,
            'driving': new_driving,
            'state_dep': new_state_dep,
            'constants': self.symsD.constants,
            'parameters': self.symsD.parameters,
            'unspecified': ut.list_diff(free_syms, new_driving),
        })

        self.symsD = new_symsD
        self.state_func_signature = (new_symsD.t, new_symsD.dimensions, new_symsD.state, new_symsD.stationary_state, new_symsD.driving)
        self.dy_dtD = dy_dtD_new
        self.real_imag_conv = real_imag_conv

        if self.output_exprD is not None:
            self.output_exprD = {sym: ex.subs(complex_subsD) for sym, ex in self.output_exprD.items()}

        if self.intermediate_calcs:
            self.intermediate_calcs = [ex.subs(complex_subsD) for ex in self.intermediate_calcs]

        if self.driving_funcs_D:
            self.set_driving(self.driving_funcs_D)

        if self.state_dep_funcs_D:
            self.set_state_dependence(self.state_dep_funcs_D)

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
        self.par0 = par0

    def set_driving(self, driving_funcs_D={}, bAlreadyRealified=False):
        """ Set functions that depend only on current value of t

        If they're not callables assume they can be substituted in and sorted out by sympy.
        """
        if isinstance(driving_funcs_D, (list, tuple)):
            if self.bDecompose_to_re_im and hasattr(self, 'real_imag_conv') and len(driving_funcs_D) == len(self.real_imag_conv.symsD_orig.driving):
                driving_funcs_D = dict(zip(self.real_imag_conv.symsD_orig.driving, driving_funcs_D))
            else:
                driving_funcs_D = dict(zip(self.symsD.driving, driving_funcs_D))

        # Normalize keys so string keys match sympy symbols
        normalized_driving = {}
        for sym, val in driving_funcs_D.items():
            if isinstance(sym, str):
                match = None
                orig_driving = getattr(self.real_imag_conv.symsD_orig, 'driving', []) if hasattr(self, 'real_imag_conv') else []
                for dsym in list(self.symsD.driving) + list(orig_driving):
                    if str(dsym) == sym:
                        match = dsym
                        break
                if match is None and hasattr(self, 'real_imag_conv'):
                    for csym in self.real_imag_conv.complex_subsD:
                        if str(csym) == sym:
                            match = csym
                            break
                if match is None:
                    match = sm.symbols(sym)
                normalized_driving[match] = val
            else:
                normalized_driving[sym] = val
        driving_funcs_D = normalized_driving

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
            for sym, f in driving_funcs_D.items():
                if getattr(sym, 'is_real', False):
                    driving_funcs_D_ri[sym] = f
                else:
                    sym_r = None
                    sym_i = None
                    if hasattr(self, 'real_imag_conv'):
                        if sym in self.real_imag_conv.symbol_mapD:
                            mapped = self.real_imag_conv.symbol_mapD[sym]
                            if isinstance(mapped, tuple):
                                sym_r, sym_i = mapped
                        elif sym in self.real_imag_conv.complex_subsD:
                            sym_r, sym_i = self.real_imag_conv.complex_subsD[sym].as_real_imag()
                    if sym_r is None:
                        sym_r = sm.symbols(f"{sym}_re", real=True)
                        sym_i = sm.symbols(f"{sym}_im", real=True)
                    driving_funcs_D_ri[sym_r] = lambda t, f=f: np.real(f(t))
                    driving_funcs_D_ri[sym_i] = lambda t, f=f: np.imag(f(t))
            driving_funcs_D = driving_funcs_D_ri
            self.symsD.driving = list(driving_funcs_D.keys())

        self.driving_funcs_D = driving_funcs_D

    def set_constants(self, constantsD):
        """Set multi-dimensional values that won't change during the simulation.

        If the values are symbolic expressions, they will be evaluated first.
        Otherwise they are assumed to be numpy arrays.
        """
        constantsD_numerical = {}
        for key, val in constantsD.items():
            in_syms = self.symsD.dimensions
            if isinstance(val, sm.Expr):
                evaluated = sm.lambdify(in_syms, val, modules="numpy")(*self.dims.axes.values())
                constantsD_numerical[key] = evaluated
            else:
                constantsD_numerical[key] = val
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
        if self.bDecompose_to_re_im and not bAlreadyRealified:
            nDims = len(self.state_shape) - 1
            symsD_orig = self.real_imag_conv.symsD_orig
            complex_subsD = self.real_imag_conv.complex_subsD
            nStates = len(symsD_orig.state)
            nDriving = len(symsD_orig.driving)

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

                def F_ri(t, dimAxes, states, driving, *args, f_ri_flat=F_ri_flat):
                    return f_ri_flat(t, *dimAxes, *states, *driving)

                if sym.is_real:
                    state_dep_funcs_D_ri[sym] = lambda *args, f_ri=F_ri: f_ri(*args)[0]
                else:
                    sym_r, sym_i = complex_subsD[sym].as_real_imag()
                    state_dep_funcs_D_ri[sym_r] = lambda *args, f_ri=F_ri: f_ri(*args)[0]
                    state_dep_funcs_D_ri[sym_i] = lambda *args, f_ri=F_ri: f_ri(*args)[1]

            state_dep_funcs_D = state_dep_funcs_D_ri
            self.symsD.state_dep = list(state_dep_funcs_D.keys())
        #FINISHED DECOMPOSITION=============================================

        self.state_dep_subs_D = subsD
        self.state_dep_funcs_D = state_dep_funcs_D

    def set_outputs(self, f=None):
        """Sets the output function for the model.

        If no function is provided, the default behavior is to return the system state.
        If the system was decomposed to real/imaginary parts, the default output function
        will reassemble them back into complex numbers.
        """
        self.outputL = []
        if f is None:
            if self.bDecompose_to_re_im:
                # Reassemble complex state from real/imag components
                orig_state_syms = self.real_imag_conv.symsD_orig.state
                new_state_syms = self.symsD.state
                symbol_mapD = self.real_imag_conv.symbol_mapD
                
                reassembly_map = []
                for sym in orig_state_syms:
                    if sym.is_real:
                        reassembly_map.append(new_state_syms.index(sym))
                    else:
                        sym_r, sym_i = symbol_mapD[sym]
                        reassembly_map.append((new_state_syms.index(sym_r), new_state_syms.index(sym_i)))

                def f(t, sim_state, *args):
                    reassembled = []
                    for item in reassembly_map:
                        if isinstance(item, tuple):
                            idx_r, idx_i = item
                            reassembled.append(sim_state[idx_r] + 1j * sim_state[idx_i])
                        else:
                            reassembled.append(sim_state[item])
                    return np.array(reassembled)
            else:
                def f(t, sim_state, *args):
                    return sim_state

        self._output_func = f

    def setup_model(self, substitutionsD={}, bLastMinuteSimplify=False, backend='numpy', bForceStateDimensions=False, solver_opts=None):
        """Prepares the system for integration and returns a model object.

        This involves:
        1. Performing substitutions and optional simplification.
        2. Preparing intermediate calculations.
        3. Dispatching to the appropriate backend setup method.
        """
        # DO SUBSTITUTIONS
        dy_dtD = {sym: ex.subs(self.state_dep_subs_D).subs(substitutionsD)
                  for sym, ex in self.dy_dtD.items()}
        
        if bLastMinuteSimplify:
            dy_dtD = {sym: ex.simplify() for sym, ex in dy_dtD.items()}

        intermediate_calcs_subbed = [expr.subs(substitutionsD) for expr in self.intermediate_calcs]

        if self._output_func is None:
            self.set_outputs()

        if backend not in self._backend_registry:
            raise ValueError(f"Unsupported backend: {backend}. Available: {list(self._backend_registry.keys())}")

        setup_func = self._backend_registry[backend]
        return setup_func(dy_dtD, intermediate_calcs_subbed, bForceStateDimensions, solver_opts=solver_opts)

    def _setup_numpy_model(self, dy_dtD, intermediate_calcs_subbed, bForceStateDimensions, solver_opts=None):
        """Setup for the Scipy/Numpy backend."""
        d_dt_fast = D_Dt_Fast_numpy(
            self.symsD.t, 
            self.symsD.dimensions, 
            self.dims.axes.values(), 
            dy_dtD, 
            self.driving_funcs_D, 
            intermediate_calcs_subbed, 
            self.state_dep_funcs_D, 
            constantsD=self.constantsD, 
            parameter_syms=self.symsD.parameters,
            parameter_vals=list(self.parametersD.values()),
            bForceStateDimensions=bForceStateDimensions, 
            dtype=self.sim_dtype
        )
        
        output_func = d_dt_fast.getLambdified(self.output_exprD) if self.output_exprD else self._output_func
        return ModelNumpy(d_dt_fast, self.par0, output_func, solver_opts=solver_opts)

    def _setup_diffrax_model(self, dy_dtD, intermediate_calcs_subbed, bForceStateDimensions, solver_opts=None):
        """Setup for the Diffrax/JAX backend."""
        if ModelDiffrax is None:
            raise ImportError("Diffrax/JAX backend dependencies are not installed.")

        # Check for 64-bit precision support in JAX
        if self.sim_dtype in (np.float64, np.complex128):
            if not jax.config.read("jax_enable_x64"):
                # Attempt to enable it automatically
                jax.config.update("jax_enable_x64", True)
                
                # Check if it actually worked (might be too late)
                if not jax.config.read("jax_enable_x64"):
                    import warnings
                    warnings.warn(
                        f"Requested 64-bit precision ({self.sim_dtype}), but JAX x64 support is disabled "
                        "and could not be enabled automatically (likely JAX has already been initialized). "
                        "Diffrax will downcast to 32-bit and use loosened tolerances. To fix, run: "
                        "'jax.config.update(\"jax_enable_x64\", True)' at the very start of your script."
                    )

        d_dt_fast = D_Dt_Fast_Diffrax(
            self.symsD.t, 
            self.symsD.dimensions, 
            self.dims.axes.values(), 
            dy_dtD, 
            self.driving_funcs_D, 
            intermediate_calcs_subbed, 
            self.state_dep_funcs_D, 
            constantsD=self.constantsD, 
            parameter_syms=self.symsD.parameters,
            parameter_vals=list(self.parametersD.values()),
            bForceStateDimensions=bForceStateDimensions, 
            dtype=self.sim_dtype
        )
        
        output_func = d_dt_fast.getLambdified(self.output_exprD) if self.output_exprD else self._output_func
        return ModelDiffrax(d_dt_fast, self.par0, output_func, solver_opts=solver_opts)

    def _setup_cyrk_model(self, dy_dtD, intermediate_calcs_subbed, bForceStateDimensions, solver_opts=None):
        """Setup for the CyRK backend."""
        if ModelCyRK is None:
            raise ImportError("CyRK dependencies are not installed.")

        d_dt_fast = D_Dt_Fast_CyRK(
            self.symsD.t, 
            self.symsD.dimensions, 
            self.dims.axes.values(), 
            dy_dtD, 
            self.driving_funcs_D, 
            intermediate_calcs_subbed, 
            self.state_dep_funcs_D, 
            constantsD=self.constantsD, 
            parameter_syms=self.symsD.parameters,
            parameter_vals=list(self.parametersD.values()),
            bForceStateDimensions=bForceStateDimensions, 
            dtype=self.sim_dtype
        )
        
        output_func = d_dt_fast.getLambdified(self.output_exprD) if self.output_exprD else self._output_func
        return ModelCyRK(d_dt_fast, self.par0, output_func, solver_opts=solver_opts)

    def revert(self, vals):
        """Add back initial conditions for stationary variables (not yet implemented)."""
        raise NotImplementedError()
