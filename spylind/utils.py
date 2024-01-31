import numpy as np
import sympy as sm

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


def normaliseSmooth(det, od):
    """Divide the nominally smooth, but disctretized function "OD" by the density
    of points it's evaluated at. Can be useful for compensating when discretizing
    using a small, or changing, number of points.
    """
    det2=np.hstack([ [2*det[0]-det[1]], det, [2*det[-1]-det[-2]] ])
    dif=np.diff(det2)
    dfSm=np.interp( np.arange(0.5, dif.size-0.5, 1), np.arange(dif.size), dif)
    #print("scl: {}".format(dfSm.mean()))
    return od*dfSm
