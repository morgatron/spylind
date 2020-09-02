import numpy as np
from numpy import pi
import pdb
from sympy.printing import sstr
import qutip as q
import sympy as sm
import os
from collections import Iterable   # drop `.abc` with Python 2.7 or lower
from itertools import product
from . import spyIVP as ivp


#TODO:
#* mesolve: add support for more qutip features, like
#   the ability to have a function coefficient directly
#* mesolve: Check the time offsets match up with qutip
#* mesolve: resolve sign change relative to qutip for driving terms
#* Add some labelling to outputs

def isiterable(obj):
    return isinstance(obj, Iterable)

def toDense(qobj):
    if type(qobj) is q.Qobj:
        return qobj.data.todense()
    else:
        return qobj

def SpreSmSp(op):
    opN=op.shape[0]
    opSM = sm.SparseMatrix(op)
    #This is essentially just a sparse block-matrix
    SpreS=sm.SparseMatrix(opN**2, opN**2, {(opN*k,opN*k):opSM for k in range(opN) })
    return SpreS

def SpostSmSp(op):
    N=op.shape[0]

    dct = {(N*l,N*k ):
           sm.SparseMatrix(N,N, {(i,i):op[k,l] for i in range(N)} ) for k,l in product(range(N), repeat=2)
        }
    SpostS = sm.SparseMatrix(N**2,N**2, dct)
    return SpostS

def liouvillianS(op, cOpL=[]):
    if op is not None:
        L = 1j*(SpreSmSp(op) - SpostSmSp((op)))

    else:
        N=cOpL[0].shape[0]
        #L = np.matrix(np.zeros((N**2,N**2), dtype='object'))
        L = sm.SparseMatrix(N**2, N**2, {})
    for cOp in cOpL:
        #cOp = np.matrix(cOp)
        cdc = cOp.H*cOp
        #cdc = herm_c(cOp)*cOp
        L += SpreSmSp(cOp)*SpostSmSp(cOp.H) -\
            (SpreSmSp(cdc) +SpostSmSp(cdc))/2
    return L

def seperate_DM_equation(eq):
    cohL = (Coh(eq.lhs))
    diagL = (Dia(eq.lhs))
    coh_dtL = (Coh(eq.rhs))
    diag_dtL = (Dia(eq.rhs))
    return diagL + cohL, diag_dtL + coh_dtL

def convMat(M):
    """Convert a pure symbollic matrix into a matrix of symbols (I think?)"""
    el = M[0, 0]
    st = sstr(el)
    base = st[:st.find('[')]
    symArr = sm.symbols('{0}:{1}(:{2})'.format(base, *M.shape))
    outputM = M.subs({elA: elB for elA, elB in zip(list(M), list(symArr))})
    return outputM

def makeHermS(M):
    """Make a matrix of symbols concretely Hermition by replacing lower
    diagonal elements with the complex conjugate of the upper diagonals."""
    Nstates = M.shape[0]
    rIL, cIL = np.tril_indices(Nstates, -1)
    rIU, cIU = np.triu_indices(Nstates, 1)
    M = M.subs({M[rl, cl]: sm.conjugate(M[cl, rl]) for ru, cu, rl, cl
                in zip(rIU, cIU, rIL, cIL)})
    return M

def makeHerm(m):
    """Purely numerical version of the above."""
    dia = np.diagonal(m).squeeze()
    return m + m.conjugate().T - np.diag(dia)

def getRhoLabels(labels):
    labs = [['{}{}'.format(lab1, lab2) for lab1 in labels] for lab2 in labels]
    return labs

def getTensoredLabels(*args):
    labels_list_list = list(args)
    labels_cur = labels_list_list.pop()
    mat_labels_cur = getRhoLabels(labels_cur)
    while labels_list_list:
        mat_labels_next = getRhoLabels(labels_list_list.pop())
        mat_labels_cur = _getTensoredLabels(mat_labels_next, mat_labels_cur)
    return mat_labels_cur

def _getTensoredLabels(mat_labels1, mat_labels2):
    N1 = len(mat_labels1)
    N2 = len(mat_labels2)
    labelM = np.empty((N1 * N2, N1 * N2), dtype='O')

    for i1 in np.arange(N1):
        for k1 in np.arange(N1):
            for i2 in np.arange(N2):
                for k2 in np.arange(N2):
                    labelM[i1 * N2 + i2, k1 * N2 + k2] = '{}|{}'.format(mat_labels1[i1][k1], mat_labels2[i2][k2])

    return labelM

def getTensoredRhoS(*args):
    """e.g getTensoredRhoS('123', 'abc') will return a DM with symbols labelled
    according to a space with levels labelled '1', '2', and '3' tensored with a
    space with levels labelled 'a', 'b', 'c'."""
    st = '\\rho_{{{}}}'
    labsMat = getTensoredLabels(*args)
    rhoM = np.empty_like(labsMat)
    N = rhoM.shape[0]
    for i in range(N):
        for k in range(N):
            if k < i:
                sym = sm.conjugate(rhoM[k, i])
            else:
                sym_st = st.format(labsMat[i, k])
                if k == i:
                    sym = sm.symbols(sym_st, real=True)
                else:
                    sym = sm.symbols(sym_st, complex=True)

            rhoM[i, k] = sym
    return sm.Matrix(rhoM)

def getRhoS(Nstates):
    def symName(i1, i2): return '\\rho_{{{0}|{1}}}'.format(i1, i2)
    M = np.empty((Nstates, Nstates), dtype='O')
    DI = np.diag_indices(Nstates)
    UI = np.triu_indices(Nstates, 1)
    for inds in zip(*DI):
        M[inds[0], inds[1]] = sm.symbols(symName(*inds), real=True)
    for inds in zip(*UI):
        sym = sm.symbols(symName(*inds), complex=True)
        M[inds[0], inds[1]] = sym
        M[inds[1], inds[0]] = sm.conjugate(sym)

    return sm.Matrix(M)

def constructBlochHamiltonian(gsEnergies, esEnergies, osc_strengths=1, T1_opt=-
                              1, br_ratio=0, decay_rates=None, coh_decay_rates=0):
    """Construct a parameterised Hamiltonian and collapse operators for driven
    transitions between a ground state manifold with energy splittings
    @gsEnergies and excited state splittings. Returns in the format needed by
    the setAtomParameters function (H0, H1, Hlst)

    @esEnergies. Ground and excited states are coupled by a Hamiltonian H1 according to the values
    Args:
        osc_strengths: A detuning between ground and excited states is given y the hamiltonian H2.
        T1_opt: Decays from the excited state manifold to ground occur at the rate 1/T1_opt.
        br_ratio: Ratios for second order transitions to make osc_strengths (if not given explicity)

    Returns: H0, (Esym,H1), Hlst, c_opL
    """
    Ngs = len(gsEnergies)
    Nes = len(esEnergies)
    Nstates = Ngs + Nes
    gsL = [q.basis(Nstates, n) for n in range(Ngs)]
    esL = [q.basis(Nstates, Ngs + n) for n in range(Nes)]
    H1 = None

    try:
        Nbrs = len(br_ratio)
    except TypeError:
        br_ratio = [k * br_ratio**k for k in range(4)]
        Nbrs = len(br_ratio)
    if not np.iterable(osc_strengths):
        strngth = osc_strengths
        osc_strengths = np.zeros((Ngs, Nes), dtype='O')
        osc_strengths[np.diag_indices(min(Ngs, Nes))] = 1.
        for k in range(Ngs):
            for i in range(Nes):
                df = int(abs(k - i))
                if df == 0:  # Like-to-like transition
                    osc_strengths[k, i] = strngth
                if df < len(br_ratio) and df > 0:  # Second order or further...
                    osc_strengths[k, i] = strngth * br_ratio[df]
    # --------------------------------------------------------
    # Add field interaction operators (oscillator strengths)
    Esym = sm.symbols('Ef', complex=True)
    for k, es in enumerate(esL):
        for i, gs in enumerate(gsL):
            term = Esym * toDense(es * gs.dag()) + sm.conjugate(Esym) * toDense(gs * es.dag())
            if H1 is None:
                H1 = 0.5 * sm.sqrt(osc_strengths[i, k]) * term
            else:
                H1 += 0.5 * sm.sqrt(osc_strengths[i, k]) * term

    # ----------------------------------------------
    # H0 Evolution
    H0 = q.zero_ket(Nstates) * q.zero_ket(Nstates).dag()
    Hdet = H0.copy()
    H0 = toDense(H0)
    for freq, ket in zip(gsEnergies, gsL):
        H0 += 2 * pi * -freq * toDense(ket * ket.dag())
    for freq, ket in zip(esEnergies, esL):
        H0 += 2 * pi * -freq * toDense(ket * ket.dag())

    Hdet = toDense(q.qdiags(np.hstack([np.zeros(Ngs), np.ones(Nes)]), offsets=0))

    # -------------------------------------------------
    # Decay. If decay_rate is not given, assume it scales
    #    with oscillator strength. Otherwise, T1_opt does nothing
    gamma = 1. / T1_opt
    if decay_rates is None:
        decay_rates = [[gamma * osc_strengths[l][k] for k in range(len(esL))]
                       for l in range(len(gsL))]

    c_opL = []
    for k, es in enumerate(esL):
        for i, gs in enumerate(gsL):
            #c_opL.append(sm.sqrt(gamma*osc_strengths[l][k]) * toDense(gs * es.dag()))
            c_opL.append(sm.sqrt(decay_rates[i][k]) * toDense(gs * es.dag()))

    # T2s
    if coh_decay_rates != 0:
        if not isiterable(coh_decay_rates):
            rate = coh_decay_rates
            coh_decay_rates = [[rate for k in range(len(esL))]
                               for i in range(len(gsL))]
        for k, es in enumerate(esL):
            for i, gs in enumerate(gsL):
                c_opL.append(sm.sqrt(coh_decay_rates[i][k]) * toDense(1j * gs * es.dag() - 1j * es * gs.dag()))

    # An attempt to compensate for 'edge' transitions having different rates
    if T1_opt != -1 and 0:
        gamma_tot = 1. / T1_opt  # = gamma_1 + 2*br*gamma_1
        # gamma_tot =
        gamma_1 = gamma_tot / (1 + 2 * br_ratio)
        gamma_2 = br_ratio * gamma_1
        # Make collapse operators
        c_opL = [sm.sqrt(gamma_1) * toDense(gs * es.dag()) for gs, es in zip(gsL, esL)]
        c_opL += [sm.sqrt(gamma_2) * toDense(gs * es.dag()) for gs, es in zip(gsL[1:], esL)]
        c_opL += [sm.sqrt(gamma_2) * toDense(gs * es.dag()) for gs, es in zip(gsL, esL[1:])]
        print("N c_ops: {}".format(len(c_opL)))
        print(c_opL)

    delt = sm.symbols('Delta', real=True)
    # Hlst=[
    return H0, (Esym, H1), (delt, Hdet), c_opL

def Coh(M): return list(np.array(M)[np.triu_indices(M.shape[0], 1)])


def Dia(M): return list(np.array(M)[np.diag_indices(M.shape[0])])


def All(M): return list(np.array(M)[np.triu_indices(M.shape[0], 0)])


def makeMESymb_cacheable(H_L, c_opL=[], e_opL=[], rhoS=None, bReturnMatrixEquation=False):
    """WIP: Idea is to have a cacheable version of makeMESymb as this can be
    an expensive calculation. Tricky to do for sympy reasons though...
    """
    raise NotImplementedError
    print(str(H_L), str(c_opL), str(e_opL))
    return


def makeMESymb(H_L, c_opL=[], e_opL=[], rhoS=None, bReturnMatrixEquation=False):
    """Take the Hamiltonia,coeficients and return density matrix evolution
    expressions Format for H: [H0, [coeff_sym1, H1], [coeff_sym2, H2] ...]"""
    #print('makeMESymb enter', flush=True)
    #pdb.set_trace()
    # Make the liouvillian-----------------------------------
    H0 = H_L.pop(0)
    H0 = sm.SparseMatrix(H0)
    Nstates = H0.shape[0]

    if rhoS is None:
        rhoS = getRhoS(Nstates)

    # Construct Hamiltonian part
    L = liouvillianS(H0)
    for el in H_L:
        if np.iterable(el) and not type(el) == q.Qobj and not type(el) == np.matrix and not type(el) == np.array:
            sym, op = el
        else:
            print("no coefficient for a hamiltonian element. Will assume it's 1")
            sym = 1.
            op = sm.SparseMatrix(el)
        L += sym * liouvillianS(op)

    for el in c_opL[:]:
        if np.iterable(el) and not type(el) == q.Qobj and not type(el) == np.matrix and not type(el) == np.array:
            coef, op = el
        else:
            coef = 1.
            op = el
        op = sm.SparseMatrix(op)
        L += coef * liouvillianS(None, cOpL=[op])

    # Get drho_dt
    drho_dtS = (L * rhoS.reshape(Nstates**2, 1)).reshape(
        Nstates, Nstates)  # *sm.Matrix(rs)
    drho_dtS.simplify()

    e_op_outL = []
    for op in e_opL:
        ex = sm.Trace(rhoS * op).doit()  # np.sum(array(H1[off_diag_ind])*array(rhoS)[off_diag_ind]).subs(Esym,1)
        ex = ex.simplify()
        e_op_outL.append(ex)
    eq = sm.Eq(rhoS, drho_dtS)
    if bReturnMatrixEquation: #returns the full matrix
        return eq, e_op_outL
    lhsL, rhsL = seperate_DM_equation(eq) #otherwise just the evolving bits
    return lhsL, rhsL, e_op_outL


#mesolve(H, rho0, tlist, c_ops=None, e_ops=None
def mesolve(H, rho0, tlist, c_ops=None, e_ops=None, dims = {}, t_dep_fL={}, coupling_fL={}, max_step_size=0.1, rtol=1e-6, atol=1e-10):
    """ Mimic the interface of qutip's mesolve
    """
    lhsL, rhsL, e_op_outL = makeMESymb(H, c_opL=c_ops, e_opL = e_ops)

    ode_s = ivp.ODESolver(dict(zip(lhsL, rhsL) ),  dims=dims, driving_syms= list(t_dep_fL.keys()))
    ode_s.set_driving(t_dep_fL)
    #ode_s.set_state_dep()

    # Convert a qobj state input to a density matrix
    if q.isket(rho0):
        rho0 = rho0*rho0.dag()
    if q.isoper(rho0):
        rho0= toDense(rho0)
    rho0 = Dia(rho0) + Coh(rho0)
    ode_s.set_initial_conditions(rho0 )
    e_op_f_L = [sm.lambdify(ode_s.symsD['prop_state_syms'],e_op ) for e_op in e_op_outL]
    def calc_expectation_vals(state):
        return [f(*state) for f in e_op_f_L]
    ode_s.set_online_processing(calc_expectation_vals) 
    ode_s.setup()
    out=ode_s.integrate(tlist, max_step_size=max_step_size, 
                                atol=atol, rtol=rtol)
    state_res = np.array(ode_s.outputL).squeeze()
    # Should do something about expectation values here. Probably evaluate them
    #if e_op_outL:
        #f = sm.lambdify(ode_s.state_syms, e_op_outL)
        #return f(state_res)
    return state_res
    #unpack H
    #make simulation

# Pretty printing of sympy matrices
try:
    import re
    from IPython.display import HTML, display, Latex
    import pandas as pd

    def smDataFrame(mat):
        """Takes a dataframe with sympy elements, and returns it in a display-
        friendly format."""
        pd.options.display.max_colwidth = 100
        matNew = np.reshape(list(map(lambda tc: '$' + sm.latex(tc) + '$', np.ravel(mat))), np.shape(mat))
        df = pd.DataFrame(matNew, index=mat.index, columns=mat.columns)
        return df

    def formatComplexDF(df, prec=1, title=None, topLeft=None):
        fpNumRE = r'(-?\d*(\.\d+)?)'
        imNumRE = r'(-?\d*(\.\d+)?j)'
        r = re.compile(r'(>\$?)\(?({0}\+)?{1}\)?(\$?<)'.format(fpNumRE, imNumRE))

        def f(m, prc=prec):
            op, _, rlSt, _, imSt, _, en = m.groups()
            if rlSt is None:
                if imSt == "0j":
                    rlSt = "0"
                else:
                    rlSt = ""
            parts = rlSt.split('.')
            if len(parts) > 1:
                rlSt = "{0}.{1}".format(parts[0], parts[1][:prc])

            parts = imSt[:-1].split('.')
            if len(parts) > 1:
                imSt = "{0}.{1}j".format(parts[0], parts[1][:prc])

            if imSt == "0j":
                imSt = ""
            bPls = True if (rlSt and imSt) else False
            # return rlSt +bPls*"+" +imSt
            return "{0}{1}{2}{3}{4}".format(op, rlSt, bPls * "+", imSt, en)

        htmlStr = df.to_html()
        if title is not None:
            titleStr = "<hr><H3>{}:</H3>".format(title)
        else:
            titleStr = ""
        if topLeft is not None:
            htmlStr = htmlStr.replace('<th></th>', '<th>{}</th>'.format(topLeft), 1)

        return HTML(titleStr + r.sub(f, htmlStr))
except ModuleNotFoundError:
    print("No pretty printing stuff (probably because no Pandas)")

if __name__ == "__main__":
    # Tests

    # sympy_me.py testSpline

    def test_makeMESymb():
        pass

    def test_constructBlochHamiltonian():
        pass

    # OLD TESTS
    def test_makeEvolutionFuncs():
        H0 = toDense(q.sigmaz() * q.sigmaz().dag())
        H1 = toDense(q.sigmax())
        c_opL = [toDense(q.destroy(2))]
        delt = sm.symbols('Delta', real=True)
        coefs, diagL, cohL, coh_dtM, diag_dtM, polS = makeEvolutionFuncs(
            H0, toDense(q.qeye(H0.shape[0])), 1 * [(delt, H1)], c_opL, bSymbOnly=True)
        return coefs, diagL, cohL, coh_dtM, diag_dtM, polS

    def test_makeEvolutionSymbollic():
        """Incomplete- test how the symbollic side of things works"""
        H0 = toDense(q.sigmaz() * q.sigmaz().dag())
        H1 = toDense(q.sigmax())
        c_opL = [toDense(q.destroy(2))]
        delt = sm.symbols('Delta', real=True)
        Ef = sm.symbols("Ef")
        # def makeEvolutionFuncsHardWork(H0, H1, HlstM=[], c_opL=[], rhoS=None):
        out = makeEvolutionSymbs(H0 + delt * H1, c_opL, rhoS=getRhoS(H0.shape[0]))
        return out

    def test_makeEvolutionFuncs_numerical():
        H0 = toDense(q.sigmaz() * q.sigmaz().dag())
        H1 = toDense(q.sigmax())
        c_opL = [toDense(q.destroy(2))]
        delt = sm.symbols('Delta', real=True)
        dcoh_dt, ddia_dt, calc_pol, recons_rho, init_state, Ndiag, Ncoh = makeEvolutionFuncs(
            H0, toDense(q.sigmax()), 1 * [(delt, H1)], c_opL, bSymbOnly=False)
        return dcoh_dt, ddia_dt, calc_pol, recons_rho, init_state, Ndiag, Ncoh

    def test_makeBlochEqSyms():
        H0 = toDense(q.sigmaz() * q.sigmaz().dag())
        H1 = toDense(q.sigmax())
        c_opL = [toDense(q.destroy(2))]
        delt = sm.symbols('Delta', real=True)
        Ef = sm.symbols("Ef")
        return makeBlochEqSymbs(H0, Hfield_L=[[Ef, H1]], Hother_L=[], c_opL=[], rhoS=None)

    #coefs, diagL, cohL, coh_dtM, diag_dtM, polS= test_makeEvolutionFuncs()
    #dcoh_dt, ddia_dt, calc_pol, recons_rho, init_state, Ndiag, Ncoh = test_makeEvolutionFuncs_numerical()
