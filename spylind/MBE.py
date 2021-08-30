""" 

A (tempoary?) replacement for the old pyMBE.py, using spylind etc. Not sure if this is a good
long term idea, since it's a pretty clunky interface. Mostly I'm just trying to avoid re-writing examples!
"""

from . import spyIVP as svp
from . import spylind as spl

"""
TODOS:
* Add a new propagation method automatically generated from a users Hamiltonian
* Try to separate out the individual calculation methods in some useful way.
"""
from scipy import integrate, interpolate
import numpy as np
from numpy import pi, newaxis, exp
from matplotlib import pyplot as plt
from collections import namedtuple
import sys
from munch import Munch as Bunch #<- just for holding the results
import pdb
from scipy.linalg import toeplitz
from scipy import interpolate
from sympy.printing import sstr
#from pylab import *
import qutip as q
import sympy as sm
import os
from . import misc
#from helper import *
#import helper as h
import tensorflow as tf
from builtins import sum
#pd.set_option('precision',3)
#FieldInfo = namedtuple("Field_info", ["wavelength", "osc_strength", "cavity"])




class MBE_1D_SVEA(object):
    """ Class to simulate the Maxwell-Bloch equations in 1D, assuming the slowly-varying envelope approximation in time but not space. Should be useful for simulating small cavities driven near resonance, simulating sub-wavelength hole-burning, or just having counter-propagating beams for some reason.


    Rough idea/plan for how this class is/should be used:


    Create a class giving the overall type of simulation to do (cavity or no, an inhomogeneous line or no, wavelength or no, one or two sim directions))

    Methods:
    setup_params(): assign the input parameters to the appropriate places
    get_dAll_dt_XX(): get the function used in the integration
    integrate(tEnd): integrate until tEnd
    getOutputs(): return the output fields vs time and the polarizations





    """

    FieldsType=namedtuple('Fields', ['t', 'EinpFw', 'EinpRv', 'EoutFw', 'EoutRv', 'EintFw', 'EintRv', 'Eintern'])
    ResType=namedtuple('Results', ['t', 'z',
            'det',  'P', 'w','pop', 'fields'] )


    def __init__(self, detAx=[0], cavityParams={}, pts_per_lambda=5, length_in_lambdas=1):
        """

        :param tAx: The time axis relevant to the input fields.
        :param Ein_Fw: Field input from the front.
        :param Ein_Rv: Field input from the back
        :param zAx: depth axis over which to simulate the medium
        :param lam: wavelength of the input light
        :param detAx: the detunings of the atoms densities in lineShape
        :param lineShape: the optical depth as a function of detuning
        :param cavityParams: parameters for a cavity surrounding the medium. Empty means no cavity.
        """
        detAx=np.atleast_1d(detAx)
        Nz=int(length_in_lambdas*pts_per_lambda)
        zAx=np.linspace(0, length_in_lambdas, Nz)
        self.params=Bunch(
                    length=length_in_lambdas,
                    zAx=np.array(zAx),
                    dz=zAx[1]-zAx[0] if zAx.size>1 else 1,
                    detAx=np.array(detAx),
                    Nz=len(zAx),
                    Ndet=len(detAx),
                    cavityParams=cavityParams,
                    inputs=Bunch(),
                    atoms=Bunch(),
                    initial=Bunch(),
                )

        self.setFieldCavityParams(**cavityParams)

        #self.setInitialState();
    def setCavityParams(self, R1, R2):
        """This is redundant"""
        self.setFieldCavityParams(R1,R2)

    def setFieldCavityParams(self, R1=0.0,R2=0.0, L=None ):
        """ Currently just cavity...
        """
        if L is None:
            L=self.params.length
        cavityParams=Bunch(
                r1=np.sqrt(R1) + 0j, r2=np.sqrt(R2) + 0j,
                t1=np.sqrt(1-R1) + 0j, t2=np.sqrt(1-R2) + 0j,
                L=L,
            )
        self.params.cavityParams=cavityParams
        self.excArrs=self.calcCavityExcitationArrays(self.params.zAx, **cavityParams)
        return

    def setFieldTypes(self, E1, *args):
        """ Set information about fields
        Fields (E1, etc) should have paremeters tr_strengths
        and wavelength.
        Questions about format for tr_strengths? Probably just an Ngs x Nes matrix? That's the current format. It means we can't include transitions between ground states, but that is likely an unusual situation, and perhaps that wouldn't be too hard to implement in the future.
        """
        raise NotImplementedError
        fields = [E1] + args
        fields_out = []
        for field in fields:
            if type(field) is not FieldInfo:
                field = FieldInfo(*field)
            fields_out.append(field)
            assert(len(field)>=2)
        self.params.fields = fields_out

    @staticmethod
    def calcCavityExcitationArrays(z, r1, r2, t1, t2, L):
        """ Pre-calculate a number of arrays that are used in the analysis.

        I'm calling these 'excitation arrays' which is not a good name but it's what I've got so far. The modes are all resultant from a simple geometric series. These are:
        @excReflFromLeft: The mode excited by light incident leftwards on the left cavity mirror due to reflecting.
        @excReflFromRight: Mode excited due to reflection off right mirror
        @excEinFw: mode excited due to field incident (moving right) on the left mirror.
        @excEinRv: mode excited due to field incdent (moving left) on right mirro.


        ... have to be careful, because reverse mode accumulates phase opposite to forward. Should instead sum all contributions to reverse and forward modes
        independently, then multiply at end.

        Also note: L is in units of wavelength
        """
        k=2*pi/1.

        import sympy
        res_file_path = os.path.join(os.path.dirname(__file__), 'MBE_mat.txt')
        txt = open(res_file_path).read()

        M_s=sympy.sympify(txt)
        r1_s, r2_s, kL_s =sympy.symbols("r_1, r_2, kL")
        Mf=sympy.lambdify([r1_s, r2_s, kL_s], M_s)
        M=Mf(r1, r2, k*L)

        rvM=exp(1j*k*(L-z) )
        fwM=exp(1j*k* z )
        alpha=r1*r2*exp(2j*k*L)
        geoSumAlpha= 1/(1-alpha)
        excArrs=Bunch()
        excArrs.M=M

        excArrs.fwMode=fwM
        excArrs.rvMode=rvM
        return excArrs

    def setInputs(self, EinFw=lambda t: 0, EinRv=lambda t: 0, tAx=None):
        """ Mainly just does the job of converting field inputs to a consistent format.
        Can take either an array of values with corresponding @tAx, which will get interpolated.
        OR, just takes straight functions.
        """

        #We need to be able to calculate the input fields at arbitrary times, ie have them in
        #the form Ein(t). If they're arrays instead of functions, we'll make an interpolation
        #function out of them. In that case, tAx needs to be defined.
        if not callable(EinFw):
            EinR=interpolate.interp1d(tAx,EinFw.real, bounds_error=False, fill_value=0)
            EinI=interpolate.interp1d(tAx,EinFw.imag, bounds_error=False, fill_value=0)
            EinFw=lambda t: (EinR(t) + 1j*EinI(t))
        if not callable(EinRv):
            EinRvR=interpolate.interp1d(tAx,EinRv.real, bounds_error=False, fill_value=0)
            EinRvI=interpolate.interp1d(tAx,EinRv.imag, bounds_error=False, fill_value=0)
            EinRv=lambda t: (EinRvR(t) + 1j*EinRvI(t))


        self.params.inputs.EinFw=EinFw
        self.params.inputs.EinRv=EinRv
        #self.dAll_dt=get_dAll_dt_noCav(p.inputs.EinFw, p.inputs.EinRv, p.Nz, p.Nt, p.Ndet, p.gamma)

    def setInitialState(self,  pop0=[1], simOutput=None):
        """ Converts initial conditions in whatever form to one the sim can take.
        """

        if simOutput is not None:
            all0=np.array(list(simOutput.diag) + list(simOutput.coh))
        else:
            print("setInitialState")
            self.params
            if np.sum(pop0) != 1:
                print("Warning: populations don't add up to 1. Scaling...")
                pop0/= np.sum(pop0)#linalg.norm(pop0)
            rho0 = np.diag(pop0).astype('c16') #make diagonal matrix
            all0 = np.array(spl.Dia(rho0) + spl.Coh(rho0))

        #self.params.initial.rho = rho0
        self.params.initial.all = all0

        if 0:
            #expand does nothing if parameters are already arrays, otherwise converts scalars to complex arrays of the right shape
            expand=lambda param: np.broadcast_to(np.array(param, dtype='c16'),
                        [self.params.Nz, self.params.Ndet])
            self.params.initial.P = expand(P0)
            self.params.initial.w = expand(w0)
            self.params.initial.popT = expand(pop0)

    def setAtomParams(self, lineShape=1, gsEnergies=[0], esEnergies=[0], T1_opt=1, osc_strengths=1, br_ratio=.1, decay_rates=None, coh_decay_rates=0, bSmoothSpect=True, detAx=None, gsLabels=None, esLabels=None):
        """ Sets properties for all atoms. This includes inhomogeneous broadening, decay times, optical depth.
        """
        print("Setting atom params")
        #Make sure lineShape is an array
        if detAx is None:
            detAx=self.params.detAx
        detAx=np.atleast_1d(detAx)

        if callable(lineShape):
            lineShape=lineShape(detAx)
        lineShape=np.broadcast_to(np.array(lineShape,dtype='f8'), detAx.size)
        if lineShape.size!=detAx.size:
            raise ValueError("detAx and lineShape must have the same size")
        if bSmoothSpect and lineShape.size>2:
            lineShape = misc.normaliseSmooth(detAx, lineShape)


        osc_strengths_sym, T1_opt_sym, br_ratio_sym = sm.symbols('A, T1, b', real=True, positive=True)
        subsD = {T1_opt_sym: T1_opt,  br_ratio_sym:br_ratio}

        # Old: try to avoid recalculations by using symbols
        if 0:
            osc=osc_strengths_sym
            try:
                if osc_strengths.size>1:
                    osc=osc_strengths
            except:
                pass
        else:
            osc = osc_strengths
        H0, [Esym, H1], [DetSym, Hdet], c_opL = spl.makeBlochOperators(gsEnergies, esEnergies, osc_strengths=osc,
                        T1_opt=T1_opt_sym, br_ratio=br_ratio_sym, decay_rates=decay_rates, coh_decay_rates=coh_decay_rates)
        Hlst = [[DetSym, Hdet]]
        if esLabels is None:
            if gsLabels is not None:
                esLabels = ['e:'+st[1:] for st in gsLabels[:len(esEnergies)]] # Assuming there are fewer excited states...
            else:
                esLabels = [r'$|e'+str(num)+'>$' for num in np.arange(len(esEnergies))]
        if gsLabels is None:
            gsLabels = [r'$|g'+str(num)+'>$' for num in np.arange(len(gsEnergies))]

        allLabels = gsLabels + esLabels
        self.params.atoms.Nground = len(gsEnergies)
        self.params.atoms.Nexcited = len(esEnergies)
        self.params.atoms.Nlevels = len(gsEnergies) + len(esEnergies)
        self.params.atoms.stateLabels = allLabels
        self.params.atoms.H0 = H0
        self.params.atoms.H1 = (Esym, 2*H1)
        self.params.atoms.c_opL = c_opL
        self.params.atoms.Hlst = Hlst
        self.params.atoms.br_ratio = [br_ratio_sym, br_ratio]
        self.params.atoms.T1_opt = [T1_opt_sym, T1_opt]
        self.params.atoms.osc_strengths = [osc_strengths_sym, osc_strengths]
        self.params.atoms.subsD = subsD

        self.params.detAx=detAx
        self.params.Ndet=detAx.size
        self.params.atoms.lineShape=lineShape


    def configure_ode(self):
        p=self.params

        # Setup atom equations of motion
        DeltaS, Hdet = p.atoms.Hlst[0]
        EexS, H1a = p.atoms.H1
        H1a = np.matrix(sm.Matrix(np.tril(H1a, -1)).subs(EexS,1)).T.conj() # Making a operator?
        H_desc = [p.atoms.H0, [EexS, H1a], [sm.conjugate(EexS), H1a.T.conj()], (DeltaS, 2*np.pi*Hdet) ]
        eq, expect_vals=spl.makeMESymb(H_desc, c_opL=p.atoms.c_opL, e_opL = [H1a], bReturnMatrixEquation=True)
        p.atoms.smEq = eq
        lhsL, rhsL = spl.seperate_DM_equation(eq.subs(p.atoms.subsD))

        # Decide on type of simulation to make
        if len(self.params.zAx)<1: # Just Bloch, no Maxell (no depth axis)
            ode_s = svp.ODESys(dict(zip(lhsL, rhsL) ),  dims={DeltaS:p.detAx }, driving_syms=[EexS])
            detAx = ode_s.dimAxes[0]
            #ode_s.set_driving({EinS: p.inputs.EinFw})

        else: #1-way, 1D sim (sometimes called "moving frame"
            zS, EinS = sm.symbols("z, E_in")
            rhsL = [el.subs(EexS, EinS+EexS) for el in rhsL]
            ode_s = svp.ODESys(dict(zip(lhsL, rhsL) ),  dims={zS: self.params.zAx, DeltaS:p.detAx }, driving_syms=[EexS])

            polF=sm.lambdify( ode_s.symsD['prop_state_syms'], expect_vals[0] )
            p.atoms.polF = polF
            zAx, detAx = ode_s.dimAxes[:2]
            dDelt = detAx[1] - detAx[0]
            dz = zAx[1] - zAx[0] if len(zAx)>1 else 1
            effectiveLineShape = p.atoms.lineShape/p.length*dz
            p.atoms.effectiveLineShape = effectiveLineShape

            def calcEintF(t, dimAxs, state,  driving, lineShape=effectiveLineShape):
                """ Field due to polarisation of atoms """
                P = -1j*np.cumsum( (polF(*state)*lineShape).sum(axis=1), axis=0)#*mode
                return P.reshape(dimAxs[0].shape)
            # For 2-way will need one of these for each direction
            ode_s.set_state_dependence({EinS:calcEintF})



        self.ode_s = ode_s
        return ode_s
        # Up to here could be cached

    def calculate(self, tSteps, max_step_size=0.1, bDoPrecalc=True):
        """ Put it together and integrate """
        #Setup
        if bDoPrecalc:
            ode_s = self.configure_ode()
        else:
            ode_s = self.ode_s
        ode_s.set_initial_conditions(self.params.initial.all*(1 + 0.0j))
        ode_s.set_driving([self.params.inputs.EinFw])
        model = ode_s.setup_model(max_step = max_step_size)
        #Integrate now
        #print("start integrating")
        res = model.integrate(tSteps)
        #return res
        outA = np.array(res)
        self.makeOutputs(tSteps, outA)
        return outA#, ode_s

        #dcoh_dt, ddia_dt, calc_polF, recons_rho, init_state, Ndia, Ncoh = makeEvolutionFuncs(p.atoms.H0, p.atoms.H1,  p.atoms.Hlst, p.atoms.c_opL, subsD=p.atoms.subsD, lamModules='numpy')
        #OLD 
        if continueFromT is not None:
            if continueFromT == -1:
                ind=-1
            else:
                ind = self.res.t.searchsorted(continueFromT)
            self.setInitialState(simOutput=Bunch(diag=self.res.popT[ind], coh=self.res.P[ind]))
        par0 = p.initial.all #Flatten just before starting calc

    def makeOutputs(self, tSteps, state_output):
        p = self.params
        if 1:
            # CALCULATE OUTPUTS
            #P,w,pop=self._unpack(np.array(outDatL))
            #outA=np.array(outL)
            res = Bunch()
            res.P = state_output[:,p.atoms.Nlevels:]
            res.popT = state_output[:,:p.atoms.Nlevels]
            res.z = self.params.zAx
            res.t = tSteps
            res.det = self.params.detAx

            #drvA = self.calcOutputFields(tSteps,outA, ode_s.tDepFD.items(), [polF])
        if len(self.params.zAx)>0: # Just Bloch, no Maxell (no depth axis)
            polF = self.params.atoms.polF
            res.fields= Bunch()
            drvA = self.calcOutputFields(tSteps, state_output, [[None, p.inputs.EinFw]], [polF])
            #Eintern, EoutFw, EoutRv, EintFw, EintRv=self.calcOutputFields(tSteps, p.inputs.EinFw, p.inputs.EinRv, Pout_tz*p.dz, self.excArrs, kL=2*pi*self.params.length)
            res.fields.EinpFw = np.array([p.inputs.EinFw(t) for t in tSteps])
            res.fields.EinpRv = np.array([p.inputs.EinRv(t) for t in tSteps])
            res.fields.EoutFw = drvA[:,0]# calculate this here
            res.fields.EoutRv = np.zeros(tSteps.size, 'c16')
            res.fields.EintFw = np.zeros(tSteps.size, 'c16')
            res.fields.EintRv = np.zeros(tSteps.size, 'c16')
            res.fields.t=tSteps
        self.res = res

    def calcOutputFields(self, tSteps,stateL, driving, polF):
        outL =[]
        for t, state in zip(tSteps, stateL):# For each time step
            l = []
            for (sym, F), pF in zip(driving, polF): #For each polarisation function. This is not doing any spatial stuff.
                val = F(t) +  -1j*np.sum(np.sum(pF(*state)*self.params.atoms.effectiveLineShape, axis=1), axis=0)
                l.append(val)
            outL.append(l)
        # This has only calculated the internal field so far, I think.
        return np.array(outL)

        #OLD
        # Make a concrete array. This should work for whatever type of function EinFwF/RvF is
        EinFw=np.array([EinFwF(t) for t in tAx])
        EinRv=np.array([EinRvF(t) for t in tAx])
        #EinFw=np.vectorize(EinFwF)(tAx)
        #EinRv=np.vectorize(EinRvF)(tAx)

        EpolFw=1*np.sum(Pave*fwMode.conj(), axis=1)
        EpolRv=1*np.sum((Pave*rvMode.conj())[:,::-1], axis=1 )

        return EinFw + EpolFw


    def showSummary(self):
        import pandas as pd
        # Get energy splittings, labels etc
        atomP = self.params.atoms
        stateLabels=atomP.stateLabels
        if atomP.Nground:
            energies = np.diag(atomP.H0)/2/np.pi
            gsEnergies = energies[:atomP.Nground]
            esEnergies = energies[atomP.Nground:]
            _plot_energy_levels(np.real(gsEnergies),np.real(esEnergies), show_ylabels=True, stateLabels=stateLabels);

        def showPrettyTable(mat, title,topLeft=None):
            df= pd.DataFrame(mat, index=stateLabels, columns=stateLabels)
            display(spl.formatComplexDF(spl.smDataFrame(df), title=title, topLeft=topLeft, prec=1))
        # Make a pandas dataframe
        showPrettyTable(atomP.H0/(np.pi*2), title=r"$H_0/(2\pi)$")
        showPrettyTable(atomP.H1[1], title=r"$H_\mathrm{int}$")
        showPrettyTable(atomP.Hlst[0][1], title=r"$H_\Delta$")
        showPrettyTable(sum(atomP.c_opL), title=r"Decay rates", topLeft="TO&darr; | FR &rarr;")

#Plotting funcs
import pylab as pl
def showInputsOutputsI(fields):
    t=fields.t
    pl.figure(figsize=(9,7))
    ax1=pl.subplot(211)
    ax1.set_title('forwards')
    ax1.plot(t, abs(fields.EinpFw)**2, label='Iin')
    ax1.plot(t, abs(fields.EoutFw)**2, label='Iout')
    ax1.set_xlabel('t')

    ax2=pl.subplot(212)
    ax2.set_title('backwards')
    ax2.plot(t, abs(fields.EinpRv)**2, label='Iin')
    ax2.plot(t, abs(fields.EoutRv)**2, label='Iout')
    ax2.set_xlabel('t')
    ax2.legend()

    #Make y limits the same on both
    lim1=ax1.get_ylim()
    lim2=ax2.get_ylim()
    tot_lim=[min(lim1[0], lim2[0]), max(lim1[1], lim2[1])]
    ax1.set_ylim(tot_lim)
    ax2.set_ylim(tot_lim)

def showInputsOutputs(fields, ylm=None, xlm=None):
    t=fields.t
    pl.figure(figsize=(9,7))
    ax1=pl.subplot(211)
    ax1.set_title('forwards')
    ax1.plot(t,fields.EinpFw.real, label='Re(Ein)')
    ax1.plot(t,fields.EinpFw.imag, label='Im(Ein)')
    ax1.plot(t,fields.EoutFw.real, label='Re(Eout)')
    ax1.plot(t,fields.EoutFw.imag, label='Im(Eout)')
    ax1.set_xlabel('t')


    ax2=pl.subplot(212)
    ax2.set_title('backwards')
    ax2.plot(t,fields.EinpRv.real, label='Re(Ein)')
    ax2.plot(t,fields.EinpRv.imag, label='Im(Ein)')
    ax2.plot(t,fields.EoutRv.real, label='Re(Eout)')
    ax2.plot(t,fields.EoutRv.imag, label='Im(Eout)')
    ax2.set_xlabel('t')
    ax2.legend()

    #Make y limits the same on both
    if ylm is None:
        lim1=ax1.get_ylim()
        lim2=ax2.get_ylim()
        tot_lim=[min(lim1[0], lim2[0]), max(lim1[1], lim2[1])]
    else:
        tot_lim=ylm
    ax1.set_ylim(tot_lim)
    ax2.set_ylim(tot_lim)

    if xlm is not None:
        ax1.set_xlim(xlm)
        ax2.set_xlim(xlm)

def showMeanPol(res, transI=0):
    pl.figure(figsize=(7,7))
    pl.title('$|P|$')
    pl.imshow(abs(res.P[:,transI].mean(axis=-1)), extent=[res.z[0], res.z[-1], res.t[-1], res.t[0]], aspect='auto')
    pl.xlabel('z')
    pl.ylabel('t')

def showFieldAbs(res):
    pl.figure(figsize=(7,7))
    pl.title('$|E|$')
    pl.imshow(abs(res.fields.Eintern), extent=[res.z[0], res.z[-1], res.t[-1], res.t[0]], aspect='auto')
    pl.xlabel('z')
    pl.ylabel('t')

def show_pop_vs_t(res, zSlc=None, whichStates=None):
    stateSlc = np.arange(res.popT.shape[1]) if whichStates is None else whichStates
    if zSlc is None:
        zSlc = np.arange(res.popT.shape[2])
    else:
        try:
            zSlc = res.z.searchsorted(zSlc)
        except Exception():
            pass; # Hopefully it's just a valid slice object?
    pop = res.popT.mean(axis=-1) #average over detuning axis
    pop = pop[:,stateSlc]
    pop = pop[..., zSlc].real
    if len(pop.shape)>2:
        pop= pop.mean(axis=-1)
    pl.figure()
    pl.plot(res.t, pop)
    pl.xlabel('$t(\mu$s)')
    pl.ylabel('pop')
    pl.legend(stateSlc)


def show_pop_delt_vs_t(res, z=np.inf, whichState=0):
    if z is np.inf:
        zSlc=res.z.size-1
    else:
        zSlc=res.z.searchsorted(z)
    pl.figure()
    pl.imshow(res.popT[:,whichState, zSlc].real, aspect='auto', extent=[res.det[0], res.det[-1], res.t[-1], res.t[0]], cmap='inferno')
    pl.colorbar()
    pl.title(r'$\rho_{{{0}, {0}}}$ at z={1}'.format(whichState, res.z[zSlc]))
    pl.xlabel('$\Delta (MHz)$')
    pl.ylabel('$t (\mu s)$')

def show_pop_delt_vs_z(res, t=np.inf, whichState=0):
    if t is np.inf:
        tSlc=res.t.size-1
    else:
        tSlc=res.t.searchsorted(t)
    pl.figure()
    pl.imshow(res.popT[tSlc, whichState].real, aspect='auto', extent=[res.det[0], res.det[-1], res.z[-1], res.z[0]], cmap='inferno')
    pl.colorbar()
    pl.title('w at t={}'.format(res.t[tSlc]))
    pl.xlabel('$\Delta (MHz)$')
    pl.ylabel('$z (\lambda\mathrm{s})$')

def show_absorprtion_spectrum(fields, InDir='Fw',OutDir='Fw', flm=None, ylm=None):
    t=fields.t
    Ein = fields.EinpFw
    Ein = fields['Einp'+InDir]
    Eout = fields['Eout'+OutDir]
    fax = pl.fftfreq(t.size, t[1]-t[0])
    Yin = pl.fft(Ein)
    Yout = pl.fft(Eout)
    A = Yin/Yout

    pl.figure(figsize=(9,7))
    ax1=pl.subplot(211)
    ax1.set_title('absorption')
    #ax1.semilogy(fax, fields.EinpFw.real, label='Re(Ein)')
    ax1.plot(fax, 10*log10(abs(A)**2), label='fw')
    ax1.set_xlabel(r'$\Delta$')
    ax1.set_ylabel('dB')
    ax1.set_xlim(flm)
    ax1.set_ylim(ylm)


    ax2=pl.subplot(212)
    ax2.set_title('trans')
    ax2.plot(fax, abs(Yin), label='Ein')
    ax2.plot(fax, abs(Yout), label='Eout')
    ax2.set_xlabel('fax')
    ax2.legend()
    ax2.set_xlim(flm)

def _plot_energy_levels(gsEnergies, esEnergies, stateLabels=None, fig=None, ax=None, figsize=(3.5,6), show_ylabels=False):
    Ngs = len(gsEnergies)
    Nes = len(esEnergies)

    gsRange = max(gsEnergies) - min(gsEnergies)
    esRange = max(esEnergies) - min (esEnergies)
    totRange = gsRange + esRange
    optSplit = 1.5*totRange

    if not fig and not ax:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    xticks = []
    yticks = []
    yticksEs = []
    x = 0
    for e_idx, e in enumerate(gsEnergies):
        ax.plot([x, x + 2], np.array([1, 1]) * e, 'b', linewidth=2)
        yticks.append(e)
    for e_idx, e in enumerate(esEnergies):
        ax.plot([x, x + 2], np.array([1, 1]) * e + optSplit, 'b', linewidth=2)
        yticksEs.append(e)
    xticks.append(x + 1)
    ax.set_frame_on(False)
    ax.set_xlim([-0,2.1])

    if show_ylabels:
        ax.axes.get_yaxis().set_visible(False)
        yticks = np.unique(np.around(yticks, 1))
        #ax.set_yticks(yticks)
        for tck in yticks:
            ax.annotate("{:.1f}".format(tck), (x+2.05, tck-0.02) )
        for tck in yticksEs:
            ax.annotate("{:.1f}".format(tck), (x+2.05, tck+optSplit-0.02) )
    else:

        ax.axes.get_yaxis().set_visible(False)

    if stateLabels:
        ax.set_xlim([-0.4,2.1])
        ax.axes.get_yaxis().set_visible(False)
        yticks = np.unique(np.around(yticks, 1))
        #ax.set_yticks(yticks)
        for tck,lab in zip(yticks, stateLabels[:len(gsEnergies)]):
            yOffs = tck
            ax.annotate(lab, (x-0.4, tck-0.02) )
        for tck,lab in zip(yticksEs, stateLabels[len(gsEnergies):]):
            yOffs = tck
            ax.annotate(lab, (x-0.4, tck+optSplit-0.02) )
    else:

        ax.axes.get_yaxis().set_visible(False)

    #if labels and 0:
    #    ax.get_xaxis().tick_bottom()
    #    ax.set_xticks(xticks)
    #    ax.set_xticklabels(labels, fontsize=16)
    #else:
    #    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

    ax.set_title('Energy Levels')


if __name__=="__main__":
    from pylab import *
    if 0:
        bw=6.0
        shft=0
        detAx = np.linspace(-bw/2,bw/2, 30)+ shft
        simObj = MBE_1D_SVEA(detAx=detAx, cavityParams={}, length_in_lambdas=1, pts_per_lambda=4)
        p=simObj.params
        zAx=p.zAx
        P0=zeros(zAx.size,dtype='c16')[:,newaxis]
        #w0=2*(MT.square_wave(zAx, 2, duty_cycle=0.9)[:,newaxis] - (0.5 +0j))
        t=linspace(0,5,500)
        EinRv=1.0*ones(t.shape)
        EinFw=0.0*ones(t.shape)
        #EinFw = MT.gauss(t,[1,0.3,0.1])
        #lineShape = 1*MT.gauss(detAx, [1, 0.0,5]) #Inh broadening
        lineShape = 0.1*ones(detAx.shape)
        simObj.setInputs(EinFw=EinFw, EinRv=EinRv, tAx=t)
        simObj.setAtomParams(lineShape=lineShape, gsEnergies=arange(1)*1, esEnergies=arange(1)*0.0, br_ratio=0.5, T1_opt=3)#0.003);
        #pop0=1*[1]+3*[0]
        pop0 = [1.0,0.0]#,0.0]
        simObj.setInitialState(pop0=pop0)
        #lineF=lambda delt: 5.0
        simObj.setCavityParams(R1=0,R2=0.0)
        simObj.excArrs.rvMode*=0
        tAx=linspace(0,8,600)
        pOut, cOut, mOut=simObj.calculate(tAx, max_step_size=0.01)


        #Side-band scan
        #t=linspace(0,10,500)
        #E = MT.chirp(t, 0,30,10)*exp(1j*2*pi*0*t)


    def hole_burn():
        bw=0.7
        shft=0
        detAx = np.linspace(-bw/2+shft,bw/2+shft,20)
        simObj = MBE_1D_SVEA(detAx=detAx, cavityParams={}, length_in_lambdas=10.0, pts_per_lambda=5)
        p=simObj.params
        zAx=p.zAx
        P0=zeros(zAx.size,dtype='c16')[:,newaxis]
        #w0=2*(MT.square_wave(zAx, 2, duty_cycle=0.9)[:,newaxis] - (0.5 +0j))
        pop0=[0.5,0.5,0,0]
        t=linspace(0,10,500)
        EinRv=0*ones(t.shape)
        EinFw=0.5*ones(t.shape)
        #EinFw=zeros(t.shape,dtype='c16')
        lineShape = MT.gauss(detAx, [1, 0.5,0]) #Inh broadening
        simObj.setInputs(EinFw=EinFw, EinRv=EinRv, tAx=t)
        simObj.setAtomParams(lineShape=lineShape, gsEnergies=[0,1], esEnergies=[0,1.5], br_ratio=0.1, T1_opt=1)#0.003);
        simObj.setInitialState(pop0=pop0)
        #lineF=lambda delt: 5.0
        simObj.setCavityParams(R1=0,R2=0.0)
        output, modeOutput=simObj.calculate(linspace(0,10,500))
        return pOut, mOut
        s=simObj
        r=s.res

        #Pout_tz=(Pout*p.atoms.lineShape*r.popT).sum(axis=-1)/p.Ndet/p.length
        Pout_tz=(Pout*p.atoms.lineShape*r.popT/p.length).mean(axis=-1)
        #Efw, Erv,EoutFw, EoutRv=simObj.calcOutputFields(s.tSteps, p.inputs.EinFw, p.inputs.EinRv, Pout_tz*s.params.dz, s.fwMode, s.rvMode)
        #res=simObj.calcOutputFields(s.tSteps, p.inputs.EinFw, p.inputs.EinRv, Pout_tz*p.dz, s.excArrs)
        #Eintern, EoutFw, EoutRv, EintFw, EintRv=res.Eintern, res.EoutFw, res.EoutRv, res.EintFw, res.EintRv
        figure(figsize=(12,10))
        alp=0.6
        lw=3
        plot(s.tSteps,abs(r.EoutFw)**2+0.0005**2, label='EoutFw', lw=lw,alpha=alp)
        plot(s.tSteps,abs(r.EoutRv)**2+0.001**2, label='EoutRv', lw=lw, alpha=alp)
        plot(t, abs(r.EinFw)**2, label='EinFw', alpha=alp,lw=lw)
        plot(t, abs(r.EinRv)**2, label='EinRv', alpha=alp,lw=lw)
        legend()
        if 0:
            figure()
            plot(abs(r.EoutRv[:,0]) )
            plot(abs(r.EoutRv[:,-1]) )


        figure(figsize=(12,10))
        imshow(abs(r.P[:,:].mean(axis=-1)))
        colorbar()
        title('P')

        figure(figsize=(12,10))
        imshow((r.w[:,:].mean(axis=-1).real))
        colorbar()
        title('w')

        figure(figsize=(12,10))
        imshow((r.popT[:,:].mean(axis=-1).real))
        colorbar()
        title('pop')
        show()
        #data=simObj.getOutputs()
        return simObj, r


    #def test_abs():
    if 1:
        bw=0.0
        shft=0
        detAx = np.linspace(-bw/2,bw/2, 1)+ shft
        simObj = MBE_1D_SVEA(detAx=detAx, cavityParams={}, length_in_lambdas=50, pts_per_lambda=6)
        p=simObj.params
        zAx=p.zAx
        P0=zeros(zAx.size,dtype='c16')[:,newaxis]
        #w0=2*(MT.square_wave(zAx, 2, duty_cycle=0.9)[:,newaxis] - (0.5 +0j))
        t=linspace(0,5,1000)
        EinRv=0.0*ones(t.shape)
        #EinFw=0.10*ones(t.shape)
        EinFw = MT.gauss(t,[1,0.3,0.1])
        #lineShape = 1*MT.gauss(detAx, [1, 0.0,5]) #Inh broadening
        lineShape = 5*ones(detAx.shape)


        simObj.setInputs(EinFw=EinFw, EinRv=EinRv, tAx=t)
        simObj.setAtomParams(lineShape=lineShape, gsEnergies=1*arange(2)-0.5, esEnergies=1*arange(2)-0.5, br_ratio=0.5, T1_opt=0.05)#0.003);
        #pop0=1*[1]+3*[0]
        pop0 = [1.0,0.0,0.0,0.0]
        simObj.setInitialState(pop0=pop0)
        #lineF=lambda delt: 5.0
        simObj.setCavityParams(R1=0.0,R2=0.)
        #simObj.excArrs.rvMode*=0
        tAx=linspace(0,3,600)
        pOut, cOut, mOut=simObj.calculate(tAx, max_step_size=0.01)
        mOut = array(mOut)
        pOut = array(pOut)
        cOut = array(cOut)

        #plot(tAx, mOut)

            #data=simObj.getOutputs()
    #freqTest()
    #s,r=hole_burn()
