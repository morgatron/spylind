import numpy as np
import sympy as sm
from scipy import integrate
import pdb
try:
    from backend_tf import ModelTensorflow, D_Dt_Fast_TF 
except ModuleNotFoundError as e:
    print("Tensorflow not accessible: {}".format(str(e)))


# These objects are supposed to represent the packaged up object to actually do the solution. It takes parameters, and returns output,, with minimum cruft. They're supposed to be made by the ODESys object.

# This object shouldn't know anything about the problem dimension. It will often however have access to a function which converts from the model's result space into whatever output is required. This function may be called 'online', ie during the calcuation, or may be used to process the results as a block once it's finished.

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


class ModelNumpy(Model):# Should really be called ModelScipy

    outputL = []
    def __init__(self, d_dt, state_at_t0, output_func, state_shape=None, name =None, method='adams',atol = 1e-12, rtol=1e-6, max_step=0.1, input_modifiers={}, **kwargs):
        """this is usually called by setup in ODESys"""

        if name is None:
            name = 'vode'
            if np.iscomplexobj(state_at_t0):
                name = 'zvode'
        self.d_dtF_orig = d_dt

        self._calc_output_user =  output_func
        self.initial_state = state_at_t0
        int_obj = integrate.ode(self.d_dtF_wrapped)
        #if self.bDecompose_to_re_im or self.default_dtype not in (np.complex64, np.complex128):
        int_obj.set_integrator(name, max_step=max_step, method = method,
                             **kwargs)  # ,order=5) # or adams
        self.scipy_integrator = int_obj

        if state_shape is None:
            state_shape = state_at_t0.shape
        self.state_shape = state_shape
        
    def _flatten(self, state):
        return state.reshape(-1)

    def _unflatten(self, flat_state):
        return flat_state.reshape(*self.state_shape)

    def d_dtF_wrapped(self, t, state_flat):
        state = self._unflatten(state_flat)
        d_dt_ = self.d_dtF_orig(t, state)
        return self._flatten(d_dt_)

    def calc_output(self, t,state):
        #state = self._unflatten(state_flat)
        driving_vals, intermediate_vals, state_dep_vals = self.d_dtF_orig._calc_all_requirements(t,state)
        return self._calc_output_user(t,state, driving_vals, intermediate_vals, state_dep_vals )

    def integrate(self, tSteps, initial_state=None, paramD={}, **kwargs):
        if initial_state is None:
            initial_state = self.initial_state
        I = self.scipy_integrator 
        I.set_initial_value(self._flatten(initial_state), tSteps[0])
        # Should look at tSteps and reset if we're asking to restart
        outputL = []
        if tSteps[0]==I.t: # Don't try and evolve to t==0. 
            outputL.append(self.calc_output(tSteps[0], initial_state))
            tSteps= tSteps[1:]
            #outputL.append(self._online_process_func(initial_state))
            #self._online_process_func(cur_state_flat)

        # Integrate
        for k, tNext in enumerate(tSteps):
            if not I.successful():
                self.lastGoodInd = k - 1
                print("Integration failed at sim time {}".format(tNext))
                break
            cur_state = self._unflatten( I.integrate(tNext) )
            # probably reshaping, calculating fields etc

            outputL.append(self.calc_output(tNext, cur_state))
        return np.array(outputL)


class D_Dt_Fast_numpy:
    """ 
    Make a function that takes t, state as input and calculates d_state_dt, which
    also depends on driving and state dependence.
    attributes:
    * dimAxes
    * Npars
    * d_dt
    """
    lambdify_modules = ['numpy', 'scipy']
    def __init__(self, t_sym, dim_syms, dimAxes, evoD, t_dep_FD, intermediate_exprs= [], state_dep_FD= {},  constantsD= {}, bForceStateDimensions = False, dtype=np.complex128):
        state_dep_syms = list(state_dep_FD.keys())
        t_dep_syms = list(t_dep_FD.keys())
        indep_syms = t_dep_syms + state_dep_syms
        state_syms = list(evoD.keys())
        constant_syms = list(constantsD.keys())
        dim_shape = tuple([len(ax) for ax in dimAxes])
        state_shape = tuple([len(evoD), *dim_shape])

        input_syms = [t_sym] + dim_syms + \
            state_syms + t_dep_syms + state_dep_syms +constant_syms
        d_dtF = sm.lambdify(input_syms, list(evoD.values()), modules= self.lambdify_modules)

        intermediate_syms =  [t_sym] + dim_syms + \
            state_syms + t_dep_syms + constant_syms
        self.intermediate_calc_F = sm.lambdify(intermediate_syms, intermediate_exprs, modules= self.lambdify_modules)

        if bForceStateDimensions: #Sometimes useful if the included functions don't always return a correctly shaped output
            d_dtF0 = d_dtF
            d_dtF = lambda *args: [np.broadcast_to(out, dim_shape)
                                    for out in d_dtF0(*args)]
        
        #needed for actual evolution
        self.d_dtF = d_dtF
        self.d_dt_current = np.ascontiguousarray(   
                            np.zeros(state_shape, dtype= dtype) )
        self.dimAxes = [ax.reshape(*( (k+0) * [1] + [ax.size] + (len(dimAxes) - k - 1) * [1]))
                                for k, ax in enumerate(dimAxes)]
        self.state_dep_f = list(state_dep_FD.values()) #should probably be tuples... maybe even named ones
        self.t_dep_f = list(t_dep_FD.values())
        self.constant_vals = list(constantsD.values())
        
        #For later inspection
        self.evoD = evoD

    def _calc_driving_vals(self, t):
        driving_vals = [f(t) for f in self.t_dep_f]
        return driving_vals

    def _calc_state_dep_vals(self, t, state, driving_vals, intermediate_vals=[], constants=[]):
        state_dep_vals = [f(t, self.dimAxes, state, driving_vals, intermediate_vals, self.constant_vals)
                            for f in self.state_dep_f]
        return state_dep_vals

    def _calc_intermediate_vals(self, t, state, driving_vals):
        '''Calculate all the bits required by the evolution function
        '''
        intermediate_vals  = self.intermediate_calc_F(t, *self.dimAxes, *state, *driving_vals, *self.constant_vals)
        return intermediate_vals

    def _calc_all_requirements(self, t, state):
        driving_vals = self._calc_driving_vals(t)
        intermediate_vals = self._calc_intermediate_vals(t, state, driving_vals)
        state_dep_vals = self._calc_state_dep_vals(t, state, driving_vals, intermediate_vals)
        return driving_vals, intermediate_vals, state_dep_vals

    def _calc_outputs(self,t,state):
        driving_vals, intermediate_vals, state_dep_vals = self._calc_all_requirements(t,state)
        
    def __call__(self, t, state):
        driving_vals, intermediate_vals, state_dep_vals = self._calc_all_requirements(t,state)
        self.d_dt_current[:] = self.d_dtF(t, *self.dimAxes, *state, *driving_vals, *state_dep_vals, *self.constant_vals)
        return self.d_dt_current


