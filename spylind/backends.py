import numpy as np
import sympy as sm
from scipy import integrate
import pdb
try:
    import tensorflow as tf
    import tensorflow_probability as tfp
    from .tfp_fixed_step import FixedStep

    tfp.math.ode.FixedStep = FixedStep
    TF_DTYPE= 'float64'
    tf.keras.backend.set_floatx(TF_DTYPE) # Presumably can get more speed, especially on a GPU, 
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
    def __init__(self, d_dt, state_at_t0, output_func, state_shape=None, name =None, method='adams',atol = 1e-12, rtol=1e-6, max_step=0.1, **kwargs):
        """this is usually called by setup in ODESys"""

        if name is None:
            name = 'vode'
            if np.iscomplexobj(state_at_t0):
                name = 'zvode'
        self.d_dtF_orig = d_dt

        self._online_process_func =  output_func
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
        #pdb.set_trace()
        d_dt_ = self.d_dtF_orig(t, state)
        return self._flatten(d_dt_)

    def integrate(self, tSteps, initial_state=None, paramD={}, **kwargs):
        if initial_state is None:
            initial_state = self.initial_state
        I = self.scipy_integrator 
        I.set_initial_value(self._flatten(initial_state), tSteps[0])
        # Should look at tSteps and reset if we're asking to restart
        outputL = []
        if tSteps[0]==I.t: # Don't try and evolve to t==0. 
            tSteps= tSteps[1:]
            outputL.append(self._online_process_func(initial_state))
            #self._online_process_func(cur_state_flat)

        # Integrate
        for k, tNext in enumerate(tSteps):
            if not I.successful():
                self.lastGoodInd = k - 1
                print("Integration failed at sim time {}".format(tNext))
                break
            cur_state = self._unflatten( I.integrate(tNext) )
            # probably reshaping, calculating fields etc
            outputL.append(self._online_process_func(cur_state))
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
    lambdify_modules = 'numpy'
    def __init__(self, t_sym, dim_syms, dimAxes, evoD, t_dep_FD, state_dep_FD= {},  bForceStateDimensions = False, dtype=np.complex128):
        state_dep_syms = list(state_dep_FD.keys())
        t_dep_syms = list(t_dep_FD.keys())
        indep_syms = t_dep_syms + state_dep_syms
        state_syms = list(evoD.keys())
        dim_shape = tuple([len(ax) for ax in dimAxes])
        state_shape = tuple([len(evoD), *dim_shape])

        input_syms = [t_sym] + dim_syms + \
            state_syms + t_dep_syms + state_dep_syms

        d_dtF = sm.lambdify(input_syms, list(evoD.values()), modules= self.lambdify_modules)
        if bForceStateDimensions: #Useful if the included functions don't always return a correctly shaped output
            d_dtF0 = d_dtF
            d_dtF = lambda *args: [np.broadcast_to(out, dim_shape)
                                    for out in d_dtF0(*args)]
        
        #needed for actual evolution
        self.d_dtF = d_dtF
        self.d_dt_current = np.ascontiguousarray(   
                            np.zeros(state_shape, dtype= dtype) )
        self.dimAxes = [ax.reshape(*(k * [1] + [ax.size] + (len(dimAxes) - k - 1) * [1]))
                                for k, ax in enumerate(dimAxes)]
        self.state_dep_f = list(state_dep_FD.values())
        self.t_dep_f = list(t_dep_FD.values())
        
        #For later inspection
        self.evoD = evoD

    def _calc_driving_vals(self, t):
        driving_vals = [f(t) for f in self.t_dep_f]
        return driving_vals

    def _calc_state_dep_vals(self, t, state, driving_vals):
        state_dep_vals = [f(t, self.dimAxes, state, driving_vals)
                            for f in self.state_dep_f]
        return state_dep_vals
        
    def __call__(self, t, state):
        driving_vals = self._calc_driving_vals(t)
        state_dep_vals = self._calc_state_dep_vals(t, state, driving_vals)
        self.d_dt_current[:] = self.d_dtF(t, *self.dimAxes, *state, *driving_vals, *state_dep_vals)
        return self.d_dt_current

        
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
        return tf.reshape(state, (-1,) ) # .reshape(-1)

    def _unflatten_state(self, state):
        return tf.reshape(state, self.state_shape)

