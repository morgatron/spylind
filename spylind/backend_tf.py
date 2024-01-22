import tensorflow as tf
import tensorflow_probability as tfp
from .tfp_fixed_step import FixedStep

tfp.math.ode.FixedStep = FixedStep
TF_DTYPE= 'float64'
tf.keras.backend.set_floatx(TF_DTYPE) # Presumably can get more speed, especially on a GPU, 


class ModelTensorflow(Model):# Should really be called ModelScipy

    outputL = []
    def __init__(self, d_dt, state_at_t0, output_func, state_shape=None, name =None, method='adams',atol = 1e-12, rtol=1e-6, max_step=0.1, input_modifiers={}, **kwargs):
        """this is usually called by setup in ODESys"""

        self.d_dtF_orig = d_dt
        self.initial_state = state_at_t0
        self._calc_output_user =  output_func

        if 0:
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

        #results_obj =tfp.math.ode.FixedStep(**kwargs).solve(self.d_dtF_orig.__call__, tSteps[0], initial_state, solution_times=tSteps)
        results_obj =tfp.math.ode.DormandPrince(**kwargs).solve(self.d_dtF_orig.__call__, tSteps[0], initial_state, solution_times=tSteps)
        return results_obj

        #return np.array(outputL)

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
    lambdify_modules = ['tensorflow', {'conjugate': tf.math.conj}]
    TF_DTYPE = 'float64'

    def __init__(self, t_sym, dim_syms, dimAxes, evoD, t_dep_FD, intermediate_exprs = [], state_dep_FD={}, constantsD = {}, bForceStateDimensions = False, dtype=tf.float64):
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
        d_dtF = tf.function(d_dtF, experimental_compile=False)

        intermediate_syms =  [t_sym] + dim_syms + \
            state_syms + t_dep_syms + constant_syms
        self.intermediate_calc_F = sm.lambdify(intermediate_syms, intermediate_exprs, modules= self.lambdify_modules)

        #needed for actual evolution
        self.d_dtF = d_dtF
        self.d_dt_current = np.ascontiguousarray(   
                            np.zeros(state_shape, dtype= dtype) )
        self.dimAxes = [ax.reshape(*(k * [1] + [ax.size] + (len(dimAxes) - k - 1) * [1]))
                                for k, ax in enumerate(dimAxes)]
        self.dimAxes = [tf.convert_to_tensor(ax, dtype = TF_DTYPE) for ax in self.dimAxes]
        self.state_dep_f = list(state_dep_FD.values()) #should probably be tuples... maybe even named ones
        self.t_dep_f = list(t_dep_FD.values())
        self.constant_vals = list(constantsD.values())
        
        #For later inspection
        self.evoD = evoD

        self.state_shape = state_shape
        self.dim_shape = dim_shape

        if 0:
            rhs = list(evoD.values())

            dim_shape = tuple([len(ax) for ax in dimAxes])
            state_shape = tuple([len(eqD), *dim_shape])

            input_syms = [tSym] + dimSyms + \
                state_syms + t_dep_syms + state_dep_syms
            self.eq = Munch(lhs=input_syms, rhs=rhs)

            self.d_dt_lam = d_dt_lam
            self.preShaped_dimAxes = [dAx.reshape(*(k * [1] + [dAx.size] + (len(dimAxes) - k - 1) * [1]))
                                    for k, dAx in enumerate(dimAxes)]
            self.preShaped_dimAxes = [tf.convert_to_tensor(ax, dtype = self.TF_DTYPE) for ax in self.preShaped_dimAxes]
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
    def _d_dt(self, t, state, driving_vals, state_dep_vals, constant_vals):
        resL = self.d_dtF(t, *self.dimAxes, *tf.unstack(state), *tf.unstack(driving_vals), *tf.unstack(state_dep_vals) , *tf.unstack(constant_vals))
        resL = [tf.broadcast_to(tf.cast(res, TF_DTYPE), self.dim_shape) for res in resL]
        return tf.stack(resL);

    #@tf.function
    def _calc_driving_vals(self, t, **driving_params):
        if not driving_params:
            return tf.stack([f(t) for f in self.t_dep_f ]) 
        else:
            return tf.stack([f(t, *driving_params[str(sym) ]) for sym,f in self.t_dep_FD.items() ]) 

    #@tf.function
    def _calc_state_dep_vals(self, t, state, driving_vals, intermediate_vals=[], constants=[]):
        return tf.stack([f(t, self.preShaped_dimAxes, state, driving_vals) for f in self.state_dep_f ]) 
        
    @tf.function
    def __call__(self, t, cur_state_flat, **driving_params):
        if 'training' in driving_params:
            driving_params.pop('training')
        state = self._unflatten_state(cur_state_flat)
        driving_vals, intermediate_vals, state_dep_vals = tf.unstack(self._calc_all_requirements(t,state))
        #self.d_dt_current[:] 
        result = self._d_dt(t, state, driving_vals, state_dep_vals, self.constant_vals)
        #
        #driving_vals = self._calc_driving_vals(t, **driving_params)
        #state_dep_vals = self._calc_state_dep_vals(t, state, driving_vals)
        #result =self._d_dt(t, state, driving_vals, state_dep_vals)
        #return cur_state_flat
        return self._flatten_state(result)
        
    def _calc_intermediate_vals(self, t, state, driving_vals):
        '''Calculate all the bits required by the evolution function
        '''
        intermediate_vals  = self.intermediate_calc_F(t, *tf.unstack(self.dimAxes), *tf.unstack(state), *tf.unstack(driving_vals), *tf.unstack(self.constant_vals))
        return tf.stack(intermediate_vals)

    def _calc_all_requirements(self, t, state):
        driving_vals = self._calc_driving_vals(t)
        intermediate_vals = self._calc_intermediate_vals(t, state, driving_vals)
        state_dep_vals = self._calc_state_dep_vals(t, state, driving_vals, intermediate_vals)
        return driving_vals, intermediate_vals, state_dep_vals

    def _calc_outputs(self,t,state):
        driving_vals, intermediate_vals, state_dep_vals = self._calc_all_requirements(t,state)
        
    def _flatten_state(self, state): 
        return tf.reshape(state, (-1,) ) # .reshape(-1)

    def _unflatten_state(self, state):
        return tf.reshape(state, self.state_shape)
