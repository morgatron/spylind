import numpy as np
import sympy as sm
from scipy import integrate
import pdb

try:
    import jax
    import jax.numpy as jnp
    import diffrax
except (ImportError, ModuleNotFoundError):
    jax = None
    diffrax = None

try:
    import CyRK
except (ImportError, ModuleNotFoundError):
    CyRK = None


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
        tSteps = np.arange(tCur + dt, tEnd, dt)  # tSteps_last_run[-1]
        return self.integrate(tSteps)

    def integrate_to(self, tEnd, dt):
        tSteps = np.arange(0, tEnd, dt)
        return self.integrate(tSteps)


class ModelNumpy(Model):  # Should really be called ModelScipy
    outputL = []

    def __init__(
        self,
        d_dt,
        state_at_t0,
        output_func,
        state_shape=None,
        solver_opts=None,
        **kwargs,
    ):
        """this is usually called by setup in ODESys"""
        opts = dict(solver_opts) if solver_opts else {}

        name = opts.pop('name', None)
        if name is None:
            name = "vode"
            if np.iscomplexobj(state_at_t0):
                name = "zvode"

        method = opts.pop('method', 'adams')
        atol = opts.pop('atol', 1e-10)
        rtol = opts.pop('rtol', 1e-8)
        max_step = opts.pop('max_step', 0.1)

        # Support both nsteps (scipy) and max_steps (diffrax) as aliases
        nsteps = opts.pop('nsteps', opts.pop('max_steps', 500))

        self.d_dtF_orig = d_dt
        self._calc_output_user = output_func
        self.initial_state = state_at_t0

        int_obj = integrate.ode(self.d_dtF_wrapped)
        int_obj.set_integrator(
            name, max_step=max_step, method=method, atol=atol, rtol=rtol, nsteps=nsteps, **opts
        )

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

    def calc_output(self, t, state):
        # state = self._unflatten(state_flat)
        return self._calc_output_user(t, state)
        # driving_vals, intermediate_vals, state_dep_vals = self.d_dtF_orig._calc_all_requirements(t,state)
        # return self._calc_output_user(t, *self.d_dtF_orig.dimAxes, *state, *driving_vals, *state_dep_vals, *self.d_dtF_orig.constant_vals)

    def integrate(self, tSteps, initial_state=None, params={}, **kwargs):
        if initial_state is None:
            initial_state = self.initial_state
        
        if params:
            self.d_dtF_orig.set_parameters(params)

        I = self.scipy_integrator
        I.set_initial_value(self._flatten(initial_state), tSteps[0])
        # Should look at tSteps and reset if we're asking to restart
        outputL = []
        cur_state = initial_state
        # Integration
        for k, tNext in enumerate(tSteps):
            if tNext == I.t:
                outputL.append(self.calc_output(tNext, cur_state))
                continue

            if not I.successful():
                self.lastGoodInd = k - 1
                print("Integration failed at sim time {}".format(tNext))
                break
            cur_state = self._unflatten(I.integrate(tNext))
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

    lambdify_modules = ["numpy", "scipy", {"adjoint": np.conj, "conjugate": np.conj}]

    def __init__(
        self,
        t_sym,
        dim_syms,
        dimAxes,
        evoD,
        t_dep_FD,
        intermediate_exprs=[],
        state_dep_FD={},
        output_exprD={},
        constantsD={},
        parameter_syms=[],
        parameter_vals=[],
        bForceStateDimensions=False,
        dtype=np.complex128,
    ):
        state_dep_syms = list(state_dep_FD.keys())
        t_dep_syms = list(t_dep_FD.keys())
        indep_syms = t_dep_syms + state_dep_syms
        state_syms = list(evoD.keys())
        constant_syms = list(constantsD.keys())
        dim_shape = tuple([len(ax) for ax in dimAxes])
        state_shape = tuple([len(evoD), *dim_shape])

        input_syms = (
            [t_sym]
            + dim_syms
            + state_syms
            + t_dep_syms
            + state_dep_syms
            + constant_syms
            + parameter_syms
        )
        d_dtF = sm.lambdify(
            input_syms, list(evoD.values()), modules=self.lambdify_modules
        )

        intermediate_syms = [t_sym] + dim_syms + state_syms + t_dep_syms + constant_syms + parameter_syms
        self.intermediate_calc_F = sm.lambdify(
            intermediate_syms, intermediate_exprs, modules=self.lambdify_modules
        )

        if bForceStateDimensions:  # Sometimes useful if the included functions don't always return a correctly shaped output
            d_dtF0 = d_dtF
            d_dtF = lambda *args: [
                np.broadcast_to(out, dim_shape) for out in d_dtF0(*args)
            ]

        # needed for actual evolution
        self.d_dtF = d_dtF
        self.d_dt_current = np.ascontiguousarray(np.zeros(state_shape, dtype=dtype))
        self.dimAxes = [
            ax.reshape(*((k + 0) * [1] + [ax.size] + (len(dimAxes) - k - 1) * [1]))
            for k, ax in enumerate(dimAxes)
        ]
        self.state_dep_f = list(
            state_dep_FD.values()
        )  # should probably be tuples... maybe even named ones
        self.t_dep_f = list(t_dep_FD.values())
        self.constant_vals = list(constantsD.values())
        self.parameter_vals = list(parameter_vals)
        self.parameter_syms = parameter_syms
        self.t_sym = t_sym
        self.state_syms = state_syms
        self.input_syms = input_syms
        self.intermediate_syms = intermediate_syms
        # For later inspection
        self.evoD = evoD

    def set_parameters(self, paramsD):
        """Update runtime parameter values."""
        # Map input dict to the correct order in parameter_vals
        for sym, val in paramsD.items():
            found = False
            for i, psym in enumerate(self.parameter_syms):
                if psym == sym or str(psym) == str(sym):
                    self.parameter_vals[i] = val
                    found = True
                    break
            if not found:
                print(f"Warning: parameter {sym} not found in model.")

    def getLambdified(self, exprD):
        l_func = sm.lambdify(
            self.input_syms, list(exprD.values()), modules=self.lambdify_modules
        )
        
        def wrapped_lambda(t, state, *args):
            driving_vals, intermediate_vals, state_dep_vals = self._calc_all_requirements(t, state)
            input_args = ([t] + self.dimAxes + list(state) + driving_vals + 
                          state_dep_vals + self.constant_vals + self.parameter_vals)
            return l_func(*input_args)

        return wrapped_lambda

    def _calc_driving_vals(self, t):
        driving_vals = [f(t) for f in self.t_dep_f]
        return driving_vals

    def _calc_state_dep_vals(
        self, t, state, driving_vals, intermediate_vals=[], constants=[]
    ):
        state_dep_vals = [
            f(
                t,
                self.dimAxes,
                state,
                driving_vals,
                intermediate_vals,
                self.constant_vals,
            )
            for f in self.state_dep_f
        ]
        return state_dep_vals

    def _calc_intermediate_vals(self, t, state, driving_vals):
        """Calculate all the bits required by the evolution function"""
        intermediate_vals = self.intermediate_calc_F(
            t, *self.dimAxes, *state, *driving_vals, *self.constant_vals, *self.parameter_vals
        )
        return intermediate_vals

    def _calc_all_requirements(self, t, state):
        driving_vals = self._calc_driving_vals(t)
        intermediate_vals = self._calc_intermediate_vals(t, state, driving_vals)
        state_dep_vals = self._calc_state_dep_vals(
            t, state, driving_vals, intermediate_vals
        )
        return driving_vals, intermediate_vals, state_dep_vals

    # def _calc_outputs(self,t,state):
    #    driving_vals, intermediate_vatls, state_dep_vals = self._calc_all_requirements(t,state)
    #    return self.calc_output_F(t, *self.dimAxes, *state, *driving_vals, *state_dep_vals, *self.constant_vals)

    def __call__(self, t, state):
        driving_vals, intermediate_vals, state_dep_vals = self._calc_all_requirements(
            t, state
        )
        res = self.d_dtF(
            t,
            *self.dimAxes,
            *state,
            *driving_vals,
            *state_dep_vals,
            *self.constant_vals,
            *self.parameter_vals
        )
        for i, val in enumerate(res):
            self.d_dt_current[i] = val
        return self.d_dt_current


class D_Dt_Fast_Diffrax(D_Dt_Fast_numpy):
    """JAX/Diffrax compatible ODE RHS."""

    lambdify_modules = ["jax", {"adjoint": jnp.conj, "conjugate": jnp.conj}]

    def __init__(self, *args, **kwargs):
        if jax is None:
            raise ImportError("JAX/Diffrax not installed.")
        super().__init__(*args, **kwargs)
        # Convert constants to jax arrays
        self.constant_vals = [jnp.array(v) for v in self.constant_vals]
        self.dimAxes = [jnp.array(ax) for ax in self.dimAxes]
        self.dim_shape = tuple([len(ax) for ax in self.dimAxes])

        # Optimize for JAX by avoiding unpacking large state vectors into individual tracers.
        # We group the symbols in the lambdify signature.
        # This requires re-lambdifying with a grouped signature.

        # We need to extract the original symbols to regroup them
        # Note: these were stored in self during super().__init__
        self.grouped_input_syms = [
            self.t_sym,
            tuple(self.input_syms[1:1+len(self.dimAxes)]), # dim_syms
            tuple(self.state_syms),
            tuple([s for s in self.input_syms if str(s).startswith('driving_') or s in self.t_dep_f]), # approximation of driving
        ]
        # Actually, let's just use the known structure from the constructor
        # Re-derive grouped syms safely
        t_sym = self.t_sym
        dim_syms = self.input_syms[1:1+len(self.dimAxes)]
        n_dim = len(dim_syms)
        n_state = len(self.state_syms)
        state_syms = self.input_syms[1+n_dim : 1+n_dim+n_state]

        # The remaining are driving, state_dep, and constants
        # We'll just group them for now to reduce argument count
        other_syms = self.input_syms[1+n_dim+n_state:]

        self.grouped_input_syms = [t_sym, tuple(dim_syms), tuple(state_syms), tuple(other_syms)]

        self.d_dtF_grouped = sm.lambdify(
            self.grouped_input_syms, list(self.evoD.values()), modules=self.lambdify_modules
        )

    def _tree_flatten(self):
        children = (
            tuple(self.constant_vals),
            tuple(self.dimAxes),
            tuple(self.parameter_vals),
        )
        # aux_data must be hashable and contain only static information.
        # Functions and symbolic objects should be handled carefully.
        aux_data = {
            't_sym': self.t_sym,
            'state_syms': self.state_syms,
            'parameter_syms': tuple(self.parameter_syms),
            'd_dtF': self.d_dtF,
            'd_dtF_grouped': getattr(self, 'd_dtF_grouped', None),
            'intermediate_calc_F': self.intermediate_calc_F,
            'state_dep_f': tuple(self.state_dep_f),
            't_dep_f': tuple(self.t_dep_f),
            'dim_shape': self.dim_shape,
            'input_syms': tuple(self.input_syms),
            'grouped_input_syms': tuple(getattr(self, 'grouped_input_syms', [])),
            'evoD': self.evoD,
            'state_dep_funcs_D': getattr(self, 'state_dep_funcs_D', {}),
        }
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        for k, v in aux_data.items():
            setattr(obj, k, v)
        obj.constant_vals = list(children[0])
        obj.dimAxes = list(children[1])
        obj.parameter_vals = list(children[2])
        return obj

    def __call__(self, t, state, args=None):
        # Diffrax signature is (t, y, args)

        # These are still Python loops, but they operate on JAX tracers
        driving_vals = [f(t) for f in self.t_dep_f]

        # We use a grouped call to intermediate_calc_F if possible, 
        # but for now let's just optimize the main d_dtF call
        intermediate_vals = self.intermediate_calc_F(
            t, *self.dimAxes, *state, *driving_vals, *self.constant_vals, *self.parameter_vals
        )

        state_dep_vals = [
            f(
                t,
                self.dimAxes,
                state,
                driving_vals,
                intermediate_vals,
                self.constant_vals,
            )
            for f in self.state_dep_f
        ]

        # Use the grouped function to avoid tracer unpacking explosion
        others = tuple(driving_vals) + tuple(state_dep_vals) + tuple(self.constant_vals) + tuple(self.parameter_vals)
        res = self.d_dtF_grouped(
            t,
            tuple(self.dimAxes),
            tuple(state), # Still unpacking into a tuple, but JAX handles tuples of tracers well
            others
        )

        res = [jnp.broadcast_to(r, self.dim_shape) for r in res]
        return jnp.stack(res)

if jax is not None:
    jax.tree_util.register_pytree_node(
        D_Dt_Fast_Diffrax,
        D_Dt_Fast_Diffrax._tree_flatten,
        D_Dt_Fast_Diffrax._tree_unflatten
    )


class ModelDiffrax:
    def __init__(
        self,
        d_dt_fast,
        initial_state,
        output_func,
        solver_opts=None,
        **kwargs,
    ):
        self.d_dt_fast = d_dt_fast
        self.initial_state = jnp.array(initial_state)
        self.output_func = output_func

        opts = dict(solver_opts) if solver_opts else {}
        self.solver = opts.pop('solver', diffrax.Dopri5())

        # Support both max_steps (diffrax) and nsteps (scipy) as aliases
        self.max_steps = opts.pop('max_steps', opts.pop('nsteps', 100000000))
        # print(f"Max_steps is {self.max_steps}")

        # Check current JAX precision to set sane default tolerances
        is_x64 = jax.config.read("jax_enable_x64")
        default_atol = 1e-9 if is_x64 else 1e-5
        default_rtol = 1e-7 if is_x64 else 1e-4

        atol = opts.pop('atol', default_atol)
        rtol = opts.pop('rtol', default_rtol)

        if not is_x64 and (atol < 1e-6 or rtol < 1e-5):
            print(f"WARNING: JAX x64 is disabled. Using requested tolerances (atol={atol}, rtol={rtol}), but be aware that 32-bit precision might cause solver issues with tight tolerances.")

        max_step = opts.pop('max_step', None)

        self.stepsize_controller = opts.pop('stepsize_controller', None)
        if self.stepsize_controller is None:
            self.stepsize_controller = diffrax.PIDController(
                rtol=rtol, atol=atol, dtmax=max_step
            )
        self.kwargs = opts # Remaining opts go to diffeqsolve


    def integrate(self, tSteps, initial_state=None, params={}, **kwargs):
        if initial_state is None:
            initial_state = self.initial_state
        else:
            initial_state = jnp.array(initial_state)

        if params:
            # For JAX, we need to create a new d_dt_fast with updated parameter_vals
            # since it's a PyTree.
            self.d_dt_fast.set_parameters(params)

        t0 = tSteps[0]
        t1 = tSteps[-1]
        dt0 = (tSteps[1] - tSteps[0]) if len(tSteps) > 1 else 0.1

        saveat = diffrax.SaveAt(ts=tSteps)

        # Diffrax expects a function (t, y, args)
        term = diffrax.ODETerm(self.d_dt_fast)

        # Use a JITted internal function to speed up repeated calls
        @jax.jit
        def solve(y0, t0, t1, dt0, tSteps, d_dt_fast_local):
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(d_dt_fast_local),
                self.solver,
                t0=t0,
                t1=t1,
                dt0=dt0,
                y0=y0,
                stepsize_controller=self.stepsize_controller,
                saveat=diffrax.SaveAt(ts=tSteps),
                max_steps=self.max_steps,
                **self.kwargs,
            )
            return sol

        sol = solve(initial_state, t0, t1, dt0, tSteps, self.d_dt_fast)

        # Process outputs efficiently using vmap if possible
        # We need a function that works on JAX tracers
        try:
            # Try vectorizing the output function
            # self.output_func(t, y)
            output_vmap = jax.vmap(self.output_func)
            res = output_vmap(tSteps, sol.ys)
            return np.array(res)
        except Exception as e:
            # Fallback to Python loop if output_func is not JAX-compatible
            outputL = []
            for i in range(len(tSteps)):
                outputL.append(self.output_func(tSteps[i], sol.ys[i]))
            return np.array(outputL)


class D_Dt_Fast_CyRK(D_Dt_Fast_numpy):
    """CyRK compatible ODE RHS."""
    def __init__(self, *args, **kwargs):
        if CyRK is None:
            raise ImportError("CyRK not installed.")
        super().__init__(*args, **kwargs)

    def __call__(self, t, state):
        # CyRK expects a function of (t, y) where y is 1D
        # If our state is multi-dimensional, it will be flattened by the model wrapper
        # but here we need to ensure we return a 1D array if needed.
        res = super().__call__(t, state)
        return self.d_dt_current.reshape(-1)


class ModelCyRK(Model):
    def __init__(
        self,
        d_dt_fast,
        initial_state,
        output_func,
        solver_opts=None,
        **kwargs,
    ):
        opts = dict(solver_opts) if solver_opts else {}
        self.d_dt_fast = d_dt_fast
        
        self.state_shape = initial_state.shape
        self.initial_state = initial_state.reshape(-1)
        self.output_func = output_func

        self.method = opts.pop('method', 'RK45') # CyRK supports RK45, DOP853 etc
        self.atol = opts.pop('atol', 1e-10)
        self.rtol = opts.pop('rtol', 1e-8)
        self.max_step = opts.pop('max_step', np.inf)
        
        # CyRK doesn't support max_steps/nsteps
        opts.pop('max_steps', None)
        opts.pop('nsteps', None)
        
        self.kwargs = opts

    def _unflatten(self, flat_state):
        return flat_state.reshape(*self.state_shape)

    def integrate(self, tSteps, initial_state=None):
        if initial_state is None:
            y_current = self.initial_state
        else:
            y_current = initial_state.reshape(-1)

        unflatten = self._unflatten
        d_dt_fast = self.d_dt_fast

        def rhs_wrapped(t, y_flat):
            y = unflatten(y_flat)
            dy = d_dt_fast(t, y)
            return dy.reshape(-1).astype(np.float64)

        outputL = []
        # Initial output
        outputL.append(self.output_func(tSteps[0], unflatten(y_current)))

        # Integrate from step to step
        for i in range(len(tSteps) - 1):
            t_start = tSteps[i]
            t_end = tSteps[i+1]
            
            if t_start == t_end:
                outputL.append(self.output_func(t_end, unflatten(y_current)))
                continue

            res = CyRK.pysolve_ivp(
                rhs_wrapped,
                (t_start, t_end),
                y_current,
                t_eval=np.array([t_end]), # Only evaluate at the end point, must be array
                method=self.method,
                atol=self.atol,
                rtol=self.rtol,
                max_step=self.max_step,
                **self.kwargs
            )
            
            if not res.success:
                print(f"CyRK integration failed at t={t_end}: {res.message}")
                break
                
            y_current = res.y[:, -1]
            outputL.append(self.output_func(t_end, unflatten(y_current)))

        return np.array(outputL)
