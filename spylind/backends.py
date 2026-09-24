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
        if hasattr(self, 'scipy_integrator'):
            tCur = self.scipy_integrator.t
        elif self.tSteps_last_run is not None:
            tCur = self.tSteps_last_run[-1]
        else:
            raise ValueError("Model has not been integrated yet; cannot continue.")
        if tEnd <= tCur:
            raise ValueError(f"Already integrated past {tEnd} (tCur is {tCur})")
        tSteps = np.arange(tCur + dt, tEnd, dt)
        return self.integrate(tSteps)

    def integrate_to(self, tEnd, dt):
        tSteps = np.arange(0, tEnd, dt)
        return self.integrate(tSteps)


class ModelNumpy(Model):  # Should really be called ModelScipy

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
        self.tSteps_last_run = tSteps
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
        # For later inspection
        self.evoD = evoD
        self.dim_syms = dim_syms
        self.dim_shape = dim_shape
        self.state_shape = state_shape
        self.constantsD = constantsD
        self.constant_syms = constant_syms
        self.t_dep_FD = t_dep_FD
        self.t_dep_syms = t_dep_syms
        self.state_dep_FD = state_dep_FD
        self.state_dep_syms = state_dep_syms
        self.intermediate_exprs = intermediate_exprs

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


class ModelDiffrax(Model):
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

        # Create JIT-compiled solve function once
        solver = self.solver
        stepsize_controller = self.stepsize_controller
        max_steps = self.max_steps
        kwargs_solve = self.kwargs

        @jax.jit
        def _solve(y0, t0, t1, dt0, tSteps, d_dt_fast_local):
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(d_dt_fast_local),
                solver,
                t0=t0,
                t1=t1,
                dt0=dt0,
                y0=y0,
                stepsize_controller=stepsize_controller,
                saveat=diffrax.SaveAt(ts=tSteps),
                max_steps=max_steps,
                **kwargs_solve,
            )
            return sol

        self._solve_jit = _solve

    def integrate(self, tSteps, initial_state=None, params={}, **kwargs):
        if initial_state is None:
            initial_state = self.initial_state
        else:
            initial_state = jnp.array(initial_state)

        if params:
            self.d_dt_fast.set_parameters(params)

        t0 = tSteps[0]
        t1 = tSteps[-1]
        dt0 = (tSteps[1] - tSteps[0]) if len(tSteps) > 1 else 0.1

        sol = self._solve_jit(initial_state, t0, t1, dt0, tSteps, self.d_dt_fast)
        self.tSteps_last_run = tSteps

        # Process outputs efficiently using vmap if possible
        try:
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
        self.max_ram_MB = opts.pop('max_ram_MB', 2048)
        self.bUseFastPath = opts.pop('fast_path', True)
        
        # CyRK doesn't support max_steps/nsteps
        opts.pop('max_steps', None)
        opts.pop('nsteps', None)
        
        self.kwargs = opts
        self.fast_callback_info = None

        if self.bUseFastPath:
            self._try_compile_fast_callback()

    def _unflatten(self, flat_state):
        return flat_state.reshape(*self.state_shape)

    def _try_compile_fast_callback(self):
        """Attempt to compile a pure C-callback via Numba / CyRK cyjit for maximum speed.
        If any aspect of the ODE system is not Numba-compatible (e.g. unjittable custom callables),
        this will cleanly catch the exception and keep self.fast_callback_info = None,
        automatically falling back to pysolve_ivp with zero user friction.
        """
        try:
            import math
            import numba as nb
            from CyRK import cyjit
            from sympy.printing.numpy import NumPyPrinter

            # If state-dependent functions exist, fall back to pysolve_ivp
            if getattr(self.d_dt_fast, 'state_dep_FD', None):
                return

            t_dep_FD = getattr(self.d_dt_fast, 't_dep_FD', {})
            jitted_driving = {}
            for sym, fn in t_dep_FD.items():
                if callable(fn):
                    try:
                        jfn = nb.njit(fn)
                        # Warmup test call with scalar float
                        _ = jfn(0.0)
                        jitted_driving[sym] = jfn
                    except Exception:
                        # Cannot JIT user's driving callable -> fallback
                        return

            state_syms = self.d_dt_fast.state_syms
            dim_syms = getattr(self.d_dt_fast, 'dim_syms', [])
            constant_syms = getattr(self.d_dt_fast, 'constant_syms', [])
            parameter_syms = getattr(self.d_dt_fast, 'parameter_syms', [])
            dim_axes = getattr(self.d_dt_fast, 'dimAxes', [])
            constant_vals = getattr(self.d_dt_fast, 'constant_vals', [])
            parameter_vals = getattr(self.d_dt_fast, 'parameter_vals', [])
            intermediate_exprs = getattr(self.d_dt_fast, 'intermediate_exprs', [])

            state_shape = self.state_shape
            n_states = len(state_syms)

            # Build args buffer layout
            args_layout = []
            curr_offset = 0

            for d_sym, d_arr in zip(dim_syms, dim_axes):
                arr = np.asarray(d_arr, dtype=np.float64)
                args_layout.append(('dim', d_sym, curr_offset, arr.size, arr.shape, arr))
                curr_offset += arr.size

            for c_sym, c_val in zip(constant_syms, constant_vals):
                if c_val is not None:
                    arr = np.asarray(c_val, dtype=np.float64)
                    args_layout.append(('const', c_sym, curr_offset, arr.size, arr.shape, arr))
                    curr_offset += arr.size

            for p_sym, p_val in zip(parameter_syms, parameter_vals):
                args_layout.append(('param', p_sym, curr_offset, 1, (), float(p_val)))
                curr_offset += 1

            total_args_size = max(curr_offset, 1)
            base_args_buf = np.zeros(total_args_size, dtype=np.float64)
            for kind, sym, offset, size, shape, val in args_layout:
                if kind in ('dim', 'const'):
                    base_args_buf[offset:offset+size] = np.ravel(val)
                elif kind == 'param':
                    base_args_buf[offset] = float(val)

            # Build substitution map to safe python identifier names
            sub_map = {}
            for i, s in enumerate(state_syms):
                sub_map[s] = sm.Symbol(f'_s_{i}')
            for i, d in enumerate(dim_syms):
                sub_map[d] = sm.Symbol(f'_d_{i}')
            for i, c in enumerate(constant_syms):
                sub_map[c] = sm.Symbol(f'_c_{i}')
            for i, p in enumerate(parameter_syms):
                sub_map[p] = sm.Symbol(f'_p_{i}')
            for i, drv in enumerate(t_dep_FD.keys()):
                sub_map[drv] = sm.Symbol(f'_drv_{i}')
            if getattr(self.d_dt_fast, 't_sym', None):
                sub_map[self.d_dt_fast.t_sym] = sm.Symbol('t')

            printer = NumPyPrinter()
            lines = [
                'def _fast_cyrk_diffeq(dy_ptr, t, y_ptr, args_ptr):',
                f'    y = nb.carray(y_ptr, {state_shape})',
                f'    dy = nb.carray(dy_ptr, {state_shape})',
                f'    args = nb.carray(args_ptr, ({total_args_size},))',
            ]

            # Unpack state
            for i in range(n_states):
                lines.append(f'    _s_{i} = y[{i}]')

            # Unpack args
            d_idx = 0
            c_idx = 0
            p_idx = 0
            for kind, sym, offset, size, shape, _ in args_layout:
                if kind == 'dim':
                    if shape == ():
                        lines.append(f'    _d_{d_idx} = args[{offset}]')
                    else:
                        lines.append(f'    _d_{d_idx} = args[{offset}:{offset+size}].reshape({shape})')
                    d_idx += 1
                elif kind == 'const':
                    if shape == () or size == 1:
                        lines.append(f'    _c_{c_idx} = args[{offset}]')
                    else:
                        lines.append(f'    _c_{c_idx} = args[{offset}:{offset+size}].reshape({shape})')
                    c_idx += 1
                elif kind == 'param':
                    lines.append(f'    _p_{p_idx} = args[{offset}]')
                    p_idx += 1

            # Driving
            for i, drv in enumerate(t_dep_FD.keys()):
                lines.append(f'    _drv_{i} = _jitted_drv_{i}(t)')

            # Intermediate expressions
            for i, expr in enumerate(intermediate_exprs):
                sub_expr = expr.subs(sub_map)
                lines.append(f'    _inter_{i} = {printer.doprint(sub_expr)}')

            # Derivatives
            for i, s in enumerate(state_syms):
                subbed_rhs = self.d_dt_fast.evoD[s].subs(sub_map)
                rhs_code = printer.doprint(subbed_rhs)
                lines.append(f'    dy[{i}] = {rhs_code}')

            fn_code = '\n'.join(lines)
            local_ns = {'nb': nb, 'numpy': np, 'math': math}
            for i, drv in enumerate(t_dep_FD.keys()):
                local_ns[f'_jitted_drv_{i}'] = jitted_driving[drv]

            exec(fn_code, local_ns)
            compiled_cfunc = cyjit(local_ns['_fast_cyrk_diffeq'])

            self.fast_callback_info = {
                'address': compiled_cfunc.address,
                'args_buf': base_args_buf,
                'args_layout': args_layout,
                'compiled_fn': compiled_cfunc,
            }
        except Exception as e:
            # Fall back to pysolve_ivp
            self.fast_callback_info = None

    def set_parameters(self, paramsD):
        """Update runtime parameter values in both d_dt_fast and the fast args buffer."""
        self.d_dt_fast.set_parameters(paramsD)
        if self.fast_callback_info:
            args_buf = self.fast_callback_info['args_buf']
            for kind, sym, offset, size, shape, _ in self.fast_callback_info['args_layout']:
                if kind == 'param':
                    for psym, val in paramsD.items():
                        if psym == sym or str(psym) == str(sym):
                            args_buf[offset] = float(val)
                            break

    def integrate(self, tSteps, initial_state=None, params={}, **kwargs):
        if initial_state is None:
            y_current = self.initial_state
        else:
            y_current = initial_state.reshape(-1)

        if params:
            self.set_parameters(params)

        unflatten = self._unflatten
        d_dt_fast = self.d_dt_fast

        if len(tSteps) <= 1:
            self.tSteps_last_run = tSteps
            return np.array([self.output_func(tSteps[0], unflatten(y_current))])

        # 1. Fast Path via nbsolve2_ivp C-callback if available
        if self.fast_callback_info is not None:
            try:
                from CyRK import nbsolve2_ivp
                addr = self.fast_callback_info['address']
                args_buf = self.fast_callback_info['args_buf']

                res = nbsolve2_ivp(
                    addr,
                    (tSteps[0], tSteps[-1]),
                    y_current,
                    t_eval=tSteps,
                    args=args_buf,
                    method=self.method,
                    atol=self.atol,
                    rtol=self.rtol,
                    max_step=self.max_step,
                    dense_output=False,
                    max_ram_MB=self.max_ram_MB,
                    **self.kwargs
                )
                if not res.success:
                    print(f"CyRK fast-path integration failed: {res.message}")

                self.tSteps_last_run = tSteps
                outputL = [self.output_func(tSteps[i], unflatten(res.y[:, i])) for i in range(len(tSteps))]
                return np.array(outputL)
            except Exception as e:
                # If nbsolve2_ivp fails for any reason, fall back to pysolve_ivp below
                pass

        # 2. Robust Fallback via pysolve_ivp Python bridge
        def rhs_wrapped(t, y_flat):
            y = unflatten(y_flat)
            dy = d_dt_fast(t, y)
            return dy.reshape(-1).astype(np.float64)

        res = CyRK.pysolve_ivp(
            rhs_wrapped,
            (tSteps[0], tSteps[-1]),
            y_current,
            t_eval=tSteps,
            method=self.method,
            atol=self.atol,
            rtol=self.rtol,
            max_step=self.max_step,
            **self.kwargs
        )

        if not res.success:
            print(f"CyRK integration failed: {res.message}")

        self.tSteps_last_run = tSteps
        outputL = [self.output_func(tSteps[i], unflatten(res.y[:, i])) for i in range(len(tSteps))]
        return np.array(outputL)
