# Spylind

Spylind is a Python library for symbolically defining and solving ensembles of quantum systems described by the Lindblad master equation, with a particular focus on rare-earth ion ensembles in solids.

It also includes **spyIVP**, a utility to bridge multi-dimensional SymPy expressions with numerical initial-value problem (ODE) solvers. This was designed for use in interactive simulation scripts and Jupyter notebooks.

> **Note:** Spylind is research software. APIs and documentation are continuing to evolve.

## Core Components

* **`spylind.spylind`**: Formulates symbolic equations of motion for density matrix elements from a given Hamiltonian and collapse operators (with support for QuTiP `Qobj` operators).
* **`spylind.spyIVP`**: Translates multi-dimensional SymPy differential equations into numerical ODE solver models, handling parameter distributions, driving functions, and ensemble dimensions.

## Solvers & Backends

* **CyRK**: Runge-Kutta ODE solving via Cython and compiled Numba C-callbacks (`CyRK.nbsolve2_ivp`), with automatic fallback to `pysolve_ivp`.
* **Diffrax (JAX)**: ODE integration with JAX-based JIT compilation.
* **NumPy / SciPy**: Standard CPU integration using `scipy.integrate`.

## Installation

Install in development mode:
```bash
pip install -e .
```

To install optional backends:
```bash
# CyRK backend
pip install -e .[cyrk]

# Diffrax (JAX) backend
pip install -e .[jax]

# Both backends
pip install -e ".[cyrk,jax]"
```

## Quick Examples

### 1. Quantum Master Equation (`spylind.spylind`)

```python
from spylind import spylind as spl
import qutip as q
import numpy as np
import sympy as sm

# Define a 2-level system symbolically
H = [0.1 * np.pi * q.sigmaz(), [sm.symbols('Omega') / 2, q.sigmax()]]
tlist = np.linspace(0, 1.0, 101)

# Solve using CyRK, Diffrax, or NumPy
res = spl.mesolve(
    H,
    q.basis(2, 0),
    tlist,
    t_dep_fL={'Omega': lambda t: 2 * np.pi},
    e_ops=[q.sigmaz()],
    backend='cyrk'  # or 'diffrax', 'numpy'
)
```

### 2. General ODE Systems & Ensembles (`spylind.spyIVP`)

```python
import numpy as np
import sympy as sm
from spylind import spyIVP as so

# Define symbolic variables
x, v = sm.symbols("x, v", real=True)
omega = sm.symbols("omega", real=True)      # Ensemble parameter (e.g. distributed frequencies)
gamma = sm.symbols("gamma", real=True)      # Damping parameter

# Symbolic equations of motion
eqs = {
    x: v,
    v: -(omega**2) * x - gamma * v
}

# Define an ensemble over a parameter distribution
omega_vals = np.linspace(0.8, 1.2, 50)
ode_sys = so.ODESys(
    eqs,
    trans_dims={omega: omega_vals},
    parameters={gamma: 0.1}
)
ode_sys.set_initial_state({x: 1.0, v: 0.0})

# Setup solver model ('cyrk', 'diffrax', or 'numpy')
model = ode_sys.setup_model(backend='cyrk')

# Integrate: returns array of shape (n_times, n_variables, n_ensemble)
t_steps = np.linspace(0, 20.0, 200)
res = model.integrate(t_steps)
```

## License

BSD 3-Clause License.
