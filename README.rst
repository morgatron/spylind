=======
Spylind
=======

Spylind is a Python library for the symbolic definition and high-performance solution of large ensembles of quantum systems described by the Lindblad master equation.

It is particularly optimized for ensembles with many degrees of freedom (e.g., inhomogeneous broadening), utilizing **JAX** and **TensorFlow** for JIT compilation and GPU acceleration.

Core Components
---------------

* **spylind**: Generates symbolic equations of motion (EOM) for density matrix elements from a Hamiltonian and collapse operators.
* **spyIVP**: A high-performance bridge between symbolic expressions and numerical ODE solvers.

Key Features
------------

* **Symbolic-to-Numerical**: Automates the transition from SymPy expressions to optimized solver code.
* **Multiple Backends**:
    * **Diffrax (JAX)**: High-performance, JIT-compiled, and GPU-capable.
    * **NumPy (SciPy)**: Robust, industry-standard CPU integration.
    * **TensorFlow**: scalable integration for large-scale problems.
* **Multi-dimensional Ensembles**: Native support for parameter sweeps and spatial dimensions.
* **QuTiP Integration**: Supports QuTiP ``Qobj`` for operator definitions.

Installation
------------

.. code-block:: bash

    pip install spylind[jax]  # Recommended for performance
    pip install spylind[tf]   # For TensorFlow support
    pip install spylind[full] # All backends

Quick Example
-------------

.. code-block:: python

    from spylind import spylind as spl
    import qutip as q
    import numpy as np
    import sympy as sm

    # Define a 2-level system symbolically
    H = [0.1 * np.pi * q.sigmaz(), [sm.symbols('Omega')/2, q.sigmax()]]
    tlist = np.linspace(0, 1.0, 101)

    # Solve using the Diffrax backend
    res = spl.mesolve(H, q.basis(2,0), tlist,
                      t_dep_fL={'Omega': lambda t: 2*np.pi},
                      e_ops=[q.sigmaz()],
                      backend='diffrax')

* License: 3-clause BSD
