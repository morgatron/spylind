# Spylind Project Overview


Spylind is nominally a Python library for the simple definition and fast solution of large ensembles of quantum systems, each described by a Lindblad master equation. It is particularly focused on Rare-earth ion ensembles in solids.

The project uses **Sympy** to derive symbolic equations of motion for density matrix elements and provides bridges to numerical solvers using **Scipy** (Numpy backend) or **Tensorflow** for high-performance integration.

The project also includes spyIVP which is a way to quickly solve initial value problems defined by multi-dimensional sympy expressions. This part has turned out to be useful on it's own. It is designed to be used in interactive simulation scripts or notebooks, and includes some diagnostic tools.

## Core Architecture

- **`spylind.spylind`**: High-level API for defining quantum systems.
    - `mesolve`: Mimics QuTiP's `mesolve` interface.
    - `makeMESymb`: Generates symbolic evolution expressions for a given Hamiltonian and collapse operators.
- **`spylind.spyIVP`**: A bridge between symbolic expressions and ODE solvers.
    - `ODESys`: The main class for representing and solving a system of ODEs (Note: historical references to `ODESolver` likely refer to this class).
- **`spylind.backends`**: Implementation of solver backends.
    - `ModelNumpy`: Uses `scipy.integrate.ode`.
    - `ModelDiffrax`: Uses `diffrax` (JAX) for high-performance, XLA-accelerated solving.
    - `ModelCyRK`: Uses `CyRK` for fast, Cython-based integration.
- **`spylind.MBE`**: Maxwell-Bloch Equations implementation.

## Key Technologies

- **Python 3.6+**
- **Sympy**: Symbolic mathematics.
- **Numpy / Scipy**: Numerical integration and array operations.
- **QuTiP**: Quantum Toolbox in Python (supported for operator definitions).
- **Tensorflow**: (Optional) Backend for accelerated solving.

## Development Workflow

### Installation
Install the package in development mode:
```bash
pip install -e .

```
The current local install is managed by a conda environment with name "pmsim". Use this environment for running tests etc.
### Running Tests
Tests are located in `spylind/tests/` and can be run using `pytest`:
```bash
pytest
```

### Building Documentation
Documentation is very very out of date.

### Examples
Examples and usage guides will be made in the  `examples/` directory as Jupyter Notebooks. Only some of these are current.

## Current Project State and Goals
- **This is an old project, with some broken parts** The main current task is to fix up broken parts.
- **Update complex decomposition code**: Make sure the automatic decomposition of complex equations into real/imaginary parts for solvers that only support real values still works.
- **tidy up code in spyIVP:** Focus on clarity and consitency. Make sure to leave diagnostic and display code such as summary() or show_signatures() which help in understanding the physics problem to be solved.
- **Write more tests for spyIVP**
- **Investigate other backends:** In particular, consider if "diffrax" or "numbakit-ode", "pytorch", or "CyRK" may be used.
