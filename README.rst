=======
Spylind
=======

.. image:: https://img.shields.io/travis/morgatron/spylind.svg
        :target: https://travis-ci.org/morgatron/spylind

.. image:: https://img.shields.io/pypi/v/spylind.svg
        :target: https://pypi.python.org/pypi/spylind


Simple definition and relatively fast solution of large ensembles of quantum systems, each described by a Lindblad master equation. This was mainly written for the classs of problems relevant to Rare-earth ion ensembles in solids, where each density matrix represents a class of tationary ions, and the driving fields (optical, RF) are large enough to be considered classical.

There is a strong overlap with qutip's 'mesolve', and the use of qutips `Qobj` objects to define operators is supported. Qutip tends to work badly when attempting to describe large ensmebls. For small systems however, spylind will be somewhat slower.

Most 'docs', if htey exist. will be in the example notebooks.


spylind and spyIVP
-----------------

The main thing this spylind does is give expressions for the evolution of density matrix elements, i.e. d_rho/d_t, based on some Hamiltonian specification. The result is a dictionary of sympy symbols with corresponding sympy expressions representing their derivatives.

To actually solve these usefully, we need to stick them into some kind of ODE solver. Practically this, means taking the equations of motion defined symbollically, turning them into a python function that takes inputs and returns derivatives, and finally integrating them using something like scipy's odeint. While straightforward enough, this can be clunky and error prone.

For that reason, spyIVP was written. It's mostly just a bridge between a method to create the python function (using sympy's 'lambdify') and sicpy's 'odeint' for the sctual solving. It takes care of making sure the arrays are the right shape etc, should make adding extra prolbem dimensions fairly easy.

spyIVP _should_ be really simple. It's become a bit bloated though, because it's trying to be a bit backend agnostic. In particular, to support using Tensorflow for increased speed and scalability.


pyMBE
-----
spylind etc are written to be the base of a more flexible version of pyMBE. That's still a goal (see MBE.py), but not done yet.


Future
--------
WHen This was started, there didn't seem to be a good solution for the actual solving, which is why spyIVP was written. Since then, the Julia language's differential equations package (DifferentialEquqations.jl) has grown and looks fantastic. On brief appearances, it may well solve several of the issues for which spyIVP was made (easy solutions of multi-dimensional equations, automatic differentiation, works on the GPU)


* Free software: 3-clause BSD license
* Documentation: (Should end up at...) https://morgatron.github.io/spylind.

Features
--------
* Incomplete!
* TODO
* Major todo: solidify interface with Tensorflow and finding of gradients.
