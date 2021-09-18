=======
Spylind
=======

.. image:: https://img.shields.io/travis/morgatron/spylind.svg
        :target: https://travis-ci.org/morgatron/spylind

.. image:: https://img.shields.io/pypi/v/spylind.svg
        :target: https://pypi.python.org/pypi/spylind


Simple definition and fast solution of large ensembles of quantum systems, each described by a Lindblad master equation. This was mainly written for the classs of problems relevant to Rare-earth ion ensembles in solids, where a _lot_ of ions are stationary, and the optical field is large.

There is a strong overlap with qutip's 'mesolve' however, and should work better than qutip for other large ensemble type problems. For small systems however, spylind will be signficantly slower, but much faster for large systems.

Most 'docs' are in the example notebooks.


spylind and spyIVP
-----------------

In the first instance, all spylind does is give expressions for the evolution of density matrix elements, based on some Hamiltonian specification. The result is a dictionary of sympy symbols with corresponding sympy expressions representing their derivatives.

To actually solve these usefully, we need to stick them into some kind of ODE solver. So that means taking the equations of motion defined symbollically, turning them into a python function that takes inputs and returns derivatives, then sticking that into a something like scipy's odeint. While straightforward enough, there are a number of gotcha's.

So for that reason, spyIVP was written. It's mostly just a bridge between sympy's 'lambdify' (which takes the symbollic equations and turns them into fast evaluated functions that take numpy arrays) and sicpy's 'odeint'. It's mainly useful to do this job when there problem has several transverse dimensions (e.g. spatial dimensions).

spyIVP _should_ be really simple. It's become a bit bloated though, because it's trying to be a bit backend agnostic. In particular, to support using Tensorflow for increased speed and scalability.


pyMBE
-----
spylind etc are written to be the base of a more flexible version of pyMBE. That's still a goal (see MBE.py), but someway off yet.


Future
--------
This was started before Julia's differential equations package had taken off. It seems likely that package will solve a lot of the probelms addressed here, at least those that spyIVP is trying to address (easy solutions of multi-dimensional equations, automatic differentiation, works on the GPU)




* Free software: 3-clause BSD license
* Documentation: (Should end up at...) https://morgatron.github.io/spylind.

Features
--------
* Incomplete!
* TODO
* Major todo: solidify interface with Tensorflow and finding of gradients.
