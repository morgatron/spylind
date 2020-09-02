=====
Overview
=====
spylind -> solving ensembles of Lindblad master equations, which can be defined in a semi-symbollic form.

This is mainly written as the basis for quickly and easily solving Maxwell-Bloch equations for rare-earth ion ensembles. It's relativey general though, and has a few useful features.

The major place it's likely useful is where a large ensemble of similar equations need to be evolved simultaneously, potentially with coupling. The Maxwell-Bloch equations are one such prblem- a large ensemble of near identical atoms interacting classically via the electromagnetic field.

There are two modules in spylind:

spyIVP.py contains the basic general purpose solver
spylind.py wraps the solver in spyIVP specifically to solve master equations. In particular, it exposes spylind.mesolve, which is in some ways a drop-in replacement for qutip's mesolve, but suitable for solving a large system of weakly coupled objects, each of which is described by a master equation.

