# CVsim
main Author: Yorick Bleiji
contributions by: Bram Boogaarts

This code has been developed to simulate a cyclic voltammetry process for metal deposition.

The module can be imported in python (Jupyter notebook) by using the following code:

from CVsim import CVsim

The class CVsim is imported as CVsim. See below for an example on how to use the class:

CV = CVsim(nu,E0,E1,E2,Ef=Ef, ncycles=ncyces)
CV.parameters(c0,Ef0,D,k0)
CV.electrode('1D')
CV.run()
CV.plot()

For more information about the input arguments use help( ... ).

Work in progress..
