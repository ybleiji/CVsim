# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:23:11 2020

@author: Yorick Bleiji
With contributions by: Bram Boogaarts

This code is written to simulate a cyclic voltammogram (CV)
"""

import numpy as np
import scipy.constants as const
const.F = const.physical_constants["Faraday constant"][0]


class CV_sim:
    """
    This class defines the cyclic voltammogram. A cyclic voltammagram has an Eint, E1, E2 and a scanrate.
    The units of the potentials are V fand the units of scan rate is in mV/s
    """
    def __init__(self, scanrate, E0, E1, E2, Ef=None, ncycles=1):
        '''
        Initialize the CV. Give the scanrate, the starting, return and stop potentials.
            scanrate: scanrate in mV/s (float)    
            E0: starting potential in V (float)
            E1: return potential in V (float)
            E2: return potential in V (float)
            
        Optional:
            Ef: end potential in V (float), when not specified it is equal to E0
            ncycles: number of cycles (int) (default = 1)
            
        '''
        self.__dir__ = dir(self)
        self.E0 = E0 # V
        self.E1 = E1 # V
        self.E2 = E2 # V
        self.scanrate = scanrate * 1e-3 # mV/s to V/s
        self.ncycles = ncycles
        if Ef is None: self.Ef = E0 # V
        else: self.Ef = Ef # V
    
    
    def parameters(self,c0, Ef0, D, k0, alpha=0.5, T_C=21):
        '''
        Specify using parameters the concentration, thermodynamic and kinetic parameters:
            c0: initial concentraion in mM (float)
            Ef0: formal potential in V (float)
            D: Diffusion constant in cm^2/s (float)
            k0: electrochemical rate constant in cm/s (float)
            
        Optional:
            alpha: transfercoefficient (float)
            T_C: temperature in deg C (float)
        '''
        self.c0 = c0 * 1e-6 # from mM to mol/cm^3
        self.Ef0 = Ef0 # in V
        self.D = D # in cm^2/s
        self.k0 = k0 # in cm/s
        self.T_C = T_C # in deg C
        self.alpha = alpha


    def ECrateconst(self, E,Ef0,k0,T=21,alpha=0.5):
        '''
        This function returns the electrochemical rate constant k in cm/s.
        k = k0 exp(-alpha*F/(RT) * (E-Ef0))
        
        Input parameters:
            E: applied potential in V (float or np.array)
            Ef0: formal standard reduction potential in V (float)
            k0: standard rate constant in cm/s (float)
        Optional:
            T: temperature in deg C (float) (default=21 deg C)
            alpha: transfercoefficient (float) (default=0.5)
        
        Output:
            k: electrochemical rate in cm/s (same type as E)     
        '''
        T = T + 273.15 # temp from C to K
        k = k0*np.exp(-alpha*const.F/(const.R*T)*(E-Ef0))
        return k
    
    
    def E_app(self, E0,E1,E2,scanrate, Ef=None, ncycles=1, E_steps=100, dE=None):
        '''
        This function creates a np.array containing the potentials.
        
        Input paramters:
            E0: start potential in V (float)
            E1: return potentential 1 in V (float)
            E2: return potentential 2 in V (float)
            scanrate: scanrate in V/s (float)
            
        Optional:
            Ef: final potential in V (float)
            ncycles: number of cycles (int) (default = 1)
            E_steps: number of steps to devide E into (int) (default = 100)
            dE: difference between E steps in V (default = None)
            
        Output:
            Eapp: np.array containing all potentials in V
            time: np.array containing the time profile in s
        '''
        Eapp = [] # create empty Eapp
        if E1 != E2:    # chcek if E2 is same as E1: if yes, it is just a linear sweep
            if dE is None:
                dE = (max(E0,E1,E2,Ef)-min(E0,E1,E2,Ef))/E_steps # determine dE by taking E_steps points per half a cycle
            sign = np.sign(E1-E0) # sign of dE, first down or up in potential
            for n in range(ncycles):
                Eapp = np.append(Eapp, np.arange(E0,E1,sign*dE)) # in V, from E0 to E1
                Eapp = np.append(Eapp, np.arange(E1,E2,-sign*dE)) # in V, return scan
                if n == ncycles-1 and Ef == E2:
                    # stop at E2, only add endpoint
                    Eapp = np.append(Eapp, E2)
                elif n == ncycles-1:
                    # in V, go back to Ef. Also include endpoint
                    Eapp = np.append(Eapp, np.arange(E2,Ef+sign*dE,sign*dE))
                    # end of the loop
                else:
                    Eapp = np.append(Eapp, np.arange(E2,E0,sign*dE)) # in V, go back to E0
                    # proceed to next cycle
        else: # linear sweep
            if dE is None: 
                dE = (E1-E0)/E_steps # divide the linear sweep in E_steps points
            Eapp = np.arange(E0,E1+dE,dE) #include also endpoint
            
        # create time array
        dt = abs(dE)/scanrate # difference in time determined by the dE and scanrate
        Time = [i*dt for i in range(len(Eapp))]
    
        return np.array(Eapp), np.array(Time)

    # import electrode functions
    from .electrode import electrode, insulator, electrode_from_image
    
    # import findsurface
    from .findsurface import findsurface

    # import the CV_1D, CV_2D, CV_3D functions
    from .CV_1D import CV_1D
    from .CV_2D import CV_2D
    from .CV_3D import CV_3D
    
    # import run functions
    from .run_simulation import run_simulation as run
    
    # import animate functions
    from .animate import animate, animate_1D, animate_2D, animate_3D
    
    # import plot functions
    from .plot import plot, activity
    
    
    def reset(self):
        '''
        Call this function to reset the parameters.
        '''
        for el in dir(self):
            if el not in self.__dir__:
                delattr(self, el)      
              

