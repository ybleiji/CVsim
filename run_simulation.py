# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:48:55 2020

@author: bleiji
"""
from .genfunc import checkkeys


def run_simulation(self, Dm=0.49, MaxIter=1e5, feedback=True, fb=None, fbN=None, gridtype=None, *args,**kwargs):
    '''
    Use run() to run the simulation.
    Optional arguments:
        Dm: diffusion model coefficient (default 0.49), DO NOT EXCEED 0.5!
        MaxIter: maximum amount of iterations (default 1e6)
        gridtype: choose from linear or nonlinear to use a linear or nonlinear grid type (default linear)
        
    Optional arguments for 1D:
        E_steps: number of steps to devide E into (int) (default = 100)
        Lx: length of the diffusion profile: x = [0, Lx] (default = 6*sqrt(D*t_sim))
        x_steps: number of points of the concentration profile in x (default = 300)
        
    Optional arguments for 1D:
        E_steps: number of steps to devide E into (int) (default = 100)
        Lx: length of the diffusion profile: x = [0, Lx] (default = 6*sqrt(D*t_sim))
        Ly: length of the electrode: y = [0, Ly] (default = Lx)
        x_steps: number of points of the concentration profile in x (default = 300)
        y_steps: number of points of the concentration profile in y (default = 300) 
    '''
    #check if the electrode is specified
    try: temp = self.dim
    except AttributeError: raise AttributeError('The dimension of the electrode is not specified! \nUse .electrode() to specify the electrode and its dimension!')
    if self.dim in ['2D','3D']:
        try: temp = self.elec
        except AttributeError: raise AttributeError('The electrode is not specified! \nUse .electrode() to specify the electrode and its dimension!')
    
    # set some paramters
    if Dm > 0.5: Dm = 0.499 # DO NOT EXCEED 0.5
    self.Dm = Dm
    self.MaxIter = MaxIter
    if fb is not None:
        feedback = fb
    self.feedback = feedback
    self.gridtype = gridtype
    if fbN is not None: self.fbN = fbN
    
    #set the amount of simulation points
    self.E_steps = kwargs.get('E_steps', 100) # default amount of E steps is 100

    # check if the nessary parameters are specified
    try:
        self.E0, self.E1, self.E2, self.Ef, self.scanrate, self.ncycles
        self.c0, self.D, self.k0, self.Ef0, self.alpha, self.T_C
    except:
        raise AttributeError('The parameters are not specified! Please specify the parameters using .parameters()')
    
    # select whick simulation to use
    if self.dim == '1D':
        # check the arguments
        allowedkeys = {'E_steps':None,'x_steps':None,'Lx':None,
                       'gridtype':None,'fbN':None}
        checkkeys('CV_1D', kwargs, allowedkeys)
        
        # set gridtype
        if gridtype is None: self.gridtype = 'nonlinear'
        
        # set the amount of x_steps
        if 'x_steps' not in dir(self):
            if self.gridtype == 'linear': self.x_steps = kwargs.get('x_steps', 300) # default amount of x steps is 300
            else: self.x_steps = kwargs.get('x_steps', 100) # default amount of x steps is 100
        
        if self.feedback: print('The 1D simulation is running!')
        self.CV_1D(**kwargs) # simulate!
    if self.dim == '2D':
        # check the arguments
        allowedkeys = {'E_steps':None,'x_steps':None,'Lx':None,
                       'y_steps':None,'Ly':None,'gridtype':None,'fbN':None}
        checkkeys('CV_2D', kwargs, allowedkeys)
        
        # set gridtype
        if gridtype is None: self.gridtype = 'linear'
        
        # set the x and y steps
        if'x_steps' not in dir(self):
            if self.gridtype == 'linear':
                self.x_steps = kwargs.get('x_steps', 300) # default amount of x steps is 300
            else: raise NotImplementedError('Nonlinear grid not yet implemented')
        if 'y_steps' not in dir(self):
            self.y_steps = kwargs.get('y_steps', 100) # default amount of y steps is 100 
        
        if self.feedback: print('The 2D simulation is running!')
        self.CV_2D(**kwargs) # simulate!
    if self.dim == '3D':
        # set gridtype
        if gridtype is None: self.gridtype = 'linear'
        
        # set the x, y and z steps
        if'x_steps' not in dir(self):
            if self.gridtype == 'linear':
                self.x_steps = kwargs.get('x_steps', 10) # default amount of x steps is 10
            else: raise NotImplementedError('Nonlinear grid not yet implemented')
        if 'y_steps' not in dir(self):
            self.y_steps = kwargs.get('y_steps', 300) # default amount of y steps is 300
        if 'z_steps' not in dir(self):
            self.z_steps = kwargs.get('z_steps', 10) # default amount of z steps is 10
        
        if self.feedback: print('The 3D simulation is running!')
        self.CV_3D()
    
    if self.feedback: print('Done!')
    self.runned = True