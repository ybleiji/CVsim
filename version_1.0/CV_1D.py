# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:04:49 2020

@author: bleiji
"""

import numpy as np
import time
import scipy.constants as const
const.F = const.physical_constants["Faraday constant"][0]


def CV_1D(self):
    '''
    This is the main function to simulate a cv.
    We are looking to the following process:
        A^+n + e^- -> B^+n-1
    We are reducing ion A to ion B by transfering 1 electron. 
    We assume that:
        the diffusion to the electrode is in 1D
        the electrode is flat and homogeneous
        the concentration profile is constant at t= 0
        the length of the diffusion profile is max Lx = 6*sqrt(D*t_sim)
        
    Specify the electrode with .electrode('1D').
        
    All the nessarely parameters must be set using __init__() and paramters().
        E_app: the apllied voltage in V (np.array)
        E0: start potential in V (float)
        E1: return potential 1 in V (float)
        E2: return potential 2 in V (float)
        scanrate: scanrate in mV/s (float)        
        Ef0: formal potential in V (float)
        c_init: intial concentration in mM (float)
        D: diffusion constant in cm^2/s (float)
        k0: standard rate constant in cm/s (float)
        T: temperature in deg C (float) (default = 21 deg C)
        alpha: transfer coefficient (float) (default = 0.5)
        Efinal: final potential in V (float) (default = E0)
        ncycles: number of cycles (int) (default = 1)
        
    Optional:
        E_steps: number of steps to devide E into (int) (default = 100)
        Lx: length of the diffusion profile: x = [0, Lx] (default = 6*sqrt(D*t_sim))
        x_steps: number of steps of the concentration profile (default = 300)
        gridtype: choose from linear or nonlinear (default 'nonlinear')
        
    Output:
        self,Eapp: potential in V (np.array)
        self.curdens: current density in mA/cm^2 (np.array)
        self.time: time in sec (np.array)
    
    --------------------------------------------------------------------------
    REMARKS:    
    In order to have a stable calculation, the value of the model diffusion constant
    D_M = D dt/dx^2 cannot exceed 0.5. So D_M < 0.5.
    This sets a limit for the max size of dt:
        dt < 0.5 dx^2/D      
    (see Bard Electrochemical methods (2001) p790-791)
    
    The paper of Jay H.Brown (2015) (doi: 10.1021/acs.jchemed.5b00225) is very useful!
    I recommend the user of this function to read his paper.
    
    If one wants to use different diffusion constants, a problem can occur.
    If D1 >> D2 or visa versa, the large D requires a large simulation length.
    Keeping the amount of x_steps constant for both D, than the dx will be way
    too large for the species with the small D. Therefore, the dx is assigned
    to the species individually.
    '''
    # initialize parameters
    E0, E1, E2, Ef = self.E0, self.E1, self.E2, self.Ef # all in V
    scanrate = self.scanrate # in V/s
    ncycles = self.ncycles 
    c_init, D, k0, Ef0 = self.c0, self.D, self.k0, self.Ef0 # in mol/cm^3, cm^2/s, cm/s and V
    alpha, T_C = self.alpha, self.T_C # T in deg C
    x_steps, E_steps = self.x_steps, self.E_steps
    
    # check if D is a list, D1 is not equal to D2
    if isinstance(D, list): D = np.array(D)
    elif isinstance(D, (float, int)): D = np.array([D,D])
    else: raise ValueError('D should be either a list with the values [D1,D2] or D should be a float (D1=D2)!')
    
    # check if c_init is a list, if yes, c_init_B is not zero
    if isinstance(c_init, list): c_init = np.array(c_init)
    elif isinstance(c_init, (float, int)): c_init = np.array([c_init,0])
    else: raise ValueError('c_init should be either a list with the values [c_init_A,c_init_B] or c_init should be a float (C_init_B = 0)!')

    # check if k0 is a list, if yes, k0B could be different than k0A
    if isinstance(k0, list): k0 = np.array(k0)
    elif isinstance(k0, (float, int)): k0 = np.array([k0,k0])
    else: raise ValueError('k0 should be either a list with the values [k0_A,k0_B] or k0 should be a float (k0_B = k0_A)!')
    
    # determine length scale of the concentration profile by finding the duration of one cycle
    t_cycle = 2*( max(E0,E1,E2,Ef)-min(E0,E1,E2,Ef) )/scanrate
    if 'Lx' in dir(self): Lx = self.Lx
    else: Lx = 6*np.sqrt(D*t_cycle*ncycles) # default length is defined as 6*sqrt(D*tsim)
    
    # creating the grid, linear or nonlinear
    if self.gridtype == 'linear':
        xvalA = np.linspace(0, Lx[0], x_steps, endpoint=True) # in cm
        xvalB = np.linspace(0, Lx[1], x_steps, endpoint=True) # in cm
        xval = np.array([xvalA,xvalB]) # length of xvalA = length of xvalB
        c_profile = np.reshape(c_init, (len(c_init),1) ) * np.ones((2,len(xvalA))) # in mol/cm^3
        
        #initize some parameters for the diffusion
        dx = xval[:,1:]-xval[:,:-1]
        # since the difference dx is one element shorter than x, we need to correct for this
        # if linear gridtype, dx is the same at all points anyway
        dx = np.append(dx,dx[:,-1].reshape((2,1)) ,axis=1) 
        dx2 = dx # no difference between dx at j or j +1/2 
    elif self.gridtype == 'nonlinear':
        a = 0.01 # non-linearity value, the larger the value, closer the spacing close to the electrode
        nonlingrid = np.cumsum(a*np.array([(1+a)**(j+1)/((1+a)**x_steps-1) for j in range(2*x_steps-1)]))
        xvalA = nonlingrid * Lx[0]/nonlingrid[-1]
        xvalB = nonlingrid * Lx[1]/nonlingrid[-1]
        xvalA, xvalB = np.insert(xvalA,0,0), np.insert(xvalB,0,0) 
        xval0 = np.array([xvalA,xvalB]) # length of xvalA = length of xvalB
        xval, xval2 = xval0[:,:-1:2], xval0[:,1::2]
        c_profile = np.reshape(c_init, (len(c_init),1) ) * np.ones((2,x_steps)) # in mol/cm^3, zeroth entry species A, first entry species B
        
        #initize some parameters for the diffusion
        dx = xval[:,1:] - xval[:,:-1] # in cm at j
        dx2 = xval2[:,1:] - xval2[:,:-1] # in cm at j-1/2 and j+1/2
        # since the difference dx is one element shorter than x, we need to correct for this
        # we can savely append the last element to the list, since this value is the fartest from the electrode
        dx = np.append(dx,dx[:,-1].reshape((2,1)) ,axis=1)
        dx2 = np.append(dx2,dx2[:,-1].reshape((2,1)) ,axis=1)
    else: 
        raise ValueError('gridtype needs to be \'linear\' or \'nonlinear\'!')
        
    #initialize the difference in time, limited by Dm
    dt = self.Dm/max(D)* min(dx.flatten())**2 # in s (max size of dt)

    # initialize butler volmer parameters
    beta = 1-alpha # alpha + beta = 1
    
    # using the time_r array, we need to setup a corresponding Eapp_r array
    # devide E in a nice amount of points
    dEmax = dt*scanrate # max dE in V
    Npoints = int(( max(E0,E1,E2,Ef)-min(E0,E1,E2,Ef) )/ dEmax + 0.5 ) # round up
    Eapp_r,time_r = self.E_app(E0,E1,E2,scanrate,Ef,ncycles,Npoints) 
    dt  = abs(time_r[1]-time_r[0]) # new dt in s, which is smaller the the old dt
    self.dt = dt
    Dm = np.reshape(np.array([D[0]/dx[0],D[1]/dx[1]])*dt, (2,x_steps)) # model diffusion coefficient times dx
        
    #warn if the length of time is large!
    if len(time_r) > 50000:
        print('The simulation might take a while since {} itterations are required!'.format(len(time_r)))
    if len(time_r) > self.MaxIter:
        raise RecursionError('The simulation needs more than {} iterations! The simulation is terminated!'.format(self.MaxIter))
    
    # create Eapp and time array for the output
    E_step = round(Npoints/E_steps)
    E_step = E_step if E_step > 0 else 1
    # warn the user if the requested amount of steps could not be used
    if E_step == 1: print('The amount of requested steps could not be used!')
    Eapp, Time = Eapp_r[0::E_step], time_r[0::E_step]  
      
    # assume Ef0 stays constant
    k_c = self.ECrateconst(Eapp_r,Ef0,k0[0],T_C,alpha) # in cm/s, reduction
    k_a = self.ECrateconst(Eapp_r,Ef0,k0[1],T_C,-beta) # in cm/s, oxidation
    
    # make empty array and initialize the first time step (first BC)
    c_temp = np.zeros((2, len(Eapp), len(xval[0]))) # c_temp[0] is c_init at the first time step
    c_temp[:,0] = c_profile # zeroth entry species A, first entry species B
    c_prev = c_temp[:,0]
    
    try: fbN = self.fbN
    except AttributeError: fbN = len(time_r)/10
    infobool = False
    if self.feedback: 
        print('Amount of iterations required: {}\n'.format(len(time_r)))
        if len(time_r) > fbN:
            infobool = True
            print('################################################')
            print('# {:>5}| {:>7}| {:>10}| {:>12} #'.format('iter','i left','Est. time left','Elapsed time'))
            print('#------|--------|---------------|--------------#')
    
    t0 = time.perf_counter()
    info = '# {:>5}| {:>7}| {:>10.3} sec| {:>8.5} sec #'
    counter_t = 1
    for idx, t in enumerate(time_r[1:],1):
        if idx == fbN: t1 = time.perf_counter()
        if idx % fbN == 0 and infobool: print(info.format(idx,len(time_r)-idx, (len(time_r)-idx)*(t1-t0)/fbN, time.perf_counter()-t0))
        c_new = c_prev + Dm*( (np.roll(c_prev,-1,axis=1) - c_prev)/dx2 - (c_prev - np.roll(c_prev,1,axis=1))/np.roll(dx2, 1, axis=1) ) # ficks second law
        c_new[:,-1] = c_new[:,-2] # at wall 2, we will always have no flux through the wall
        # flux set by the potential and concentration at the surface
        flux = (k_a[idx]*c_new[1,1] - k_c[idx]*c_new[0,1])/(1 + dx[1,0]/D[1]*k_a[idx] + dx[0,0]/D[0]*k_c[idx] ) # in mol/cm^s/s
        bc_A = c_new[0,1] + dx[0,0]/D[0] * flux # in mol/cm^3
        c_new[0,0] = bc_A if bc_A > 0 else 0 # prevents that c of A drops below 0
        bc_B = c_new[1,1] - dx[1,0]/D[1] * flux # in mol/cm^3
        c_new[1,0] = bc_B if bc_B > 0 else 0 # prevents that c of B drops below 0        
        c_prev = c_new  # overwrite previous value with new values
        if t == Time[-1]: # last value
            c_temp[:,counter_t] = c_new 
            break
        elif t == Time[counter_t]: # check if the output timestamp is reached, if yes store value
            c_temp[:,counter_t] = c_new 
            counter_t += 1
    
    # =============================================================================
    #     Optional: inlcude surface depense Ef0
    #     Delete the k_c and k_a before the for loop.
    #     replace the flux in the for loop by this piece of code:
    # =============================================================================
        # #concenration dependent Ef0
        # if c_prev[1,0] == 0: #prevent that it goes to inf
        #     Ef0n = Ef0 + const.R*T_K/const.F * np.log(c_prev[0,0])
        # elif c_prev[0,0] == 0: #prevent that it goes to inf
        #     Ef0n = Ef0 + const.R*T_K/const.F * np.log(1/c_prev[1,0])
        # else: Ef0n = Ef0 + const.R*T_K/const.F * np.log(c_prev[0,0]/c_prev[1,0])
        # k_c = ECrateconst(Eapp_r[idx-1], Ef0n, k0[0], T_C, alpha)
        # k_a = ECrateconst(Eapp_r[idx-1], Ef0n, k0[1], T_C, -beta)
        # flux = (k_a*c_new[1,1] - k_c*c_new[0,1])/(1 + dx[1]/D[1]*k_a + dx[0]/D[0]*k_c ) # in mol/cm^s/s
            
    
    # current is determined by the flux of species A
    flux = - D[0]/dx[0,0] *( c_temp[0,:,1]-c_temp[0,:,0] )
    curdens = const.F * flux *1e3 # to mA/cm^2
    
    c_temp_A = c_temp[0] * 1e3 #c_init # to mM
    c_temp_B = c_temp[1] * 1e3 #c_init # to mM
        
    if self.feedback: 
        tot_t = time.perf_counter() - t0
        unitstr = 'sec'
        if tot_t > 120:
            unitstr = 'min'
            tot_t /= 60
        print('################################################\n')
        print(('Total duration of the simulation: {:.4} '+unitstr).format(tot_t))
    
    self.time = Time
    self.Eapp = Eapp
    self.curdens = curdens
    self.xA, self.xB = xval[0], xval[1]
    self.cA, self.cB = c_temp_A, c_temp_B
