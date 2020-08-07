# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:16:43 2020

@author: bleiji
"""

import numpy as np
import time
import scipy.constants as const
const.F = const.physical_constants["Faraday constant"][0]


def CV_2D(self, *args, **kwargs):
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
        E_steps: number of steps to devide E into (int) (default = 100)
        Lx: length of the diffusion profile: x = [0, Lx] (default = 6*sqrt(D*t_sim))
        Ly: length of the diffusion profile: x = [0, Ly] (default = Lx)
        x_steps: number of steps of the concentration profile (default = 300)
        y_steps: number of steps of the concentration profile (default = 100)
        
        NOT AVAILABLE: gridtype: choose from linear or nonlinear (default 'nonlinear') 
    
    Specify the electrode with .electrode('2D', electrode=electrode):
        electrode: 2D np.array contains 0 if no electrode is present,
                   contains 1 if electrode is present
                   
    Specify the blocking layer with .insulator('2D', block=blockinglayer):
        block: 2D np.array contains 0 if no blocking layer is present,
               contains 1 if blocking layer is present
        
    Output:
        self,Eapp: potential in V (np.array)
        self.curdens: current density in mA/cm^2 (np.array)
        self.time: time in sec (np.array)
    
    --------------------------------------------------------------------------
    REMARKS:   
    In order to have a stable calculation, the maximum value of the diffusion
    model coefficient is:
        Dmx + Dmy < 0.5, Dmi = D dt/(di)^2
    Which implies that:
        dt < 0.5 / D (dx^2 dy^2)/((dx)^2 + (dy)^2)
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
    electrode = self.elec
    x_steps, y_steps, E_steps = self.x_steps, self.y_steps, self.E_steps
    
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
    
    # When using a different D for species 1 then for species 2, and when an
    # electrode is used with a specific shape, the simulation area of 
    # species 2 should be exactly equal to species 1 to prevent errors
    
    # determine length scale of the concentration profile by finding the duration of one cycle
    t_cycle = 2*( max(E0,E1,E2,Ef)-min(E0,E1,E2,Ef) )/scanrate
    if 'Lx' in dir(self): Lx = self.Lx
    else: Lx = 6*np.sqrt(D*t_cycle*np.sqrt(ncycles)) # default length is defined as 6*sqrt(D*tsim)
    if electrode is not None: Lx = np.max(Lx) # select the max value of Lx and use them for both species if a custom electrode is given
    if not isinstance(Lx,np.ndarray): Lx = np.array([Lx]*2) # check if Lx is an array:
    xvalA = np.linspace(0, Lx[0], x_steps, endpoint=True) # in cm
    xvalB = np.linspace(0, Lx[1], x_steps, endpoint=True) # in cm
    xval = np.array([xvalA, xvalB])
    
    if 'Ly' in dir(self): Ly = self.Ly
    else: Ly = Lx # default length is defined as 6*sqrt(D*tsim)      
    if not isinstance(Ly,np.ndarray): Ly = np.array([Ly]*2)
    yvalA = np.linspace(0, Ly[0], y_steps, endpoint=True) # in cm
    yvalB = np.linspace(0, Ly[1], y_steps, endpoint=True) # in cm
    yval = np.array([yvalA, yvalB])
    
    if electrode is None:
        electrode = np.zeros((y_steps,x_steps))
        electrode[:,0] = 1
        self.elec = electrode


    #check if a blocking layer is present
    if 'block' in dir(self):
        # use findsurface to find the elements and directions of the surface of the blocking layer
        block_surf, blocked, empty, normal, block_surf0, blocked0, normal0 = self.findsurface(self.block)
        self.block_surf, self.blocked = block_surf, blocked
        elec_block = electrode+2*self.block
        electrode = np.where(elec_block==1,1,0) # remove part of electrode that is blocked
        electrolyte = np.where(np.logical_or(electrode,self.block)==0,1,0)
        blockpres, lenblock = True, len(blocked) # blocking layer is present, set the length
    else:
        # where is the electrolyte and check if list elec is not empty
        electrolyte = np.where(electrode==0,1,0)
        blockpres = False # blocking layer is not present
    
    # use findsurface to find the elements and directions of the surface
    el_surf, elec, empty, normal, el_surf0, elec0, normal0 = self.findsurface(electrode)
    # el_surf, elec, empty, normal = self.findsurface(electrode)
    self.el, self.el_surf = elec, el_surf
    # el_surf is a list containing the positions of the surface of the electrode (x=dx)
    # elec is a list containing the positions of the electrode (x=0)
    
    # allow the ions to difusse into the surface of the elec/blocking layer (needed for BC)
    # electrolyte[tuple(elec0)] = 1
    for el in elec:
        electrolyte[tuple(el)] = 1
    # if blockpres: electrolyte[tuple(blocked0)] = 1
    if blockpres: 
        for el in blocked:
            electrolyte[tuple(el)] = 1
            
    # create the mesh and _profiles
    meshA = np.meshgrid(xvalA,yvalA)
    meshB = np.meshgrid(xvalB,yvalB)
    c_profileA, c_profileB = c_init[0]*electrolyte, c_init[1]*electrolyte # in mol/cm^3
    c_profile = np.array([c_profileA,c_profileB]) # in mol/cm^3

    # initialize butler volmer parameters
    beta = 1-alpha # alpha + beta = 1
    
    #initize some parameters for the diffusion
    dx, dy = [abs(el[:,1]-el[:,0]) for el in [xval, yval]] # in cm
    dt = min(self.Dm/D * (dx*dy)**2/(dx**2+dy**2)) # in s (select the smallest of the two)
    Dm_x, Dm_y = [np.reshape(D*dt/el**2, (2,1,1)) for el in [dx,dy]]
    diff = np.delete([dx,dx,dy,dy],empty, axis=0) # list with differences corresponding to non empty elements
    dD = [(el/D*np.array([1,-1])).reshape((2,1)) for el in diff] # speeds up the simulation, [1,-1] is for positive and negative flux
    self.dt = dt
    
    
    
    
    
    
    # normalize the normal vector using the dx and dy values
    x_or_y_dir = np.delete(np.array([0,0,1,1]), empty) # zeroth entry is x, first entry is y
    for idx, el in enumerate(normal):
        temp = abs(el) * np.array([[dx[0]],[dy[0]]])
        normal[idx] = (temp/(np.sqrt(temp[0]**2 + temp[1]**2)))[x_or_y_dir[idx]]
        
    # temp = normal0 * np.array([[dx[0]],[dy[0]]])
    # normal0 = (temp/(np.sqrt(temp[0]**2 + temp[1]**2)))
    
    
    
    # return
    
    
    
    
    
    
    
    
    
    
    
    
    # create time and Eapp array using dt
    dEmax = dt*scanrate # max dE in V
    Npoints = int(( max(E0,E1,E2,Ef)-min(E0,E1,E2,Ef) )/ dEmax + 0.5 ) # round up
    Eapp_r,time_r = self.E_app(E0,E1,E2,scanrate,Ef,ncycles,Npoints) 
    dt  = abs(time_r[1]-time_r[0]) # new dt in s, which is smaller the the old dt

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
    # make time_r shorter by stopping at the last timestamp of Time
    time_r = time_r[:np.where(time_r == Time[-1])[0][0]+1]
    
    # assume Ef0 stays constant
    k_c = self.ECrateconst(Eapp_r,Ef0,k0[0],T_C,alpha)
    k_a = self.ECrateconst(Eapp_r,Ef0,k0[1],T_C,-beta)
    k = np.array([k_c,k_a]) # in cm/s, c: reduction and a: oxidation
    
    # make dictionary to speed up the simulation 
    c_cache = {}
    c_cache[0] = c_profile
    c_prev = c_profile
    
    try: fbN = self.fbN
    except AttributeError: fbN = int(round(len(time_r)/11))
    infobool = False
    if self.feedback: 
        print('Amount of iterations required: {}\n'.format(len(time_r)))
        if len(time_r) >  fbN:
            infobool = True
            print('################################################')
            print('# {:>6}| {:>7}| {:>10}| {:>12} #'.format('iter','i left','Est. time left','Elapsed time'))
            print('#-------|--------|---------------|--------------#')
    
    infosec = '# {:>6}| {:>7}| {:>10.4} sec| {:>8.5} sec #'
    infomin = '# {:>6}| {:>7}| {:>10.4} min| {:>8.5} sec #'
    counter_t = 1
    t0 = time.perf_counter()
    
    # start time of the sim
    for idx,t in enumerate(time_r[1:],1):
        if idx % fbN == 0 and infobool: # some feedback
            if idx == fbN: t1 = time.perf_counter()
            tleft = (len(time_r)-idx)*(t1-t0)/fbN
            if tleft > 120: print(infomin.format(idx,len(time_r)-idx, (len(time_r)-idx)*(t1-t0)/fbN/60, time.perf_counter()-t0))
            else: print(infosec.format(idx,len(time_r)-idx, (len(time_r)-idx)*(t1-t0)/fbN, time.perf_counter()-t0))
        cnewx = Dm_x*( np.roll(c_prev,-1, axis=2) - 2*c_prev + np.roll(c_prev,1,axis=2) )
        cnewy = Dm_y*( np.roll(c_prev,-1, axis=1) - 2*c_prev + np.roll(c_prev,1,axis=1) )
        c_new = (c_prev + cnewx + cnewy) * electrolyte # non zero concentrations where electrolyte = 1
        # no flux through the walls
        c_new[:,[0,-1]], c_new[:,:,[0,-1]] = c_new[:,[1,-2]], c_new[:,:,[1,-2]]
        #set the corners, no flux through the corners
        c_new[:,[0,-1,0,-1],[0,-1,-1,0]] = c_new[:,[1,-2,1,-2],[1,-2,-2,1]] 
        if blockpres:
            for idx2 in range(lenblock):  # no flux through the wall of the blocking layer
                c_new[:,blocked[idx2][0],blocked[idx2][1]] = c_new[:,block_surf[idx2][0],block_surf[idx2][1]]
        
        
        
        # rewrite this to one array instead of list with 4 arrays
        
        
        
        for idx2,el in enumerate(el_surf):
            flux = (k_a[idx]*c_new[1,el[0],el[1]] - k_c[idx]*c_new[0,el[0],el[1]])/(1 + np.dot(diff[idx2]/D,k[:,idx]) ) # in mol/cm^s/s
            bc = c_new[:,el[0],el[1]] + dD[idx2] * flux * normal[idx2] # normalized new value at the surface in mol/cm^3 
            c_new[:,elec[idx2][0],elec[idx2][1]] = np.where(bc>0, bc, 0) # along y (x=0), prevents that c drops below 0
       
        
       
        c_prev = c_new # overwrite previous value with new value
        if t == Time[counter_t]: # check if the output timestamp is reached, if yes store value
            c_cache[counter_t] = c_new
            counter_t += 1  
    #end of the sim 
            
    c_new = np.array(list(c_cache.values())) # extract the values from the directory
    
    flux = np.ones((len(Time),1))
    # current is determined by the flux of species A at the surface!
    for idx,el in enumerate(el_surf):
        first, last = np.where(el[0]==0)[0], np.where(el[0]==y_steps-1)[0]
        el_surf[idx] = np.delete(el_surf[idx], np.append(first,last), axis=1)
        elec[idx] = np.delete(elec[idx], np.append(first,last), axis=1)
        newflux = - D[0]/diff[idx][0] *( c_new[:,0,el_surf[idx][0],el_surf[idx][1]] - c_new[:,0,elec[idx][0],elec[idx][1]] )
        flux = np.append(flux, newflux, axis = 1)
    flux = np.delete(flux, 0, axis=1) # remove the first colum, since we needed it for the correct shape
    curdens = const.F * flux *1e3 # to mA/cm^2
    
    
    
    
    
    
    
    # repeat for the x direction!!
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # integrate the current density to obtain the line current density
    L_elec, diff = np.array([]), np.delete([dy,dy,dx,dx],empty, axis=0)[:,0] # list with the lengths of the surface of the elements (so posxpos has a surface length of dy)
    for idx,el in enumerate(elec): L_elec = np.append(L_elec, np.ones(len(el[0]))*diff[idx] )
    
    # devide the total line current by the total circumference of the electrode
    ave_curdens = np.sum(curdens*L_elec, axis=1) / np.sum(L_elec) # in mA/cm^2
    curdens = np.append(curdens, ave_curdens.reshape((len(curdens),1)), axis=1) # last element of curdens will be the average
         
    if self.feedback: 
        tot_t = time.perf_counter() - t0
        unitstr = 'sec'
        if tot_t > 120:
            unitstr = 'min'
            tot_t /= 60
        if infobool: print('################################################\n')
        print(('Total duration of the simulation: {:.4} '+unitstr).format(tot_t))
 
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
    
    self.time = Time
    self.Eapp = Eapp
    self.curdens = curdens
    self.meshA, self.meshB = meshA, meshB
    self.cA, self.cB = c_new[:,0]*1e3, c_new[:,1]*1e3
