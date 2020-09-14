# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:19:47 2020

@author: bleiji
"""

import numpy as np
import time
import scipy.constants as const
const.F = const.physical_constants["Faraday constant"][0]

def CV_3D(self, *args, **kwargs):
    '''
    This is the main function to simulate a cv.
    We are looking to the following process:
    A^+n + e^- -> B^+n-1
    We are reducing ion A to ion B by transfering 1 electron. 
    We assume that:
        the diffusion to the electrode is in 3D
        the concentration profile is constant at t= 0
        the length of the diffusion profile is max Lx = 6*sqrt(D*t_sim)
    
    All the nessarely parameters must be set using __init__() and parameters().
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
    z_steps: number of steps of the concentration profile (default = 100)
    
    NOT AVAILABLE: gridtype: choose from linear or nonlinear (default 'nonlinear') 
 
    Specify the electrode with .electrode('3D', electrode=electrode):
    electrode: 3D np.array contains 0 if no electrode is present,
    contains 1 if electrode is present
    
    Specify the blocking layer with .insulator('3D', block=blockinglayer):
    block: 3D np.array contains 0 if no blocking layer is present,
    contains 1 if blocking layer is present
    
    Output:
    self,Eapp: potential in V (np.array)
    self.curdens: current density in mA/cm^2 (np.array)
    self.time: time in sec (np.array)
    
    --------------------------------------------------------------------------
    REMARKS:   
    In order to have a stable calculation, the maximum value of the diffusion
    model coefficient is:
    Dmx + Dmy + Dmz < 0.5, Dmi = D dt/(di)^2
    Which implies that:
    dt < 0.5 (dx^2 dy^2 dz^2)/( (dy*dz)^2 + (dx*dz)^2 + (dx*dy)^2 ) / D
    (see Bard Electrochemical methods (2001) p790-791)
    
    The paper of Jay H.Brown (2015) (doi: 10.1021/acs.jchemed.5b00225) is very useful!
    I recommend the user of this function to read his paper.
    
    If one wants to use different diffusion constants, a problem can occur.
    If D1 >> D2 or visa versa, the large D requires a large simulation length.
    Keeping the amount of x_steps constant for both D, than the dx will be way
    too large for the species with the small D. Therefore, the dx is assigned
    to the species individually.
    
    --------------------------------------------------------------------------
    GOVERNING EQUATIONS:
    Jox = (flux_a-flux_c)/(1 + C0(ka + kc))
    flux_a = ka( Cx C100,red + Cy C010,red + Cz C001,red)
    flux_c = kc( Cx C100,ox + Cy C010,ox + Cz C001,ox)
    C0,ox = C0 Jox + Cx C100,ox + Cy C010,ox + Cz C001,ox
    C0  = dx dy dx /den
    Cx  = nx dy dz /den
    Cy  = dx ny dz /den
    Cz  = dx dy nz /den
    den = nx dy dz + ny dx dz + nz dx dy
    '''
    
    # initialize parameters
    E0, E1, E2, Ef = self.E0, self.E1, self.E2, self.Ef # all in V
    scanrate = self.scanrate # in V/s
    ncycles = self.ncycles 
    c_init, D, k0, Ef0 = self.c0, self.D, self.k0, self.Ef0 # in mol/cm^3, cm^2/s, cm/s and V
    alpha, T_C = self.alpha, self.T_C # T in deg C
    electrode = self.elec
    x_steps, y_steps, z_steps, E_steps = self.x_steps, self.y_steps, self.z_steps, self.E_steps
    
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
    else: Lx = 6*np.sqrt(D*t_cycle*np.sqrt(ncycles)) # default length is defined as 6*sqrt(D*tsim)
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

    if 'Lz' in dir(self): Lz = self.Lz
    else: Lz = Lx # default length is defined as 6*sqrt(D*tsim)      
    if not isinstance(Lz,np.ndarray): Lz = np.array([Lz]*2)
    zvalA = np.linspace(0, Lz[0], z_steps, endpoint=True) # in cm
    zvalB = np.linspace(0, Lz[1], z_steps, endpoint=True) # in cm
    zval = np.array([zvalA, zvalB])
     
    # Create default flat electrode in x,z plane
    if electrode is None:
        electrode = np.zeros((z_steps,y_steps,x_steps),dtype=np.int32)
        electrode[:,:,0] = 1
        self.elec = electrode
      
    # check if the electrodes is at one of the walls:
    wall_electrode_x, wall_electrode_y, wall_electrode_z = False, False, False
    if ((electrode[:,:,0] == 1).all()) and ((electrode[:,:,1:] == 0).all()): wall_electrode_x = True
    elif ((electrode[:,0,:] == 1).all()) and ((electrode[:,1:,:] == 0).all()): wall_electrode_y = True
    elif ((electrode[0,:,:] == 1).all()) and ((electrode[1:,:,:] == 0).all()): wall_electrode_z = True
    
    # use findsurface to find the elements and directions of the surface
    elec, normal = self.findsurface(electrode)
    # elec is a list containing the positions of the electrode
    # normal is a list containing the corresponding normal vectors (not normalized)
    # the zeroth entry is the z value, first is y and second is x
    
    # find the surface elements by displacing the electrode element with the normal vector
    el_surfx = np.array([elec[0], elec[1], elec[2] + normal[2]]).astype(np.int)
    el_surfy = np.array([elec[0], elec[1] + normal[1], elec[2]]).astype(np.int)
    el_surfz = np.array([elec[0] + normal[0], elec[1], elec[2]]).astype(np.int)
    
    #check if a blocking layer is present
    if 'block' in dir(self): # use findsurface to find the elements and directions of the surface of the blocking layer
        blocked, normal_block  = self.findsurface(self.block)
        blocked_surf =  (blocked + normal_block).astype(int)
        electrolyte = np.where(np.logical_or(electrode,self.block)==0,1,0)
        
        # find the elements which directly border to the insulator
        blocked_surfz = blocked + np.array([normal_block[0],np.zeros(len(normal_block[0])),np.zeros(len(normal_block[0]))])
        blocked_surfy = blocked + np.array([np.zeros(len(normal_block[0])),normal_block[1],np.zeros(len(normal_block[0]))])
        blocked_surfx = blocked + np.array([np.zeros(len(normal_block[0])),np.zeros(len(normal_block[0])),normal_block[2]])
        blocked_surf_ext = np.unique(np.concatenate((blocked_surfz,blocked_surfy,blocked_surfx),axis=1),axis=1).astype(int)
        
        # remove the elements from normal and elec which are blocked or are next to the insulator surface
        ext_ins = self.block.copy()
        ext_ins[tuple(blocked_surf_ext)] = 1 # elec is also not allowed on the surface of the insulator
        temp = np.concatenate((elec,np.array(np.where(ext_ins==1))),axis=1)
        uniq, inv, cnt = np.unique(temp, axis=1, return_counts = True, return_inverse=True) # find the positions of the duplicates with cnt
        dup_idx = np.where(cnt[inv]>1)[0] # select the duplicates in the original order
        dup_idx = dup_idx[np.where(dup_idx <= len(elec.T))[0]] # take only the duplicates from the elec array
        elec, normal = np.delete(elec.T,dup_idx,axis=0).T, np.delete(normal.T,dup_idx,axis=0).T # remove the duplicates
        el_surfx, el_surfy, el_surfz = np.delete(el_surfx.T,dup_idx,axis=0).T, np.delete(el_surfy.T,dup_idx,axis=0).T, np.delete(el_surfz.T,dup_idx,axis=0).T # remove the duplicates
        
        self.blocked = blocked
        blockpres = True # blocking layer is present
    else: # where is the electrolyte and check if list elec0 is not empty
        electrolyte = np.where(electrode==0,1,0)
        blockpres = False # blocking layer is not present
    
    #find the surface elements of the inverse of the electrolyte
    el_electrolyte, _ = self.findsurface(np.where(electrolyte == 1,0,1))    
    electrolyte[tuple(el_electrolyte)] = 1 # allow the ions to difusse into the surface of the electrode and insulating layer (needed for BC)
    
    # create the mesh and c_profiles
    meshA = np.meshgrid(xvalA,yvalA,zvalA)
    meshB = np.meshgrid(xvalB,yvalB,zvalB)
    
    c_profileA, c_profileB = c_init[0]*electrolyte, c_init[1]*electrolyte # in mol/cm^3
    c_profile = np.array([c_profileA,c_profileB]) # in mol/cm^3
    
    # initialize butler volmer parameters
    beta = 1-alpha # alpha + beta = 1
    
    # initialize some parameters for the diffusion
    dx, dy, dz = [abs(el[:,1]-el[:,0]) for el in [xval, yval, zval]] # in cm
    dt = min( self.Dm/D * (dx*dy*dz)**2/( (dy*dz)**2 + (dx*dz)**2 + (dx*dy)**2 ) ) # in s (select the smallest of the two)
    Dm_x, Dm_y, Dm_z = [np.reshape(D*dt/el**2, (2,1,1,1)) for el in [dx,dy,dz]]
    self.dt = dt
    
    # normalize the normal vector
    norm_dx_dy_dz = normal * np.array([[dz[0]],[dy[0]],[dx[0]]])
    normal = abs(norm_dx_dy_dz/(np.sqrt(norm_dx_dy_dz[0]**2 + norm_dx_dy_dz[1]**2 + norm_dx_dy_dz[2]**2)))

    # some constants to speed up the simulation
    # normal[2] = nx, normal[1] = ny, normal[0] = nz
    dx, dy, dz  = dx.reshape((2,1)), dy.reshape((2,1)), dz.reshape((2,1))
    denom = normal[2] *dy*dz + normal[1] *dx*dz + normal[0] *dx*dy
    C0 = dx * dy * dz / (denom * D.reshape((2,1)))
    Cx = normal[2] * dy * dz / denom
    Cy = normal[1] * dx * dz / denom
    Cz = normal[0] * dx * dy / denom
    
    # create time and Eapp array using dt
    dEmax = dt*scanrate # max dE in V
    Npoints = int(( max(E0,E1,E2,Ef)-min(E0,E1,E2,Ef) )/ dEmax + 0.5 ) # round up
    Eapp_r,time_r = self.E_app(E0,E1,E2,scanrate,Ef,ncycles,Npoints) 
    dt  = abs(time_r[1]-time_r[0]) # new dt in s, which is smaller the the old dt
      
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
    
    # make dictionary to speed up the simulation 
    c_cache = {}
    c_profile0 = c_profile.copy()
    if blockpres: c_profile0[:,blocked[0],blocked[1],blocked[2]] = 0 # set c equal to zero at the surface of the blocking layer
    c_cache[0], c_prev = c_profile0, c_profile
    
    try: fbN = self.fbN
    except AttributeError: fbN = int(round(len(time_r)/11))
    infobool = False
    # check if n0 of iter < MaxIter
    if len(time_r) > self.MaxIter:
       raise RecursionError('The simulation needs more than {} iterations! The simulation is terminated!'.format(self.MaxIter))
    # give some feedback
    if self.feedback:
         #warn if the length of time is large!
        if len(time_r) > 50000: print('The simulation might take a while since {} itterations are required!'.format(len(time_r)))
        else: print('Amount of iterations required: {}\n'.format(len(time_r)))
        if len(time_r) >  fbN:
            infobool = True
            print('#################################################')
            print('# {:>6}| {:>7}| {:>10}| {:>12} #'.format('iter','i left','Est. time left','Elapsed time'))
            print('#-------|--------|---------------|--------------#')
        
    infosec = '# {:>6}| {:>7}| {:>10.4} sec| {:>8.5} sec #'
    infomin = '# {:>6}| {:>7}| {:>10.4} min| {:>8.5} sec #'
    
    # start time of the sim
    t0, counter_t = time.perf_counter(), 1
    for idx,t in enumerate(time_r[1:],1):
        if idx % fbN == 0 and infobool: 
            if idx == fbN: t1 = time.perf_counter()
            tleft = (len(time_r)-idx)*(t1-t0)/fbN
            if tleft > 120: print(infomin.format(idx,len(time_r)-idx, (len(time_r)-idx)*(t1-t0)/fbN/60, time.perf_counter()-t0))
            else: print(infosec.format(idx,len(time_r)-idx, (len(time_r)-idx)*(t1-t0)/fbN, time.perf_counter()-t0))
        cnewx = Dm_x*( np.roll(c_prev,-1, axis=3) - 2*c_prev + np.roll(c_prev,1,axis=3) )
        cnewy = Dm_y*( np.roll(c_prev,-1, axis=2) - 2*c_prev + np.roll(c_prev,1,axis=2) )
        cnewz = Dm_z*( np.roll(c_prev,-1, axis=1) - 2*c_prev + np.roll(c_prev,1,axis=1) )
        c_new = (c_prev + cnewx + cnewy + cnewz)*electrolyte
        # no flux through the walls
        c_new[:,[0,-1]], c_new[:,:,[0,-1]], c_new[:,:,:,[0,-1]] = c_new[:,[1,-2]], c_new[:,:,[1,-2]], c_new[:,:,:,[1,-2]]
        # set the edges and corners of the simulation space separately
        c_new[:,[0,-1,0,-1],[0,-1,-1,0]] = c_new[:,[1,-2,1,-2],[1,-2,-2,1]] # lines along x
        c_new[:,[0,-1,0,-1],:,[0,-1,-1,0]] = c_new[:,[1,-2,1,-2],:,[1,-2,-2,1]] # lines along y
        c_new[:,:,[0,-1,0,-1],[0,-1,-1,0]] = c_new[:,:,[1,-2,1,-2],[1,-2,-2,1]] # lines along z
        c_new[:,[0,-1,0,0,0,-1,-1,-1],[0,-1,0,-1,-1,0,0,-1],[0,-1,-1,0,-1,0,-1,0]] = c_new[:,[1,-2,1,1,1,-2,-2,-2],[1,-2,1,-2,-2,1,1,-2],[1,-2,-2,1,-2,1,-2,1]] # corners
        # no flux through the wall of the blocking layer
        if blockpres: c_new[:,blocked[0],blocked[1],blocked[2]] = c_new[:,blocked_surf[0],blocked_surf[1],blocked_surf[2]]
        # electrochemical part        
        flux_a = k_a[idx]*(Cx*c_new[1,el_surfx[0],el_surfx[1],el_surfx[2]] + Cy*c_new[1,el_surfy[0],el_surfy[1],el_surfy[2]] +  Cz*c_new[1,el_surfz[0],el_surfz[1],el_surfz[2]]) # calculate the flux for the anodic reaction
        flux_c = k_c[idx]*(Cx*c_new[0,el_surfx[0],el_surfx[1],el_surfx[2]] + Cy*c_new[0,el_surfy[0],el_surfy[1],el_surfy[2]] +  Cz*c_new[0,el_surfz[0],el_surfz[1],el_surfz[2]]) # calculate the flux for the cathodic reaction
        flux = (flux_a - flux_c)/(1 + C0*(k_a[idx] + k_c[idx])) *np.array([[1],[-1]]) # change the sign the the order species: Jox = - Jred
        bc = C0*flux + Cx*c_new[:,el_surfx[0], el_surfx[1],el_surfx[2]] + Cy*c_new[:,el_surfy[0],el_surfy[1],el_surfy[2]] + Cz*c_new[:,el_surfz[0],el_surfz[1],el_surfz[2]]
        c_new[:,elec[0],elec[1],elec[2]] = np.where(bc > 0, bc, 0) # prevents that c drops below 0
        c_prev = c_new # overwrite previous value with new value
        if t == Time[counter_t]: # check if the output timestamp is reached, if yes store value
            cnew = c_new.copy()
            if blockpres: cnew[:,blocked[0],blocked[1],blocked[2]] = 0 # set c equal to zero at the surface of the blocking layer
            c_cache[counter_t] = cnew
            counter_t += 1  
    #end of the sim
    
    # extract the values from the directory
    c_new = np.array(list(c_cache.values())) # extract the values from the directory
    
    # find the flux noraml to the surface
    flux_x = normal[2]/dx[0] * (c_new[:,0,el_surfx[0],el_surfx[1],el_surfx[2]] - c_new[:,0,elec[0],elec[1],elec[2]])
    flux_y = normal[1]/dy[0] * (c_new[:,0,el_surfy[0],el_surfy[1],el_surfy[2]] - c_new[:,0,elec[0],elec[1],elec[2]])
    flux_z = normal[0]/dz[0] * (c_new[:,0,el_surfz[0],el_surfz[1],el_surfz[2]] - c_new[:,0,elec[0],elec[1],elec[2]])
    flux = - D[0] * (flux_x + flux_y + flux_z)
    curdens = const.F * flux * 1e3 # to mA/cm^2
    
    # convert normalized vector back to [1,1] for corner elements
    normal[tuple(np.where((normal > 0 ) & (normal < 1 )))] = 1
    
    # integrate the current density to obtain the line current density
    L_elec = normal * np.array([dz[0],dy[0],dx[0]]).reshape((3,1))
    L_elec = np.sqrt(L_elec[0]**2+L_elec[1]**2+L_elec[2]**2)
    
    # devide the total line current by the total circumference of the electrode
    ave_curdens = np.sum(curdens*L_elec, axis=1) / np.sum(L_elec) # in mA/cm^2
    curdens = np.append(curdens, ave_curdens.reshape((len(curdens),1)), axis=1) # last element of curdens will be the average
    
    # remove the elements which are at the edge of the simulation area
    # border : [0] = {0,zsteps-1}, [1] = {0,ysteps-1}, [2] = {0,xsteps-1}
    edge_x, edge_y, edge_z = np.array([]), np.array([]), np.array([])
    if not wall_electrode_x: edge_x = np.where((elec[2] == 0) | (elec[2] == x_steps-1))[0]
    if not wall_electrode_y: edge_y = np.where((elec[1] == 0) | (elec[1] == y_steps-1))[0]
    if not wall_electrode_z: edge_z = np.where((elec[0] == 0) | (elec[0] == z_steps-1))[0]
    
    edge = np.concatenate((edge_x, edge_y, edge_z))
    curdens = np.delete(curdens,edge,axis=1)
    elec, normal = np.delete(elec,edge,axis=1),  np.delete(normal,edge,axis=1)
    
    if self.feedback: 
        tot_t = time.perf_counter() - t0
        unitstr = 'sec'
        if tot_t > 120:
            unitstr = 'min'
            tot_t /= 60
        if infobool: print('#################################################\n')
        print(('Total duration of the simulation: {:.4} '+unitstr).format(tot_t))
    
    self.el, self.normal = elec, normal
    self.time, self.Eapp, self.curdens = Time, Eapp, curdens
    self.meshA, self.meshB = meshA, meshB
    self.cA, self.cB = c_new[:,0]*1e3, c_new[:,1]*1e3