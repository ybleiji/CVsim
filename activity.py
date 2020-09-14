# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 16:51:28 2020

@author: bleiji
"""
import numpy as np
import matplotlib.pyplot as plt
from .genfunc import fancyplot, find_nearest

def activity(self, x=None, y=None, z=None, tstart=None, tend=None,
             units=['um','C/cm2'], showhis=False, *args, **kwargs):
    '''
    This functions calculates the hotspots of an arbitrary shaped electrode.
    The activity is defined as the total charge transferred. The function is 
    available for the 2D and 3D cases only!
    
    Paramters:
        x: select a plane along the x direction (int), 3D only
        y: select a plane along the y direction (int), 3D only
        z: select a plane along the z direction (int), 3D only
        tstart: lower bound of the integral in s (float) (default = 0)
        tend: upper bound of the itegral in s (float) (default = t(E = E1))
        units: specify the units of the labels: length unit will be the same for all dimension: ['um','C/cm2']
        showhis: returns a histogram of the charge per pixel (bool)
        saveas: save the figure with the filename given
    '''
    x0,y0,z0 = x,y,z # rename x to x0, etc
    if tstart is None: tstart = 0
    else: tstart = find_nearest(self.time, tstart)
    if tend is None: tend = self.Eapp.argmin()
    else: tend = find_nearest(self.time, tend)
    
    if self.dim == '1D':
        t = self.time[tstart:tend]
        curdens = self.curdens[tstart:tend]
        chargedens = abs(np.trapz(curdens, t, axis=0))
        print("The total transfered charge between {} s and {} s is {:.4f} mC/cm^2.".format(tstart,tend,chargedens))
        return
    
    # integrate the current density to time: charge density = int curdens dt 
    t = self.time[tstart:tend]
    curdens = self.curdens[tstart:tend,:-1]
    chargedens = abs(np.trapz(curdens, t, axis=0))
    
    # set the charge transfer grid, use the size of the electrode
    chargedensgrid = self.elec*0
    
    # check which spatial units to use
    if units[0] == 'um' or units[0] == '$\\mu m$': f, units[0] = 1e4, '$\\mu m$'
    elif units[0] == 'mm': f = 10
    elif units[0] == 'cm': f = 1
    elif units[0] == 'm': f = 1e-2
    else: raise ValueError('The zeroth element of units needs to be um, mm, cm or m only!')
    
    # check which units to use for the concentration
    if units[1] == 'C/cm2': pass
    elif units[1] in ['C/mm2', 'uC/um2']: chargedens = chargedens*1e-2
    elif units[1] == 'C/um2': chargedens = chargedens*1e-8
    elif units[1] == 'mC/cm2': chargedens = chargedens*1e6
    elif units[1] == 'mC/um2': chargedens = chargedens*1e-5
    else: raise ValueError('The first element of units needs to be \'C/cm2\', \'C/um2\', \'C/mm2\', \'C/cm2\', \'mC/cm2\' or \'mC/um2\' only!  {} is not allowed!'.format(units[-1]))
    barunit = units[1].replace('2', '$^2$')

    # 2D and 3D will be different:
    if self.dim == '2D':
        elec = self.el
    
        # set the values of the grid
        chargedensgrid[tuple(elec)] = chargedens
        
        # set the X and Y values
        X,Y = self.meshA
        XX,YY = f*X, f*Y # scale the x and y axis
        if 'label' not in kwargs: kwargs['label'] = ['x ('+units[0]+')','y ('+units[0]+')']
       
    elif self.dim == '3D':
        # raise an error when no plane is specified
        if x0 is None and y0 is None and z0 is None: raise ValueError('Please select a plane using the argument x=..., or y=..., or z=...!')
        elec = self.el
        
        # set the values of the grid
        chargedensgrid[tuple(elec)] = chargedens
        
        # set the X, Y and Z values
        X,Y,Z = self.meshA
        X,Y,Z = f*X, f*Y, f*Z
        x,y,z = X[0,:,0],  Y[:,0,0], Z[0,0,:]
        
        # select the 2D plane to be plotted
        if x0 is not None:
            XX, YY = Y[:,0,:], Z[:,0,:] # select the y and z values
            chargedensgrid = chargedensgrid[:,:,x0].T
            if 'label' not in kwargs: kwargs['label'] = ['y ('+units[0]+')','z ('+units[0]+')']
            kwargs['title'] = 'x = {:.2f} {}'.format(x[x0],units[0])
        elif y0 is not None:
            XX, YY = X[0,:,:], Z[0,:,:] # select the x and z values
            chargedensgrid = chargedensgrid[:,y0,:].T
            if 'label' not in kwargs: kwargs['label'] = ['x ('+units[0]+')','z ('+units[0]+')']
            kwargs['title'] = 'y = {:.2f} {}'.format(y[y0],units[0])
        elif z0 is not None:
            XX, YY = X[:,:,0], Y[:,:,0] # select the x and y values
            chargedensgrid = chargedensgrid[z0,:,:]
            if 'label' not in kwargs: kwargs['label'] = ['x ('+units[0]+')','y ('+units[0]+')']
            kwargs['title'] = 'z = {:.2f} {}'.format(z[z0],units[0])
    
    
    # check the xlim and ylim with the scaled values
    if 'xlim' not in kwargs: kwargs['xlim'] = [0, max(XX.flatten())]
    if 'ylim' not in kwargs: kwargs['ylim'] = [0, max(YY.flatten())]
    saveas = kwargs.get('saveas', None)
    if saveas != None: 
        if 'fontsize' not in kwargs: kwargs['fontsize'] = 10
    barlabel = kwargs.get('barlabel', 'Charge density ('+barunit+')' )
    if 'minbar' not in kwargs: minbar = 0
    if 'maxbar' not in kwargs: 
        maxbar = round(max(chargedens.flatten()))
        if maxbar == 0: maxbar = max(chargedens.flatten()) # prevent that maxbar will be zero
    if 'title' not in kwargs: kwargs['title'] = "t$_{start}$ = " + str(tstart) + " s, t$_{end}$ = " + str(tend) +" s" # show the range of integration

    # remove all the keys which should not be passed to fancyplot
    for key in ['saveas', 'minbar', 'maxbar', 'barlabel']: 
        if key in kwargs: del kwargs[key]
    
    # plot the pcolormesh
    fig, ax = fancyplot(**kwargs, returnfig = True)
    colorplot = plt.pcolormesh(XX,YY, chargedensgrid, cmap = 'inferno', vmax=minbar, vmin=maxbar, shading='gouraud') 
    bar = fig.colorbar(colorplot)
    bar.set_label(label=barlabel) #add a label to the colorbar
    # save the figure if wanted
    if saveas != None: plt.savefig(fname=saveas, dpi=300, bbox_inches='tight')
    plt.show()
    
        
    # plot the histogram if showhis
    if showhis:
        # fig, ax = fancyplot([],[])
        plt.hist(chargedens, bins='auto')
        # plt.title("Histogram with 'auto' bins")
        plt.title('Parameters: D={} cm$^2$/s, k$_0$={} cm/s, $\\nu$= {} mV/s'.format(self.D, self.k0, 1000*self.scanrate))
        plt.xlabel('Charge density ('+barunit+')')
        plt.ylabel('Amount')
        plt.show()
    
    self.chargedens = chargedens