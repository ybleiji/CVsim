# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:42:57 2020

@author: bleiji
"""

import numpy as np
import matplotlib.pyplot as plt
from .genfunc import checkkeys, fancyplot, find_nearest


def plot(self, Type='CV', *args, **kwargs):
    '''
    Use this function to plot the results.
    Input paramters:
        Type: should be one of:
                'CV': plot the current vs the potential
                'tE': plot the potential vs time
                'tI': plot the current vs time
                'tcA': plot the surface concentration of A vs time
                'tcB': plot the surface concentration of B vs time
                'tc': plot both the surface concentration of A and B vs time
    Optional:
        units: specifies the units of the current density (default mA/cm^2)
        ylim = [float,float]
        xlim = [float,float]
        title = string
        legend = [string]
        imagesize = [float,float]
        fontsize = int
        grid = bool
        saveas = string
        origin = bool
        linestyle = string
        color = string
        loc = string
        xscale = 'linear', 'log'
        yscale = 'linear', 'log'
    '''
    retfig = kwargs.get('returnfig', False)
    
    if not self.runned: raise Warning('Run the simulation first using .run()!')
    if Type not in ['CV', 'tE', 'tI', 'tcA', 'tcB', 'tc']:
        raise ValueError('Type should be one of the following: \'CV\', \'tE\', \'tI\', \'tcA\', \'tcB\' or \'tc\'!')
    
    # copy the kwargs and remove the keys that are not used in the plot
    plotoptions = kwargs.copy()
    try: del plotoptions['units'] # remove keys which are not used in the plot
    except KeyError: pass

    # check the units
    if Type == 'CV' or Type == 'tI':
        units = kwargs.get('units', 'mA/cm2')
        if units == 'mA/cm2':
            units = '(mA/cm$^2$)'
            curdens = self.curdens
        elif units == 'A/cm2':
            units = '(A/cm$^2$)'
            curdens = 1e-3*self.curdens
        elif units == 'A/m2':
            units = '(A/m$^2$)'
            curdens = 10*self.curdens
        else:
            raise ValueError('The units should be \'mA/cm2\', \'mA/m2\' or \'A/m2\'!')
    elif Type == 'tcA' or Type == 'tcB' or Type == 'tc':
        units = kwargs.get('units', 'mM')
        if units == 'mM':
            cA = 1e3*self.cA
            cB = 1e3*self.cB
        elif units == 'M':
            cA = self.cA
            cB = self.cB
        else:
            raise ValueError('The units should be \'mM\' or \'M\'!')
        plotoptions['label'] = ['Time (s)', 'Concentration ('+units+')']
    
    
    #check keys
    allowedkeys = {'ylim':None,'xlim':None,'title':None,'legend':None,
                   'imagesize':None,'fontsize':None,'grid':None,'saveas':None,
                   'origin':None,'style':None,'loc':None,'color':None,
                   'xscale':None,'yscale':None,'units':None,'returnfig':None,
                   'showfig':None }
    checkkeys('plot()', kwargs, allowedkeys)

    if Type == 'CV':
        plotoptions['label'] = ['Potential (V)', 'Current density '+units]
        if self.dim == '1D': 
            fig = fancyplot(self.Eapp, curdens, **plotoptions)
        elif self.dim == '2D' or self.dim=='3D':
            plotoptions['title'] = 'Average current density'
            fig = fancyplot(self.Eapp, curdens[:,-1], **plotoptions)
            
            
    elif Type == 'tE':
        plotoptions['label'] = ['Time (s)', 'Potential (V)']
        fig = fancyplot(self.time, self.Eapp, **plotoptions)
        
        
    elif Type == 'tI':
        plotoptions['label'] = ['Time (s)', 'Current density '+units]
        if self.dim == '1D': 
            fig = fancyplot(self.time, curdens, **plotoptions)
        elif self.dim == '2D' or self.dim=='3D':
            plotoptions['title'] = 'Average current density'
            fig = fancyplot(self.time, curdens[:,-1], **plotoptions)
            
            
    elif Type == 'tcA':
        try: plotoptions['title'] # check if title is specified
        except KeyError: plotoptions['title'] = 'Surface concentration vs time'
        if self.dim == '1D':
            fig = fancyplot(self.time, cA[:,0], **plotoptions)
        elif self.dim == '2D':
            
            # update this!
            fig = fancyplot(self.time, cA[:,50,0], **plotoptions)
            
            
    elif Type == 'tcB':
        try: plotoptions['title'] # check if title is specified
        except KeyError: plotoptions['title'] = 'Surface concentration vs time'
        if self.dim == '1D':
            fig = fancyplot(self.time, cB[:,0], **plotoptions)
        elif self.dim == '2D':
            
            # update this!
            fig = fancyplot(self.time, cB[:,50,0], **plotoptions)
            
            
    elif Type == 'tc':
        try: plotoptions['legend'] # check if legend is specified
        except KeyError: plotoptions['legend'] = ['A','B']
        try: plotoptions['title'] # check if title is specified
        except KeyError: plotoptions['title'] = 'Surface concentration vs time'
        if self.dim == '1D':
            fig = fancyplot([[self.time, cA[:,0]],[self.time, cB[:,0]]], **plotoptions)
        elif self.dim == '2D':
            
            # update this!
            fig = fancyplot([[self.time, cA[:,50,0]],[self.time, cB[:,50,0]]], **plotoptions)
            
        elif self.dim == '3D':
            raise NotImplementedError('3D plot not implemented yet!')
   
    if retfig: return fig


def activity(self, tstart=None, tend=None, units=['um','C/cm2'], showhis=False,
             *args, **kwargs):
    '''
    This functions calculates the hotspots of an arbitrary shaped electrode.
    The activity is defined as the total charge transferred.
    
    Paramters:
        tstart: lower bound of the integral in s (float) (default = 0)
        tend: upper bound of the itegral in s (float) (default = t(E = E1))
        units: specify the units of the labels
    
    '''
    if self.dim != '2D': raise ValueError('This function is only available for 2D!')
    
    if tstart is None: tstart = 0
    else: tstart = find_nearest(self.time, tstart)
    if tend is None: tend = self.Eapp.argmin()
    else: tend = find_nearest(self.time, tend)
        
    # integrate the current density to time: charge density = int curdens dt 
    t = self.time[tstart:tend]
    curdens = self.curdens[tstart:tend,:-1]
    chargedens = abs(np.trapz(curdens, t, axis=0))

    # set the charge transfer grid
    chargedensgrid = self.elec*0
    elements = np.array([[],[]])
    for el in self.el: elements = np.append(elements, el,axis=1)
    chargedensgrid[tuple(elements.astype(int))] = chargedens
    
    f =[]
    # check which spatial units to use
    if units[0] == 'um': f.append(1e4)
    elif units[0] == 'mm': f.append(10)
    elif units[0] == 'cm': f.append(1)
    elif units[0] == 'm': f.append(1e-2)
    else: raise ValueError('The first element of units needs to be um, mm, cm or m only!')
    xunit = units[0]
    
    if len(units) != 2:            
        if units[1] == 'um': f.append(1e4)
        elif units[1] == 'mm': f.append(10)
        elif units[1] == 'cm': f.append(1)
        elif units[1] == 'm': f.append(1e-2)
        else: raise ValueError('The second element of units needs to be um, mm, cm or m only!')
        yunit = units[1]
    else: 
        f.append(f[0])
        yunit = xunit
    
    # check which units to use for the concentration
    if units[-1] == 'C/cm2': pass
    elif units[-1] in ['C/mm2', 'uC/um2']: chargedens = chargedens*1e-2
    elif units[-1] == 'C/um2': chargedens = chargedens*1e-8
    elif units[-1] == 'mC/cm2': chargedens = chargedens*1e6
    elif units[-1] == 'mC/um2': chargedens = chargedens*1e-5
    else: raise ValueError('The last element of units needs to be \'C/cm2\', \'C/um2\', \'C/mm2\', \'C/cm2\', \'mC/cm2\' or \'mC/um2\' only!  {} is not allowed!'.format(units[-1]))
    barunit = units[-1].replace('2', '$^2$')
    
    # set the X and Y values
    X,Y = self.meshA
    X,Y = f[0]*X, f[1]*Y
    
    # check the xlim and ylim with the scaled values
    if 'xlim' not in kwargs: kwargs['xlim'] = [0, round(max(X.flatten()))]
    if 'ylim' not in kwargs: kwargs['ylim'] = [0, round(max(Y.flatten()))]
    if 'label' not in kwargs: kwargs['label'] = ['x ('+xunit+')','y ('+yunit+')']
    saveas = kwargs.get('saveas', None)
    if 'saveas' in kwargs: del kwargs['saveas']
    if saveas != None: 
        if 'fontsize' not in kwargs: kwargs['fontsize'] = 10
    barlabel = kwargs.get('barlabel', 'Charge density ('+barunit+')' )
    minbar = kwargs.get('minbar', 0)
    maxbar = kwargs.get('maxbar', round(max(chargedens.flatten())))
    
    # plot the pcolormesh
    fig, ax = fancyplot(**kwargs, returnfig = True)
    colorplot = plt.pcolormesh(X,Y, chargedensgrid, cmap = 'inferno', vmax=minbar, vmin=maxbar) 
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