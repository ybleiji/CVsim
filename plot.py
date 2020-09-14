# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:42:57 2020

@author: bleiji
"""

# import numpy as np
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
    if Type not in ['CV', 'CV all', 'tE', 'tI', 'tcA', 'tcB', 'tc', 'te', 'ti', 'tca', 'tcb']:
        raise ValueError('Type should be one of the following: \'CV\', \'CV all\', \'tE\', \'tI\', \'tcA\', \'tcB\' or \'tc\'!')
    
    # copy the kwargs and remove the keys that are not used in the plot
    plotoptions = kwargs.copy()
    try: del plotoptions['units'] # remove keys which are not used in the plot
    except KeyError: pass

    # check the units
    if Type == 'CV' or Type == 'CV all' or Type == 'tI' or Type == 'ti':
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
    elif Type == 'tcA' or Type == 'tcB' or Type == 'tca' or Type == 'tcb' or Type == 'tc':
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
        # this is for ploting the current versus the potential
        # only the average current density will be shown, this is the closest the the experiment
        plotoptions['label'] = ['Potential (V)', 'Current density '+units]
        if self.dim == '1D': 
            fig = fancyplot(self.Eapp, curdens, **plotoptions)
        elif self.dim == '2D' or self.dim=='3D':
            plotoptions['title'] = 'Average current density'
            plotoptions['color'] = 'r' 
            fig = fancyplot(self.Eapp, curdens[:,-1], **plotoptions)
            
    if Type == 'CV all':
        # plotting all CV's of all elements together with its averaged value
        plotoptions['label'] = ['Potential (V)', 'Current density '+units]
        if self.dim == '1D': 
            fig = fancyplot(self.Eapp, curdens, **plotoptions)
        elif self.dim == '2D' or self.dim=='3D':
            plotoptions['title'] = 'Average current density'
            plotoptions['color'] = 'r'
            plotoptions['showfig'] = False
            plt.plot(self.Eapp, curdens[:,:-1], 'gray')
            plt.plot(self.Eapp, curdens[:,-1], 'r')
            fig = fancyplot(**plotoptions)
            
    elif Type == 'tE' or Type == 'te':
        # plot the time vs potential
        plotoptions['label'] = ['Time (s)', 'Potential (V)']
        fig = fancyplot(self.time, self.Eapp, **plotoptions)
        
    elif Type == 'tI' or Type == 'ti':
        # plot the average current density vs time
        plotoptions['label'] = ['Time (s)', 'Current density '+units]
        if self.dim == '1D': fig = fancyplot(self.time, curdens, **plotoptions)
        elif self.dim == '2D' or self.dim=='3D':
            plotoptions['title'] = 'Average current density'
            fig = fancyplot(self.time, curdens[:,-1], **plotoptions)
            
    elif Type == 'tcA' or Type == 'tca':
        # plot the concentration of A vs "the distance from the electrode"
        try: plotoptions['title'] # check if title is specified
        except KeyError: plotoptions['title'] = 'Surface concentration of A vs time'
        if self.dim == '1D': fig = fancyplot(self.time, cA[:,0], **plotoptions)
        elif self.dim == '2D': tcplot2D(self.time,self.meshA, cA, 'A', *args, **kwargs)
        elif self.dim == '3D': tcplot3D(self.time, self.meshA, cA, 'A', *args, **kwargs)
            
    elif Type == 'tcB' or Type == 'tcb':
        # plot the concentration of B vs "the distance from the electrode"
        try: plotoptions['title'] # check if title is specified
        except KeyError: plotoptions['title'] = 'Surface concentration of B vs time'
        if self.dim == '1D': fig = fancyplot(self.time, cB[:,0], **plotoptions)
        elif self.dim == '2D': tcplot2D(self.time,self.meshA, cB, 'B', *args, **kwargs)
        elif self.dim == '3D': tcplot3D(self.time, self.meshB, cB, 'B', *args, **kwargs)  
            
    elif Type == 'tc':
        # plot both the concentrations of A and B vs the "distance from the electrode"
        try: plotoptions['legend'] # check if legend is specified
        except KeyError: plotoptions['legend'] = ['A','B']
        try: plotoptions['title'] # check if title is specified
        except KeyError: plotoptions['title'] = 'Surface concentration vs time'
        if self.dim == '1D':
            fig = fancyplot([[self.time, cA[:,0]],[self.time, cB[:,0]]], **plotoptions)
        elif self.dim == '2D':
            if len(args) == 0:
                raise ValueError('The argument should be a tuple specifying which pixel to show!')
            if type(args[0]) == tuple:
                x0,y0 = args[0]
                X,Y = self.meshA
                x,y = X[0,:],  Y[:,0]
                # auto units: nm, um or cm (standard)
                if x[-1] < 0.1: x, unitsx = x*1e4, '$\\mu m$' # convert to um
                elif 0.1 <= x[-1] < 10: x, unitsx = x*10, 'mm' # convert to mm
                else: unitsx = 'cm' # default
                if y[-1] < 0.1: y, unitsy = y*1e4, '$\\mu m$' # convert to um
                elif 0.1 <= y[-1] < 10: y, unitsy = y*10, 'mm' # convert to mm
                else: unitsy = 'cm' # default
                
                plotoptions['label'] = ['Concentration (mM)', 'Time (s)']
                plotoptions['title'] = 'x$_0$: {:.1f} {}, y$_0$: {:.1f} {}'.format(x[x0],unitsx,y[y0],unitsy)
                plotoptions['style'] = ['r', 'k']
                fig = fancyplot([[self.time, cA[:,y0,x0]],[self.time, cB[:,y0,x0]]], **plotoptions)
        elif self.dim == '3D':
            if len(args) == 0:
                raise ValueError('The argument should be a tuple specifying which pixel to show!')
            if type(args[0]) == tuple:
                x0,y0,z0 = args[0]
                X,Y,Z = self.meshA
                x,y,z = X[0,:,0],  Y[:,0,0], Z[0,0,:]
                # auto units: nm, um or cm (standard)
                if x[-1] < 0.1: x, unitsx = x*1e4, '$\\mu m$' # convert to um
                elif 0.1 <= x[-1] < 10: x, unitsx = x*10, 'mm' # convert to mm
                else: unitsx = 'cm' # default
                if y[-1] < 0.1: y, unitsy = y*1e4, '$\\mu m$' # convert to um
                elif 0.1 <= y[-1] < 10: y, unitsy = y*10, 'mm' # convert to mm
                else: unitsy = 'cm' # default
                if z[-1] < 0.1: z, unitsz = z*1e4, '$\\mu m$' # convert to um
                elif 0.1 <= z[-1] < 10: z, unitsz = z*10, 'mm' # convert to mm
                else: unitsz = 'cm' # default
                
                plotoptions['label'] = ['Concentration (mM)', 'Time (s)']
                plotoptions['title'] = 'x$_0$: {:.1f} {}, y$_0$: {:.1f} {}, z$_0$: {:.1f} {}'.format(x[x0],unitsx,y[y0],unitsy,z[z0],unitsz)
                plotoptions['style'] = ['r', 'k']
                fig = fancyplot([[self.time, cA[:,z0,y0,x0]],[self.time, cB[:,z0,y0,x0]]], **plotoptions)
            else: raise TypeError('One should give the indices of the location to be plotten using a tuple: (0,2,1) for example.')
   
    if retfig: return fig

def tcplot2D(time, mesh, conc, AorB, *args, **kwargs):
    '''
    This function will be used for plotting a 2D plot of the concentration over time
    '''
    saveas = kwargs.get('saveas', None)
    
    X,Y = mesh
    x,y = X[0,:],  Y[:,0]
    # auto units: nm, um or cm (standard)
    if x[-1] < 0.1: x, unitsx = x*1e4, '$\\mu m$' # convert to um
    elif 0.1 <= x[-1] < 10: x, unitsx = x*10, 'mm' # convert to mm
    else: unitsx = 'cm' # default
    if y[-1] < 0.1: y, unitsy = y*1e4, '$\\mu m$' # convert to um
    elif 0.1 <= y[-1] < 10: y, unitsy = y*10, 'mm' # convert to mm
    else: unitsy = 'cm' # default
        
    x0, y0 = 0, int(len(y)/2)
    options = {'grid': False,
               'origin': False,
               'returnfig': True,
               'saveas': None}
    
    if len(args) == 0 or args[0] == 'x': # x slice is the default
        plot2D = True
        if len(args) > 1: y0 = args[1]
        options['title'] = 'y$_0$: {:.1f} {}'.format(y[y0],unitsy)
        options['label'] = ['Time (s)', 'Position x ('+unitsx+')']
        fig, ax = fancyplot(**options)
        quad = ax.pcolormesh(time, x, conc[:, y0, :].T, cmap='inferno')
    elif args[0] == 'y':
        plot2D = True
        if len(args) > 1: x0 = args[1]
        options['title'] = 'x$_0$: {:.1f} {}'.format(x[x0],unitsx)
        options['label'] = ['Time (s)', 'Position y ('+unitsy+')']
        fig, ax = fancyplot(**options)
        quad = ax.pcolormesh(time, y, conc[:, :, x0].T, cmap='inferno')
    elif type(args[0]) is tuple:
        plot2D = False
        x0,y0 = args[0]
        fancyplot(time, conc[:,y0,x0], 
                  label= ['Time (s)', 'Concentration of '+AorB+' (mM)'], 
                  title = 'x$_0$: {:.1f} {}, y$_0$: {:.1f} {}'.format(x[x0],unitsx,y[y0],unitsy))
        if saveas != None: plt.savefig(fname=saveas, dpi=300, bbox_inches='tight')
        plt.show()
    else: raise ValueError('The only valid inputs are: \'x\', \'y\' or a postion in the 2D matrix (x,y)')
    
    if plot2D:
        bar = fig.colorbar(quad)
        bar.set_label(label='Concentration of '+AorB+' (mM)') #add a label to the colorbar
        if saveas != None: plt.savefig(fname=saveas, dpi=300, bbox_inches='tight')
        plt.show()

def tcplot3D(time, mesh, conc, AorB, *args, **kwargs):
    '''
    This function will be used for plotting a 2D plot of the concentration over time
    '''
    saveas = kwargs.get('saveas', None)
    
    X,Y,Z = mesh
    x,y,z = X[0,:,0],  Y[:,0,0], Z[0,0,:]
    # auto units: nm, um or cm (standard)
    if x[-1] < 0.1: x, unitsx = x*1e4, '$\\mu m$' # convert to um
    elif 0.1 <= x[-1] < 10: x, unitsx = x*10, 'mm' # convert to mm
    else: unitsx = 'cm' # default
    if y[-1] < 0.1: y, unitsy = y*1e4, '$\\mu m$' # convert to um
    elif 0.1 <= y[-1] < 10: y, unitsy = y*10, 'mm' # convert to mm
    else: unitsy = 'cm' # default
    if z[-1] < 0.1: z, unitsz = z*1e4, '$\\mu m$' # convert to um
    elif 0.1 <= z[-1] < 10: z, unitsz = z*10, 'mm' # convert to mm
    else: unitsz = 'cm' # default
        
    x0, y0, z0 = 0, int(len(y)/2), int(len(z)/2)
    options = {'grid': False,
               'origin': False,
               'returnfig': True,
               'saveas': None}
    
    if len(args) == 0 or args[0] == 'x': # x slice is the default
        plot2D = True
        if len(args) > 1: y0 = args[1]
        if len(args) > 2: z0 = args[2]
        options['title'] = 'y$_0$: {:.1f} {}, z$_0$: {:.1f} {}'.format(y[y0],unitsy,z[z0],unitsz)
        options['label'] = ['Time (s)', 'Position x ('+unitsx+')']
        fig, ax = fancyplot(**options)
        quad = ax.pcolormesh(time, x, conc[:, z0, y0, :].T, cmap='inferno')
    elif args[0] == 'y':
        plot2D = True
        if len(args) > 1: x0 = args[1]
        if len(args) > 2: z0 = args[2]
        options['title'] = 'x$_0$: {:.1f} {}, z$_0$: {:.1f} {}'.format(x[x0],unitsx,z[z0],unitsz)
        options['label'] = ['Time (s)', 'Position y ('+unitsy+')']
        fig, ax = fancyplot(**options)
        quad = ax.pcolormesh(time, y, conc[:, z0, :, x0].T, cmap='inferno')
    elif args[0] == 'z':
        plot2D = True
        if len(args) > 1: x0 = args[1]
        if len(args) > 2: y0 = args[2]
        options['title'] = 'x$_0$: {:.1f} {}, y$_0$: {:.1f} {}'.format(x[x0],unitsx,y[y0],unitsy)
        options['label'] = ['Time (s)', 'Position z ('+unitsz+')']
        fig, ax = fancyplot(**options)
        quad = ax.pcolormesh(time, z, conc[:, :, y0, x0].T, cmap='inferno')
    elif type(args[0]) is tuple:
        plot2D = False
        x0,y0,z0 = args[0]
        fancyplot(time, conc[:,z0,y0,x0], 
                  label= ['Time (s)', 'Concentration of '+AorB+' (mM)'], 
                  title = 'x$_0$: {:.1f} {}, y$_0$: {:.1f} {}, z$_0$: {:.1f} {}'.format(x[x0],unitsx,y[y0],unitsy,z[z0],unitsz))
        if saveas != None: plt.savefig(fname=saveas, dpi=300, bbox_inches='tight')
        plt.show()
    else: raise ValueError('The only valid inputs are: \'x\', \'y\', \'z\' or a postion in the 3D matrix (x,y,z)')
    
    if plot2D:
        bar = fig.colorbar(quad)
        bar.set_label(label='Concentration of '+AorB+' (mM)') #add a label to the colorbar
        if saveas != None: plt.savefig(fname=saveas, dpi=300, bbox_inches='tight')
        plt.show()

