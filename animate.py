# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:21:20 2020

@author: bleiji
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.constants as const
const.F = const.physical_constants["Faraday constant"][0]
import GeneralFunctions as gf



def animate(self, *args, **kwargs):
    '''
    This is the general function animate. Call .animate to show the animation.
    Depending on the dimension, the animate_1D(), animate_2D() or animate_3D()
    fuctions will be called.
    '''
    if self.dim == '1D':
        anim = self.animate_1D(**kwargs)
    elif self.dim == '2D':
        anim = self.animate_2D(**kwargs)
    elif self.dim == '3D':
        anim = self.animate_3D(**kwargs)
    
    return anim
     
   

def animate_1D(self, *args, **kwargs):
    '''
    This function shows the animation of the concentraion of A and B over time.
    This function makes use of the animate() function from GeneralFunctions.
    '''
    kwargs['t_an'] = self.time
    kwargs['E_app'] = self.Eapp
    if 'color' not in kwargs: kwargs['color'] = ['r','k']
    if 'marker' not in kwargs: kwargs['marker'] = 'o'
    if 'legend' not in kwargs: kwargs['legend'] = ['[A]','[B]']
    if 'imagesize' not in kwargs: kwargs['imagesize'] = (7,5)
    units = kwargs.get('units',['um','mM'])
    
    # check which spatial units to use
    if units[0] == 'um': x = [1e4*self.xA, 1e4*self.xB]
    elif units[0] == 'mm': x = [10*self.xA, 10*self.xB]
    elif units[0] == 'cm': x = [self.xA, self.xB]
    elif units[0] == 'm': x = [1e-2*self.xA, 1e-2*self.xB]
    else: raise ValueError('The first element of units needs to be um, mm, cm or m only!')
    
    # check which units to use for the concentration
    if units[1] == 'mM': c = [1e3*self.cA, 1e3*self.cB]
    elif units[1] == 'M': c = [self.cA, self.cB]
    else: raise ValueError('The second element of units needs to be mM or M only!')
    
    # check the xlim and ylim with the scaled values
    if 'xlim' not in kwargs: kwargs['xlim'] = [0, max(x[0])/4]
    if 'ylim' not in kwargs: kwargs['ylim'] = [0, 1.1*max(np.array(c).flatten())]
    if 'label' not in kwargs: kwargs['label'] = ['Distance from electrode ('+units[0]+')','Concentration ('+units[1]+')']
    
    anim = gf.animate(x, c, **kwargs)
    return anim

   

def animate_2D(self, *args, **kwargs):
    '''
    This function shows the animation of the concentraion of A and B over time.
    This function makes use of the animate_2D() function from GeneralFunctions.
    
    Parameters:
        see GeneralFunctions.animate_2D()
        
    Additional:
        species: specify which species to animate 'A' or 'B'
        units: specify the units of the x,y and label. (default = ['um','mM'])
    
    '''
    kwargs['t_an'] = self.time # specify the time steps
    kwargs['E_app'] = self.Eapp # specify the potential steps
    if 'imagesize' not in kwargs: kwargs['imagesize'] = (7,5) # set imagesize
    if 'cmap' not in kwargs: kwargs['cmap'] = 'inferno' # set color map
    if 'frames' not in kwargs: kwargs['frames'] = len(self.time)
    units = kwargs.get('units',['um','mM'])
    species = kwargs.get('species', 'A') # select which specices to animate
    if 'species' in kwargs: del kwargs['species']
    if 'units' in kwargs: del kwargs['units']
    
    # select the c and mesh of the species:
    if species == 'A': # select the mesh and c for species A
        mesh = self.meshA
        c = self.cA
    elif species == 'B': # select the mesh and c for species B
        mesh = self.meshB
        c = self.cB
    else: raise ValueError('Choose between the species \'A\' and \'B\' only!')

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
    x = [f[0]*mesh[0], f[1]*mesh[1]]
    
    # check which units to use for the concentration
    if units[-1] == 'mM': c = 1e3*c
    elif units[-1] == 'M': pass
    else: raise ValueError('The last element of units needs to be mM or M only!')
    
    # check the xlim and ylim with the scaled values
    if 'xlim' not in kwargs: kwargs['xlim'] = [0, round(max(np.array(x[0]).flatten()))]
    if 'ylim' not in kwargs: kwargs['ylim'] = [0, round(max(np.array(x[1]).flatten()))]
    if 'minbar' not in kwargs: kwargs['minbar'] = 0
    if 'maxbar' not in kwargs: kwargs['maxbar'] = round(max(c.flatten()))
    if 'label' not in kwargs: kwargs['label'] = ['x ('+xunit+')','y ('+yunit+')']
    if 'barlabel' not in kwargs: kwargs['barlabel'] = 'Concentration ('+units[-1]+')'
    if 'outline' not in kwargs: kwargs['outline'] =  self.el
    
    # animate!
    anim = gf.animate_2D(x, c, **kwargs)
    return anim # animations needs to be returned in order to be seen in ipython

def animate_3D(self, x='all', y='all', z='all', *args, **kwargs):
    '''
    This function shows the animation of the concentraion of A and B over time.
    It shows the evolution of the concentration in a cross section of the 3D simulation space.
    Determine said cross section by specifying the value of ONE coordinate for
    which the cross section (planar) should be taken.
    This function makes use of the animate_2D() function from GeneralFunctions.
    
    Parameters:
        One of x, y or z to determine the cross section plane.
        Otherwise see GeneralFunctions.animate_2D()
        
    Additional:
        species: specify which species to animate 'A' or 'B'
        units: specify the units of the x,y and label. (default = ['um','mM'])
        depth unit: unit of the coordinate position of the cross section
    '''
    if [x,y,z].count('all') != 2:
        raise ValueError('Exactly one coordinate must be specified for the cross section!')
    
    kwargs['t_an'] = self.time # specify the time steps
    kwargs['E_app'] = self.Eapp # specify the potential steps
    if 'imagesize' not in kwargs: kwargs['imagesize'] = (7,5) # set imagesize
    if 'cmap' not in kwargs: kwargs['cmap'] = 'inferno' # set color map
    if 'frames' not in kwargs: kwargs['frames'] = len(self.time)
    units = kwargs.get('units',['um','mM'])
    depth_unit = kwargs.get('depth unit','um')
    species = kwargs.get('species', 'A') # select which species to animate
    if 'species' in kwargs: del kwargs['species']
    if 'units' in kwargs: del kwargs['units']
    if 'depth unit' in kwargs: del kwargs['depth unit']
    
    # select the c and mesh of the species:
    if species == 'A': # select the mesh and c for species A
        if x != 'all':
            yval = self.meshA[1][:,0,0]
            zval = self.meshA[2][0,0]
            mesh = np.meshgrid(yval,zval)
            c = self.cA[:,:,:,x]
            title_3D_dist = (self.meshA[0][0,1,0]-self.meshA[0][0,0,0])*x
        elif y != 'all':
            xval = self.meshA[0][0,:,0]
            zval = self.meshA[2][0,0]
            mesh = np.meshgrid(xval,zval)
            c = self.cA[:,:,y,:]
            title_3D_dist = (self.meshA[1][1,0,0]-self.meshA[1][0,0,0])*y
        elif z != 'all':
            xval = self.meshA[0][0,:,0]
            yval = self.meshA[1][:,0,0]
            mesh = np.meshgrid(xval,yval)
            c = self.cA[:,z,:,:]
            title_3D_dist = (self.meshA[2][0,0,1]-self.meshA[2][0,0,0])*z
    elif species == 'B': # select the mesh and c for species B
        if x != 'all':
            yval = self.meshB[1][:,0,0]
            zval = self.meshB[2][0,0]
            mesh = np.meshgrid(yval,zval)
            c = self.cB[:,:,:,x]
            title_3D_dist = (self.meshB[0][0,1,0]-self.meshB[0][0,0,0])*x
        elif y != 'all':
            xval = self.meshB[0][0,:,0]
            zval = self.meshB[2][0,0]
            mesh = np.meshgrid(xval,zval)
            c = self.cB[:,:,y,:]
            title_3D_dist = (self.meshB[1][1,0,0]-self.meshB[1][0,0,0])*y
        elif z != 'all':
            xval = self.meshB[0][0,:,0]
            yval = self.meshB[1][:,0,0]
            mesh = np.meshgrid(xval,yval)
            c = self.cB[:,z,:,:]
            title_3D_dist = (self.meshB[2][0,0,1]-self.meshB[2][0,0,0])*z
    else: raise ValueError('Choose between the species \'A\' and \'B\' only!')

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
    coord = [f[0]*mesh[0], f[1]*mesh[1]]
    
    if depth_unit == 'um': kwargs['title 3D'] = ', {:.3f}'.format(title_3D_dist*1e4)+' um'
    elif depth_unit == 'mm': kwargs['title 3D'] = ', {:.3f}'.format(title_3D_dist*10)+' mm'
    elif depth_unit == 'cm': kwargs['title 3D'] = ', {:.3f}'.format(title_3D_dist*1)+' cm'
    elif depth_unit == 'm': kwargs['title 3D'] = ', {:.3f}'.format(title_3D_dist*1e-2)+' m'
    else: raise ValueError('The depth unit needs to be um, mm, cm or m only!')
    
    # check which units to use for the concentration
    if units[-1] == 'mM': c = 1e3*c
    elif units[-1] == 'M': pass
    else: raise ValueError('The last element of units needs to be mM or M only!')
    
    # check the xlim and ylim with the scaled values
    if 'xlim' not in kwargs: kwargs['xlim'] = [0, round(max(np.array(coord[0]).flatten()))]
    if 'ylim' not in kwargs: kwargs['ylim'] = [0, round(max(np.array(coord[1]).flatten()))]
    if 'minbar' not in kwargs: kwargs['minbar'] = 0
    if 'maxbar' not in kwargs: kwargs['maxbar'] = round(max(c.flatten()))
    if 'label' not in kwargs: kwargs['label'] = ['x ('+xunit+')','y ('+yunit+')']
    if 'barlabel' not in kwargs: kwargs['barlabel'] = 'Concentration ('+units[-1]+')'
    if 'outline' not in kwargs:
        empty_inv = list(np.delete(range(6),self.empty))
        outline = []
        length = 0
        
        if x !='all':
            for i,el in enumerate(self.el):
                if empty_inv[i] == 0 or empty_inv[i] == 1: continue
                temp = el[:,el[2]==x]
                pos = temp[(0,1),:]
                outline.append(pos)
                length = length + len(pos.flatten())
            if length==0: print('The specified cross section does not include an electrode surface!')
            else: kwargs['outline'] = outline
            
        if y !='all':
            for i,el in enumerate(self.el):
                if empty_inv[i] == 2 or empty_inv[i] == 3: continue
                temp = el[:,el[1]==y]
                pos = temp[(0,2),:]
                outline.append(pos)
                length = length + len(pos.flatten())
            if length==0: print('The specified cross section does not include an electrode surface!')
            else:kwargs['outline'] = outline
            
        if z !='all':
            for i,el in enumerate(self.el):
                if empty_inv[i] == 4 or empty_inv[i] == 5: continue
                temp = el[:,el[0]==z]
                pos = temp[(1,2),:]
                outline.append(pos)
                length = length + len(pos.flatten())
            if length==0: print('The specified cross section does not include an electrode surface!')
            else:kwargs['outline'] = outline
    
    # animate!
    anim = gf.animate_2D(coord, c, **kwargs)
    return anim # animations needs to be returned in order to be seen in ipython    
