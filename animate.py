# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:21:20 2020

@author: bleiji
"""

import numpy as np
import scipy.constants as const
const.F = const.physical_constants["Faraday constant"][0]
from .genfunc import Animate, Animate_2D



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
    This function makes use of the animate() function from genfunc.
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
    
    anim = Animate(x, c, **kwargs)
    return anim
   

def animate_2D(self, *args, **kwargs):
    '''
    This function shows the animation of the concentraion of A and B over time.
    This function makes use of the animate_2D() function from genfunc.
    
    Parameters:
        see genfunc.animate_2D()
        
    Additional:
        species: specify which species to animate 'A' or 'B'
        units: specify the units of the x,y and label. (default = ['um','mM'])
        resolution: specify the resolution of the animation in %. When 100% (default) all pixels are shown
    
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
    resolution = kwargs.get('resolution', 100)
    if 'resolution' in kwargs: del kwargs['resolution']
    
    # select the c and mesh of the species:
    if species == 'A' or species == 'a': # select the mesh and c for species A
        mesh = self.meshA
        c = self.cA
    elif species == 'B' or species == 'b': # select the mesh and c for species B
        mesh = self.meshB
        c = self.cB
    else: raise ValueError('Choose between the species \'A\' and \'B\' only!')
    
    elec = self.el # positions of the electrode
    
    # the mesh can be quite large sometimes, it is not always nessesary to display all pixels
    if resolution < 100:
        idx0 = np.round(np.linspace(0, mesh[0].shape[0] - 1, round(resolution/100 * mesh[0].shape[0]))).astype(int) # zeroth dim to shrink
        idx1 = np.round(np.linspace(0, mesh[0].shape[1] - 1, round(resolution/100 * mesh[0].shape[1]))).astype(int) # first dim to shrink
        mesh_LQ = mesh.copy()
        mesh_LQ[0], mesh_LQ[1] = mesh_LQ[0][idx0][:,idx1], mesh_LQ[1][idx0][:,idx1]
        c_LQ = c.copy()
        c_LQ = c[:,idx0][:,:,idx1]
        c, mesh = c_LQ, mesh_LQ
        if elec != []:
            elec_LQ = np.array(np.round(elec.copy() * resolution/100-0.5)).astype(int) # shrink down the electrode pos
            elec = np.unique(elec_LQ, axis=1) # remove all the duplicates

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
    if 'outline' not in kwargs: kwargs['outline'] =  elec
    
    # animate!
    anim = Animate_2D(x, c, **kwargs)
    return anim # animations needs to be returned in order to be seen in ipython

def animate_3D(self, x='all', y='all', z='all', *args, **kwargs):
    '''
    This function shows the animation of the concentraion of A and B over time.
    It shows the evolution of the concentration in a cross section of the 3D simulation space.
    Determine the cross section by specifying the value of ONE coordinate for
    which the cross section (planar) should be taken.
    This function makes use of the animate_2D() function from genfunc.
    
    Parameters:
        One of x, y or z to determine the cross section plane. (x=0 for example)
        Otherwise see genfunc.animate_2D()
        
    Additional:
        species: specify which species to animate 'A' or 'B'
        units: specify the units of the x,y and label. (default = ['um','mM'])
        depth unit: unit of the coordinate position of the cross section
        resolution: specify the resolution of the animation in %. When 100% (default) all pixels are shown
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
    resolution = kwargs.get('resolution', 100)
    if 'resolution' in kwargs: del kwargs['resolution']
    
    # select the c and mesh of the species:
    if species == 'A': # select the mesh and c for species A
        mesh, c = self.meshA, self.cA
    elif species == 'B': # select the mesh and c for species B
        mesh, c = self.meshB, self.cB
    else: raise ValueError('Choose between species \'A\' or \'B\'!')
    
    # create mesh and c in 2D
    if x != 'all':
        yval, zval = mesh[1][:,0,0], mesh[2][0,0]
        title_3D_dist = (mesh[0][0,1,0]-mesh[0][0,0,0])*x
        mesh, c = np.meshgrid(yval,zval), c[:,:,:,x]
        titlestr = 'x'
    elif y != 'all':
        xval, zval = mesh[0][0,:,0], mesh[2][0,0]
        title_3D_dist = (mesh[1][1,0,0]-mesh[1][0,0,0])*y
        mesh, c = np.meshgrid(xval,zval), c[:,:,y,:]
        titlestr = 'y'
    elif z != 'all':
        xval, yval = mesh[0][0,:,0], mesh[1][:,0,0]
        title_3D_dist = (mesh[2][0,0,1]-mesh[2][0,0,0])*z
        mesh, c = np.meshgrid(xval,yval), c[:,z,:,:]
        titlestr = 'z'
        
    elec = self.el
    if 'outline' not in kwargs: # select the outline
        if x != 'all':
            select_el = elec[2] == x # select the x value
            elec = np.array((elec[0][select_el], elec[1][select_el]))
        elif y != 'all':
            select_el = elec[1] == y # select the y value
            elec = np.array((elec[0][select_el], elec[2][select_el]))
        elif z != 'all':
            select_el = elec[0] == z # select the z value
            elec = np.array((elec[1][select_el], elec[2][select_el]))
        kwargs['outline'] = elec
    
    # the mesh can be quite large sometimes, it is not always nessesary to display all pixels
    if resolution < 100:
        idx0 = np.round(np.linspace(0, mesh[0].shape[0] - 1, round(resolution/100 * mesh[0].shape[0]))).astype(int) # zeroth dim to shrink
        idx1 = np.round(np.linspace(0, mesh[0].shape[1] - 1, round(resolution/100 * mesh[0].shape[1]))).astype(int) # first dim to shrink
        mesh_LQ = mesh.copy()
        mesh_LQ[0], mesh_LQ[1] = mesh_LQ[0][idx0][:,idx1], mesh_LQ[1][idx0][:,idx1]
        c_LQ = c.copy()
        c_LQ = c[:,idx0][:,:,idx1]
        c, mesh = c_LQ, mesh_LQ
        if elec != []:
            elec_LQ = np.array(np.round(elec.copy() * resolution/100-0.5)).astype(int) # shrink down the electrode pos
            elec = np.unique(elec_LQ, axis=1) # remove all the duplicates
            kwargs['outline'] = elec
        
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
    
    if depth_unit == 'um': kwargs['title 3D'] = ', {} = {:.0f}'.format(titlestr, title_3D_dist*1e4)+' um'
    elif depth_unit == 'mm': kwargs['title 3D'] = ', {} = {:.0f}'.format(titlestr, title_3D_dist*10)+' mm'
    elif depth_unit == 'cm': kwargs['title 3D'] = ', {} = {:.0f}'.format(titlestr, title_3D_dist*1)+' cm'
    elif depth_unit == 'm': kwargs['title 3D'] = ', {} = {:.0f}'.format(titlestr, title_3D_dist*1e-2)+' m'
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
    if 'barlabel' not in kwargs: kwargs['barlabel'] = 'Concentration ('+units[-1]+')'
    
    if 'label' not in kwargs: # set the label
        if x != 'all': kwargs['label'] = ['y ('+xunit+')','z ('+yunit+')']
        elif y != 'all': kwargs['label'] = ['x ('+xunit+')','z ('+yunit+')']
        elif z != 'all': kwargs['label'] = ['x ('+xunit+')','y ('+yunit+')']
    
    # animate!
    anim = Animate_2D(coord, c, **kwargs)
    return anim # animations needs to be returned in order to be seen in ipython    
