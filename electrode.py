# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:29:47 2020

@author: bleiji
"""

import numpy as np
import matplotlib.pyplot as plt
import GeneralFunctions as gf
from PIL import Image


def electrode(self, dim, electrode=None, gridtype='linear', extx=False,
              Lxel=None, Lyel=None, Lzel=None):
    '''
    Use electrode() to specify the electrode.
    If 1D:
        use electrode('1D')
    If 2D:
        use electrode('2D', 2D np.array)
    If 3D:
        use electrode('3D', 3D np.array)
        
    Input parameters:
        dim: specify the dimension (1D, 2D, 3D)
        electrode: 2D/3D binary ndarray specifying 1 for electrode, 0 for electrodelyte (ndarray)
        gridtype: linear or non-linear grid (string, default=linear)
        extx: extend the x axis to L = 6*sqrt(D*t_sim) (boolean, default=False)
        Lxel: length of the electrode in x in cm (float)
        Lyel: length of the electrode in y in cm (float)
        Lzel: length of the electrode in z in cm (float)
    '''
    if dim == '1D':
        pass # do nothing
    elif dim == '2D':
        if type(electrode) == type(None):
            self.elec = None # left wall is going to be the electrode
            
        elif electrode.shape > (0,0) and extx: # extrapolate the x direction 
            raise NotImplementedError('No option to extrapolate the electrode yet!')
        
        
        elif electrode.shape > (0,0): # use the electrode dimension as simulation dimensions
            self.elec = electrode
            # use the amount of steps determined by the electrode
            self.y_steps, self.x_steps = electrode.shape
            if Lxel is not None: self.Lx = Lxel
            if Lyel is not None: self.Ly = Lyel
            
            
            
        # electrode should be specified, and the surface should be found
            
            
            
    elif dim == '3D':
        if electrode is None:
            self.elec = None # xz-plane will be the electrode
        elif electrode.shape > (0,0,0):
            self.elec = electrode
            # use the amount of steps determined by the electrode
            self.z_steps, self.y_steps, self.x_steps = electrode.shape
            if Lxel is not None: self.Lx = Lxel
            if Lyel is not None: self.Ly = Lyel
            if Lzel is not None: self.Lz = Lzel
            
        # electrode should be specified, and the surface should be found
        
    else: raise ValueError('dim should be \'1D\', \'2D\' or \'3D\'!')
    self.dim = dim



def insulator(self, dim, block, show=False, *args, **kwargs):
    ''' 
    This functions creates add an insulating material which will not modify
    the concentration. This so called blocking layer blocks the diffusion
    of the ions. So in these regions, no ions can diffusion into.
    
    Input parameters:
        dim: dimension, choose from '2D' or '3D'.
        block: matrix of the blocking layer (2D np.array)
        
    Optional:
        
    '''
    # check the dimension
    if dim == '2D':
        # check if the shape correspond to the shape of the electrode
        if self.elec is None:
            #set the x_steps and y_steps according to the blocking layer array
            self.x_steps, self.y_steps = block.T.shape
        elif block.shape != self.elec.shape:
            raise ValueError('The shape of the bloking layer array does not correspond to the shape of the electrode array! Shapes: {} and {}'.format(block.shape,self.elec.shape))
        
        # set the length of the blocking layer
        if 'Lx' in kwargs and 'Lx' not in dir(self): self.Lx = kwargs['Lx']
        elif 'Lx' not in dir(self): self.Lx = self.x_steps*1e-4
        if 'Ly' in kwargs and 'Ly' not in dir(self): self.Ly = kwargs['Ly']
        elif 'Ly' not in dir(self): self.Ly = self.y_steps*1e-4
        
        
    if dim == '3D':
        # check if the shape corresponds to the shape of the electrode
        if self.elec is None:
            #set the x_steps, y_steps and z_steps according to the blocking layer array
            self.z_steps, self.y_steps, self.x_steps = block.shape
        elif block.shape != self.elec.shape:
            raise ValueError('The shape of the blocking layer array does not correspond to the shape of the electrode array! Shapes: {} and {}'.format(block.shape,self.elec.shape))
    
    self.block = block
    if show:
        x = np.linspace(0,self.Lx*1e4,self.x_steps)
        y = np.linspace(0,self.Ly*1e4,self.y_steps)
        X,Y = np.meshgrid(x,y)
        plt.pcolormesh(X,Y,block, cmap='binary')
        plt.xlabel('x (um)')
        plt.ylabel('y (um)')
        plt.show()
 
    
def electrode_from_image(self, imagepath, show=False, *args, **kwargs):
    '''
    This functions generates an electrode from an image.
    Input: 
        imagepath: path to image (string)
        show: show the image when true (default false)
        x_steps: devide image is x number of pixels (int)
        y_steps: devide image is y number of pixels (int)
        Lx: sets the length of x in cm (float)
        Ly: sets the length of y in cm (float)
    '''
    # import image
    image = Image.open(imagepath).convert('L')
    xl, yl = list(image.size) # extract the number of pixels
    
    # set the number of steps equal to the number of pixels by default
    if 'x_steps' in kwargs: self.x_steps = kwargs['x_steps']
    else: self.x_steps = xl 
    if 'y_steps' in kwargs: self.y_steps = kwargs['y_steps']
    else: self.y_steps = yl
    
    # set the length of the electrode, length is equal to the pixels in tens of um
    if 'Lxel' in kwargs: Lx = kwargs['Lxel']
    else: Lx = xl*1e-3 # in cm
    if 'Lyel' in kwargs: Ly = kwargs['Lyel']
    else: Ly = yl*1e-3 # in cm

    # reshape the image to the given amount of x and y steps
    image = np.asarray(image.resize((self.x_steps-1,self.y_steps)))
    image = np.where(image/255 == 1,0,1)
    electrode_image = (np.append(image.T,[np.zeros(self.y_steps)],axis=0).T)[::-1]
    self.electrode('2D',electrode=electrode_image, Lxel=Lx, Lyel=Ly)
    
    if show:
        x = np.linspace(0,Lx*10,self.x_steps)
        y = np.linspace(0,Ly*10,self.y_steps)
        X,Y = np.meshgrid(x,y)
        surf_image = self.findsurface(electrode_image, retsurf=True)
        plt.pcolormesh(X,Y,surf_image[-1], cmap='binary')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.title(imagepath)
        plt.show()