    # -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:39:03 2020

@author: bleiji
"""
import numpy as np
from itertools import combinations


def findsurface(self, electrode, retsurf=False):
    '''
    This function finds the surface of the electrode. It returns the
    elements of the surface of the electrode and the normal0 vectors corresponding
    to the electrode elements. When retsurf=True, it returns an array with 0 
    for no electrode, 1 for electrode, and 2 for the surface of the electrode.
    This is for displaying the surface.
    
    Input parameters:
        dim: dimension, either '2D' or '3D'
        electrode: array containing 0 for no electrode and 1 for electrode (2D or 3D ndarray)
        (optional) retsurf: set the surfgace of the electrode to the value 2 in the electrode array (default = False)
    
    Output parameters:
        elec0: list with the positions of the surface of the electrode (2D ndarray)
        normal0: normal0 vector to the surface of the electrode (2D ndarray)
        (optional) electrode_surf: array with 0 if electrolyte, 1 if electrode, 
                   2 if surface, 3 if corner element (2D or 3D ndarray)
    '''
    if self.dim == '2D':
        posxpos, posxneg = [], []
        posypos, posyneg = [], []
        
        # along y
        for x,row in enumerate(electrode.T): # choose x coordinate
            prev = 0
            for y,el in enumerate(row): # go along y coordinate
                if el and not prev:
                    prev = 1
                    if y > 0: posyneg.append([y-1,x]) 
                if not el and prev:
                    prev = 0
                    posypos.append([y,x])
                    
        # along x
        for y,row in enumerate(electrode): # choose y coordinategg
            prev = 0
            for x,el in enumerate(row): # go along x coordinate
                if el and not prev:
                    prev = 1
                    if x > 0: posxneg.append([y,x-1])
                if not el and prev:
                    prev = 0
                    posxpos.append([y,x])
        
        posxpos, posypos = np.array(posxpos).T, np.array(posypos).T
        posxneg, posyneg = np.array(posxneg).T, np.array(posyneg).T
        
        el_surf = [posxpos, posxneg, posypos, posyneg] # list with surface positions
        if el_surf == []: raise ValueError('The electrode was not correctly specified!')
        
        # before looping, filter out the empty elements:
        empty = [idx for idx in range(len(el_surf)) if el_surf[idx].size==0]
        if empty != []: el_surf = np.delete(el_surf,empty)
        
        # specify the surface elements
        displace = np.array([[0,0,-1,1],[-1,1,0,0]]).T.reshape((4,2,1))
        displace = np.delete(displace, empty, axis=0)
        elec0 = [el_surf[i] + displace[i] for i in range(len(el_surf))] # list with electrode positions
        
        # find the normal to the surface
        normalvec = np.delete(np.array([[[0,1]],[[0,-1]],[[1,0]],[[-1,0,]]], dtype=int), empty, axis=0) # dy, dx
        cornerel, normal0 = np.array([[],[]]), []
        for i, el in enumerate(elec0): # search within elec0 for corner elem
            cornerel = np.concatenate((cornerel, el), axis=1)
            normal0.append(el*0+normalvec[i].T)
        uniq, count = np.unique(cornerel.T, return_counts=True, axis=0)
        cornerel = uniq[count > 1].T.astype(np.int) # here the positions of the corner elements in the 2D matrix are being saved 
         
        # merge elec and normal to one 1D ndarray
        elec, normal =  elec0[0], normal0[0]
        for i in np.arange(1,len(elec0)):
            elec = np.concatenate((elec, elec0[i]), axis=1)
            normal = np.concatenate((normal, normal0[i]), axis=1)    
            
        # sort the elec and normal arrays
        elec, normal = elec.T, normal.T
        for i in [1,0]:
            sortel = elec[:,i].argsort(kind='mergesort')
            elec, normal = elec[sortel], normal[sortel]
        
        # there are 4 situations, the elements of the surf can border to:
        # 1 element:  just x or y dir
        # 2 elements: corner element, flux will be the sum of the two corrected by the normal0 vector
        # 3 elements: only the unique direction should survive, so (1,0) + (0,1) + (-1,0) sums up to (0,1)
        # 4 elements: no direction survices, there is no electrode...       
        uniq, idx_uniq, count = np.unique(elec, axis=0, return_counts=True, return_index=True)
        dup = np.where(count>1)[0] # select only the duplicates
        for el, cnt in zip(uniq[dup],count[dup]): # find all elements which are duplicates
            find = np.intersect1d(np.where(elec[:,0]==el[0]), np.where(elec[:,1]==el[1]))
            normal[find] = np.sum(normal[find],axis=0) # sum up their normal vectors
            
        # select only the unique elements
        elec, normal = uniq.T, normal[idx_uniq].T
        
        if retsurf:
            electrode_surf = electrode.copy()
            for el in el_surf: electrode_surf[tuple(el)] = 2 # give the electrode surface a different value
            # plot corner elements
            electrode_surf[tuple(cornerel)] = 3 # give the electrode corners a different value
            return elec, normal, electrode_surf
        else:
            return elec, normal
        
        
        # empty can be removed in the future
        
        # check if elec0 and el_surf can be removed -> blocking layer/activity
     
        
     
    #############    3D part  ################## 
    
    elif self.dim == '3D':
        posxpos, posxneg = [], []
        posypos, posyneg = [], []
        poszpos, poszneg = [], []
        
        # along z
        for x,plane in enumerate(np.transpose(electrode,axes=(2,1,0))): # choose yz-plane (determines x coordinate)
            for y,row in enumerate(plane): # choose y coordinate in the plane
                prev = 0
                for z,el in enumerate(row): # go along z coordinate
                    if el and not prev:
                        prev = 1
                        if z > 0: poszneg.append([z-1,y,x]) 
                    if not el and prev:
                        prev = 0
                        poszpos.append([z,y,x])
        
        # along y
        for z,plane in enumerate(electrode): # choose xy-plane (determines z coordinate)
            for x,row in enumerate(plane.T): # choose x coordinate in the plane
                prev = 0
                for y,el in enumerate(row): # go along y coordinate
                    if el and not prev:
                        prev = 1
                        if y > 0: posyneg.append([z,y-1,x]) 
                    if not el and prev:
                        prev = 0
                        posypos.append([z,y,x])
                    
        # along x
        for y,plane in enumerate(np.transpose(electrode,axes=(1,0,2))): # choose xz-plane (determines y coordinate)
            for z,row in enumerate(plane): # choose z coordinate in the plane
                prev = 0
                for x,el in enumerate(row): # go along x coordinate
                    if el and not prev:
                        prev = 1
                        if x > 0: posxneg.append([z,y,x-1])
                    if not el and prev:
                        prev = 0
                        posxpos.append([z,y,x])
                    
        posxpos, posypos, poszpos = np.array(posxpos).T, np.array(posypos).T, np.array(poszpos).T
        posxneg, posyneg, poszneg = np.array(posxneg).T, np.array(posyneg).T, np.array(poszneg).T
        
        el_surf = [posxpos, posxneg, posypos, posyneg, poszpos, poszneg] # list with surface positions
        if el_surf == []: raise ValueError('The electrode was not correctly specified!')
        
        # before looping, filter out the empty elements:
        empty = [idx for idx in range(len(el_surf)) if el_surf[idx].size==0]
        if empty != []: el_surf = np.delete(el_surf,empty)
        
        # specify the surface elements
        displace = np.array([[0,0,0,0,-1,1],[0,0,-1,1,0,0],[-1,1,0,0,0,0]]).T.reshape((6,3,1))
        displace = np.delete(displace, empty, axis=0)
        elec0 = [el_surf[i] + displace[i] for i in range(len(el_surf))] # list with electrode positions
        
        # find the normal to the surface
        normalvec = np.delete(np.array([[[1,0,0]],[[-1,0,0]],[[0,1,0]],[[0,-1,0]],[[0,0,1]],[[0,0,-1]]]), empty, axis=0)
        normalvec = np.delete(np.array([[[0,0,1]],[[0,0,-1]],[[0,1,0]],[[0,-1,0]],[[1,0,0]],[[-1,0,0]]]), empty, axis=0)
        normal0 = []
        for i, el in enumerate(elec0): normal0.append(el*0.0+normalvec[i].T)
        
        # merge elec and normal to one 1D ndarray
        elec, normal = elec0[0], normal0[0]
        for i in np.arange(1,len(elec0)):
            elec = np.concatenate((elec, elec0[i]),axis=1)
            normal = np.concatenate((normal, normal0[i]), axis=1)
            
        # sort the elec and normal arrays
        elec, normal = elec.T, normal.T
        for i in [2,1,0]:
            sortel = elec[:,i].argsort(kind='mergesort')
            elec, normal = elec[sortel], normal[sortel]
            
        # there are 6 situations, the elements of the surf can border to:
        # 1 element:  just x, y or z dir
        # 2 elements: 2D edge element, flux will be the sum of the two corrected by the normal0 vector
        # 3 elements: 3D corner element
        # 4 elements: Corner of a slab: (1,0,0) + (-1,0,0) + (0,1,0) + (0,0,1) = (0,1,1)       
        # 5 elements: Top of a rod, only one dir should survive
        # 6 elements: this piece is not connected to the rest of the electrode, no direction should survive: (0,0,0)
        uniq, idx_uniq, count = np.unique(elec, axis=0, return_counts=True, return_index=True)
        dup = np.where(count>1)[0] # select only the duplicates
        for el, cnt in zip(uniq[dup],count[dup]): # find all elements which are duplicates
            find = np.intersect1d(np.intersect1d(np.where(elec[:,0]==el[0]), np.where(elec[:,1]==el[1])), np.where(elec[:,2]==el[2]))
            normal[find] = np.sum(normal[find],axis=0) # sum up their normal vectors
            
        # select only the unique elements
        elec, normal = uniq.T, normal[idx_uniq].T
        
        if retsurf:
            electrode_surf = electrode.copy()
            for el in el_surf: electrode_surf[tuple(el)] = 2 # give the electrode surface a different value
            if len(normalvec)>1:
                edges = uniq[count == 2].T.astype(np.int)
                corners = uniq[count > 2].T.astype(np.int)
                electrode_surf[tuple(edges)] = 3 # give the electrode edges a different value
                electrode_surf[tuple(corners)] = 4 # give the electrode corners a different value
            
            return elec, normal, electrode_surf
        
        else:
            return elec, normal, el_surf, elec0, empty, normal0
        
    else: raise ValueError('dim should be \'2D\' or \'3D\'!')    
