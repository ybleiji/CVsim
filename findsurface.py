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
    elements of the surface of the electrode, the elements which are the 
    electrode, the normal vectors corresponding to the electrode elements,
    and it returns the empty directions. When retsurf=True, it
    returns an array with 0 for no electrode, 1 for electrode, and 2 for
    the surface of the electrode. This is for displaying the surface.
    
    Input parameters:
        dim: dimension, either '2D' or '3D'
        electrode: array containing 0 for no electrode and 1 for electrode (2D or 3D ndarray)
        (optional) retsurf: set the surface of the electrode to the value 2 in the electrode array (default = False)
    
    Output parameters:
        el_surf: list with the positions of the surface of the electrode in the electrolyte (1D ndarray)
        elec: list with the positions of the surface of the electrode (1D ndarray)
        empty: list with directions which does not contain an electrode (1D list)
        (optional) electrode_surf: array with 0 if electrolyte, 1 if electrode, 2 if surface (2D or 3D ndarray)
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
        elec = [el_surf[i] + displace[i] for i in range(len(el_surf))] # list with electrode positions
        
        # find the normal to the surface
        normalvec = np.delete(np.array([[[1,0]],[[1,0]],[[0,1]],[[0,1]]]), empty, axis=0)
        
        normalvec = np.delete(np.array([[[1,0]],[[-1,0]],[[0,1]],[[0,-1]]]), empty, axis=0)
        
        cornerel, normal = np.array([[],[]]), []
        for i, el in enumerate(elec): # search within elec for corner elem
            cornerel = np.concatenate((cornerel, el), axis=1)
            normal.append(el*0.0+normalvec[i].T)
        uniq, count = np.unique(cornerel.T, return_counts=True, axis=0)
        cornerel = uniq[count > 1].T.astype(np.int) # here the positions of the corner elements in the 2D matrix are being saved 
         
        
        
        
        
        #### make one array of elec, el_surf and normal instead of all the different directions
        
        # merge the el_surf, elec and normal to one 1D ndarray
        el_surf0, elec0, normal0 = el_surf[0], elec[0], normal[0]
        for i in np.arange(1,len(elec)):
            elec0 = np.concatenate((elec0, elec[i]), axis=1)
            el_surf0 = np.concatenate((el_surf0, el_surf[i]), axis=1)
            normal0 = np.concatenate((normal0, normal[i]), axis=1)    
            
        # sort the elec, el_surf and norm array
        elec0, el_surf0, normal0 = elec0.T, el_surf0.T, normal0.T
        for i in [1,0]:
            sortel = elec0[:,i].argsort(kind='mergesort')
            elec0, el_surf0, normal0 = elec0[sortel], el_surf0[sortel], normal0[sortel]
        
        # since the array is sorted, it finds the duplicates by looking to the previous element
        prevel, prevnorm = 0, 0
        # delidx = []
        for idx, (el, norm) in enumerate(zip(elec0,normal0)):
            if all(prevel == el): normal0[idx-1] = normal0[idx] = norm + prevnorm # add the two vectors
            prevel, prevnorm = el, norm
        # elec0, normal0 = np.delete(elec0,delidx,0), np.delete(normal0,delidx,0) # remove the duplicates
        # print(elec0)
        # print(normal0)
        elec0, el_surf0, normal0 = elec0.T, el_surf0.T, normal0.T
        
        if retsurf:
            electrode_surf = electrode.copy()
            for el in el_surf: electrode_surf[tuple(el)] = 2 # give the electrode surface a different value
            # plot corner elements
            electrode_surf[tuple(cornerel)] = 3 # give the electrode corners a different value
            
            return el_surf, elec, empty, normal, electrode_surf
        
        else:
            return el_surf, elec, empty, normal, el_surf0, elec0, normal0
     
        
     
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
        elec = [el_surf[i] + displace[i] for i in range(len(el_surf))] # list with electrode positions
        
        # find the normal to the surface
        normalvec = np.delete(np.array([[[1,0,0]],[[1,0,0]],[[0,1,0]],[[0,1,0]],[[0,0,1]],[[0,0,1]]]), empty, axis=0)
        normal = []
        for i, el in enumerate(elec):
            normal.append(el*0.0+normalvec[i].T)
        
        if len(normalvec)>1: # edges and corners are only present for more than one surface direction
                            
            # assign the correct normal vectors to the edge elements
            empty_inv = list(np.delete(range(6),empty))
            comb_indices = list(combinations(empty_inv,2))
            normalvec_edge = np.array(list(combinations(normalvec,2)))
            normalvec_edge = np.sum(normalvec_edge,axis=1)
            edges = np.array([[],[],[]])
            for i,comb in enumerate(list(combinations(elec,2))):
                if comb_indices[i] in ((0,1),(2,3),(4,5)): continue # skip combinations of the same coordinate
                duplicates_temp = np.concatenate(comb,axis=1)
                uniq_temp, count_temp = np.unique(duplicates_temp.T, return_counts=True, axis=0)
                edges_temp = uniq_temp[count_temp == 2].T.astype(np.int)
                edges = np.concatenate((edges,edges_temp), axis=1)
                a0, b0, c0 = np.in1d(comb[0][0], edges_temp[0]), np.in1d(comb[0][1], edges_temp[1]), np.in1d(comb[0][2], edges_temp[2])
                a1, b1, c1 = np.in1d(comb[1][0], edges_temp[0]), np.in1d(comb[1][1], edges_temp[1]), np.in1d(comb[1][2], edges_temp[2])
                for j in np.where(a0*b0*c0)[0]: normal[comb_indices[i][0]][:,j] = normalvec_edge[i][0]
                for j in np.where(a1*b1*c1)[0]: normal[comb_indices[i][1]][:,j] = normalvec_edge[i][0]
                
            duplicates = np.array([[],[],[]])
            for i, el in enumerate(elec): # search within elec for duplicates
                duplicates = np.concatenate((duplicates, el), axis=1)
            uniq, count = np.unique(duplicates.T, return_counts=True, axis=0)
            
            # here the normal vector [1,1,1] is assigned to each corner element of the 3D matrix
            corners = uniq[count == 3].T.astype(np.int)
            for i,el in enumerate(elec):
                a, b, c = np.in1d(el[0], corners[0]), np.in1d(el[1], corners[1]), np.in1d(el[2], corners[2])
                for j in np.where(a*b*c)[0]: normal[i][:,j] = np.ones(3)
            
        if retsurf:
            electrode_surf = electrode.copy()
            for el in el_surf: electrode_surf[tuple(el)] = 2 # give the electrode surface a different value
            if len(normalvec)>1:
                electrode_surf[tuple(edges)] = 3 # give the electrode edges a different value
                electrode_surf[tuple(corners)] = 4 # give the electrode corners a different value
            
            return el_surf, elec, empty, normal, electrode_surf
        
        else:
            return el_surf, elec, empty, normal
        
    else: raise ValueError('dim should be \'2D\' or \'3D\'!')    
