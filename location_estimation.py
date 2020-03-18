#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:29:22 2020

@author: YuxuanLong
"""

## Optimization for ML 

### Project --- Camera location estimation

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def dataMaker(dist_from_center, r_range, num, d):
    r = np.random.random((num, )) * r_range + dist_from_center
    phi = np.random.random((num, )) * 2 * np.pi # angle from z-axis
    theta = np.random.random(num) * np.pi # angle from x-axis
    
    T = np.zeros((num, d))
    T[:, 0] = r * np.sin(phi) * np.cos(theta)
    T[:, 1] = r * np.sin(phi) * np.sin(theta)
    T[:, 2] = r * np.cos(phi)
    
    return T

def graphBuilder(T, num, edge_connection_prob, dist_thre):
    # create adjacency matrix
    W = np.random.random((num, num))
    W = (W < edge_connection_prob).astype(np.float32)
    W -= np.diag(np.diag(W))
    
    # thresholding by distance
    dist_map = np.array([np.sqrt(np.sum((T[i, :] - T) ** 2, axis = 1)) for i in range(num)])
    W[dist_map > dist_thre] = 0
    
    # ensure that W is asymmetric
    mask1 = (W + W.T) / 2 == 1.0
    random_matrix = np.random.random((num, num))
    mask2 = np.triu(random_matrix > 0.5)
    mask2[np.tril(mask2.T == 0)] = True
    W[mask1 & mask2] = 0
    
    # compute average degree
    print('The average degree is ', np.mean(np.sum(W, axis = 1)))
    
    return W
    

if __name__ == "__main__":
    num = 50
    d = 3 # constant
    dist_from_center = 1
    r_range = 1
    
    edge_connection_prob = 0.8 # only if two cameras have distance smaller than dist_thre
    dist_thre = 1.5
    
    # make up camera locations
    T = dataMaker(dist_from_center, r_range, num, d)
    
    # visualize
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(T[:, 0], T[:, 1], T[:, 2])
    
    # build camera graph
    W = graphBuilder(T, num, edge_connection_prob, dist_thre)
    
    