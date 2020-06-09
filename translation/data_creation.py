#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:29:22 2020

@author: YuxuanLong
"""

## Optimization for ML 

### Data synthesis for camera locations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


SEED = 2020
np.random.seed(SEED)


def dataMaker(dist_from_center, r_range, num, d):
    """
    Synthesize camera locations, which are randomly distributed in the sphere
    """
    r = np.random.random((num, )) * r_range + dist_from_center
    phi = np.random.random((num, )) * 2 * np.pi # angle from z-axis
    theta = np.random.random(num) * np.pi # angle from x-axis
    
    T = np.zeros((num, d))
    T[:, 0] = r * np.sin(phi) * np.cos(theta)
    T[:, 1] = r * np.sin(phi) * np.sin(theta)
    T[:, 2] = r * np.cos(phi)
    
    return T

def graphBuilder(T, num, edge_connection_prob, dist_thre):
    """
    Create a viewing graph with random connections between the nearby cameras
    """
    
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
    num = 100 # number of cameras
    d = 3 # dimension of location, default to 3
    dist_from_center = 1 # minimum distance between the camera locations and the world center
    r_range = 1 # radius range of locations from the minimum distance
    
    edge_connection_prob = 0.5  # connection probility between cameras
    dist_thre = 1.5 # two cameras can connect only if they have distance smaller than dist_thre
    
    # make up camera locations
    T = dataMaker(dist_from_center, r_range, num, d)
    
    # visualize the locations
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    ax.scatter(T[:, 0], T[:, 1], T[:, 2])
    
    # build a viewing graph
    W = graphBuilder(T, num, edge_connection_prob, dist_thre)
    
    edge_num = int(W.sum()) # number of edges in the graph
    
    # create pairwise direction observations
    V = np.zeros((edge_num, d)) # pairwise direction observations
    row = 0
    for i in range(num):
        for j in range(num):
            if W[i, j] == 1.0:
                ti = T[i, :]
                tj = T[j, :]
                V[row, :] = (ti - tj) / np.linalg.norm(ti - tj)
                row += 1
    
    # save synthetic data
    np.save('./data/W.npy', W)
    np.save('./data/T.npy', T)
    np.save('./data/V.npy', V)
    
    