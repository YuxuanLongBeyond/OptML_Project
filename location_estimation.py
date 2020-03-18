#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 20:49:38 2020

@author: YuxuanLong
"""

## Optimization for ML 

### Project --- Camera location estimation
import numpy as np
SEED = 2020
np.random.seed(SEED)

if __name__ == "__main__":
    noise_variance = 0.001 # adjust it for experiment
    
    W = np.load('./data/W.npy') # graph adjacency matrix
    V = np.load('./data/V.npy') # collection of direction unit vectors
    

    num = W.shape[0]
    edge_num = V.shape[0]
    
    # add Gaussian noise to V
    V += np.random.randn(edge_num, 3) * np.sqrt(noise_variance)
    
    # linear operator for extracting ti - tj, s.t. row of Rs T is (ti - tj)
    Rs = np.zeros((edge_num, num))
    index = np.arange(num)
    row = 0
    for i in range(num):
        r = W[i, :] == 1.0
        s = r.sum()
        Rs[np.arange(row, row + s), i] = 1
        Rs[np.arange(row, row + s), index[r]] = -1
        row += s
    
    # augmented form of Rs
    R = np.kron(Rs, np.eye(3))  # R_{ij} t = ti - tj
        
    # linear operator s.t. At = 0
    A = np.zeros((3, 3 * num)) 
    A[0, 0::3] = 1
    A[1, 1::3] = 1
    A[2, 2::3] = 1
    
    # complementary projector (null space is only spanned by direction vectors)
    v = V.reshape((3 * edge_num, ))
    Pc = np.eye(3 * edge_num, 3 * edge_num) - np.outer(v, v)
    mask = np.kron(np.eye(edge_num, edge_num), np.ones((3, 3))) == 0
    Pc[mask] = 0
    P = np.eye(3 * edge_num, 3 * edge_num) - Pc
    
    # L s.t. the squared l2 loss is t^T L t
    L = np.dot(R.T, np.dot(Pc, R))
    
    # linear operator s.t. b^T t = 1
    b = np.dot(R.T, v)
    
    
    
    
    
    # implement QP with squared l2 loss
    C = np.zeros((4 + 3 * num, 4 + 3 * num))
    C[0:(3 * num), (3 * num):(3 * num + 3)] = A.T
    C[0:(3 * num), (3 * num + 3):] = b.reshape((3 * num, 1))
    C += C.T
    C[0: 3*num, 0:3*num] = L
    C_inv = np.linalg.inv(C)
    t = C_inv[0:(3 * num), -1]
    
    # implement scaled ADMM or kicked ADMM, where rho = 2
    kicked = True
    max_iter = 1000
    tau = 0.0000001 # the smaller, the greater penalty on l1
    ratio = 10
    tol = 1e-8
    
#    C = np.zeros((4 + 3 * num, 4 + 3 * num))
#    C[0:(3 * num), (3 * num):(3 * num + 3)] = A.T
#    C[0:(3 * num), (3 * num + 3):] = b.reshape((3 * num, 1))
#    C += C.T
    C[0: 3*num, 0:3*num] = np.dot(R.T, R)
    C_inv = np.linalg.inv(C)

    D = np.dot(C_inv[0:(3 * num), 0:(3 * num)], R.T)
    
    
    y = np.zeros((3 * edge_num, ))
#    y = np.dot(R, t)
    lam = np.zeros((3 * edge_num, ))
    l1_loss = 100
#    t = np.zeros((3 * edge_num, ))
    for i in range(max_iter):
        # update T
        w = y - lam
        t = np.dot(D, w) + C_inv[0:(3 * num), -1]
        
        # update Y
        z = np.dot(R, t) + lam
        P_z = np.dot(P, z)
        Pc_z = np.dot(Pc, z)
        y = P_z + np.sign(Pc_z) * np.maximum(Pc_z - 1 / tau, 0)
        
        # update lambda
        lam += np.dot(R, t) - y
        
        # primal
        l1_loss_next = np.sum(np.abs(np.dot(Pc, y)))
        if kicked and abs(l1_loss_next - l1_loss) < tol:
            tau *= ratio
            lam /= ratio
        l1_loss = l1_loss_next
        print(l1_loss)
        

    # implement IRLS
    

    # compare with GT data
    # we should compute a scale and a translation
    
    T = np.load('./data/T.npy') # ground truth
    
    
    
    