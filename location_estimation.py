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


def define_parameters(W, V, num, edge_num):
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
    
    parameters = {}
    parameters['A'] = A
    parameters['Rs'] = Rs
    parameters['R'] = R
    parameters['L'] = L
    parameters['b'] = b
    parameters['P'] = P
    parameters['Pc'] = Pc
    return parameters

def compute_all_l2(Pc, y):
    Pc_y = np.dot(Pc, y)
    l2_norm_all = np.sqrt(Pc_y[0::3] ** 2 + Pc_y[1::3] ** 2 + Pc_y[2::3] ** 2)
    return l2_norm_all

def evaluate_obj(parameters, t):
    Pc = parameters['Pc']
    R = parameters['R']
    y = np.dot(R, t)
    return np.sum(compute_all_l2(Pc, y))

def qp_solver(L, parameters, num):
    A = parameters['A']
    b = parameters['b']
    
    # implement QP with squared l2 loss
    C = np.zeros((4 + 3 * num, 4 + 3 * num))
    C[0:(3 * num), (3 * num):(3 * num + 3)] = A.T
    C[0:(3 * num), (3 * num + 3):] = b.reshape((3 * num, 1))
    C += C.T
    C[0: 3*num, 0:3*num] = L
    C_inv = np.linalg.inv(C)
    return C_inv[0:(3 * num), -1]   

def admm_solver(parameters, num, edge_num, max_iter, l1_prox, delta, tau, kicked, epsilon, ratio):
    # implement scaled ADMM or kicked ADMM, where rho = 2
    A = parameters['A']
    R = parameters['R']
    b = parameters['b']
    P = parameters['P']
    Pc = parameters['Pc']
    
    C = np.zeros((4 + 3 * num, 4 + 3 * num))
    C[0:(3 * num), (3 * num):(3 * num + 3)] = A.T
    C[0:(3 * num), (3 * num + 3):] = b.reshape((3 * num, 1))
    C += C.T
    C[0: 3*num, 0:3*num] = np.dot(R.T, R)
    C_inv = np.linalg.inv(C)

    D = np.dot(C_inv[0:(3 * num), 0:(3 * num)], R.T)
    
    y = np.zeros((3 * edge_num, ))
    lam = np.zeros((3 * edge_num, ))
    l2_loss = 100
    for i in range(max_iter):
        # update T
        w = y - lam
        t = np.dot(D, w) + C_inv[0:(3 * num), -1]
        
        # update Y
        z = np.dot(R, t) + lam
        P_z = np.dot(P, z)
        Pc_z = np.dot(Pc, z)
        if l1_prox:
            y = P_z + np.sign(Pc_z) * np.maximum(Pc_z - 1 / tau, 0)
        else:
            # l2 proximal
            all_l2_norm = compute_all_l2(Pc, z)
            tem = np.kron(all_l2_norm, np.ones((3, ))) + delta
            y = P_z + np.maximum(1 - 1 / (tau * tem), 0) * Pc_z
        
        # update lambda
        lam += np.dot(R, t) - y
        
        # primal
        l2_loss_next = np.sum(compute_all_l2(Pc, y))
        if kicked and abs(l2_loss_next - l2_loss) < epsilon:
            tau *= ratio
            lam /= ratio
        l2_loss = l2_loss_next
#        print('L2 loss: ', l2_loss)
    return t

def irls_solver(parameters, num, edge_num, max_iter, delta):
    Pc = parameters['Pc']
    R = parameters['R']
    W = 1
    for i in range(max_iter):
        # update L
        P_new = W * Pc
        L = np.dot(R.T, np.dot(P_new, R))
        
        # solve t
        t = qp_solver(L, parameters, num)
        
        # update weight
        y = np.dot(R, t)
        squared_norm = compute_all_l2(Pc, y) ** 2
        W = np.kron(np.diag(1 / np.sqrt(squared_norm + delta)), np.ones((3, 3)))
        
#        print('Loss: ', evaluate_obj(parameters, t))
        
    return t

if __name__ == "__main__":
    noise_variance = 0.01 # adjust it for experiment
    
    W = np.load('./data/W.npy') # graph adjacency matrix
    V = np.load('./data/V.npy') # collection of direction unit vectors
    

    num = W.shape[0]
    edge_num = V.shape[0]
    
    # add Gaussian noise to V
    V += np.random.randn(edge_num, 3) * np.sqrt(noise_variance)
    
    # define parameters such as operators and projectors
    parameters = define_parameters(W, V, num, edge_num)
    
    
    # QP
    t = qp_solver(parameters['L'], parameters, num)
    loss = evaluate_obj(parameters, t)
    print('L2 loss of QP: ', loss)
    
    # ADMM
    kicked = False
    l1_prox = True
    max_iter = 100
    delta = 0.000001
    tau = 0.1 # the smaller, the greater penalty on l1
    ratio = 10
    epsilon = 1e-8
    t = admm_solver(parameters, num, edge_num, max_iter, l1_prox, delta, tau, kicked, epsilon, ratio)
    loss = evaluate_obj(parameters, t)
    print('L2 loss of ADMM: ', loss)

    # IRLS
    max_iter = 100
    delta = 0.00001
    t = irls_solver(parameters, num, edge_num, max_iter, delta)
    loss = evaluate_obj(parameters, t)
    print('L2 loss of IRLS: ', loss)

    
    # compare with GT data
    # we should compute a scale and a translation
    
#    T = np.load('./data/T.npy') # ground truth
#    t = T.reshape((3 * num, ))
#    loss = evaluate_obj(parameters, t)
#    print('L2 loss of ground truth: ', loss)
    
    