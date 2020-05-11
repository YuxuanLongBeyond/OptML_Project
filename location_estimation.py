#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 20:49:38 2020

@author: YuxuanLong
"""

## Optimization for ML 

### Project --- Camera location estimation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SEED = 2020
np.random.seed(SEED)


def define_parameters(W, V, num, edge_num):
    # linear operator for extracting ti - tj, s.t. row of Qs T is (ti - tj)
    Qs = np.zeros((edge_num, num))
    index = np.arange(num)
    row = 0
    for i in range(num):
        r = W[i, :] == 1.0
        s = r.sum()
        Qs[np.arange(row, row + s), i] = 1
        Qs[np.arange(row, row + s), index[r]] = -1
        row += s
    
    # augmented form of Qs
    Q = np.kron(Qs, np.eye(3))  # Q_{ij} t = ti - tj
        
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
    L = np.dot(Q.T, np.dot(Pc, Q))
    
    # linear operator s.t. b^T t = 1
    b = np.dot(Q.T, v)
    
    A_tilde = np.concatenate((A, b.reshape((1, -1))), axis = 0)
    
    parameters = {}
    parameters['A'] = A
    parameters['Q'] = Q
    parameters['L'] = L
    parameters['b'] = b
    parameters['P'] = P
    parameters['Pc'] = Pc
    parameters['A_tilde'] = A_tilde
    return parameters

def compute_all_l2(Pc, y):
    Pc_y = np.dot(Pc, y)
    l2_norm_all = np.sqrt(Pc_y[0::3] ** 2 + Pc_y[1::3] ** 2 + Pc_y[2::3] ** 2)
    return l2_norm_all

def evaluate_obj(parameters, t):
    Pc = parameters['Pc']
    Q = parameters['Q']
    y = np.dot(Q, t)
    return np.sum(compute_all_l2(Pc, y))

def qp_solver(L, parameters, num):
#    # implement QP with squared l2 loss
    A_tilde = parameters['A_tilde']
    tem = np.linalg.solve(L, A_tilde.T)
    return tem.dot(np.linalg.solve(A_tilde.dot(tem), [0,0,0,1]))
    

def admm_solver(parameters, y0, lam0, num, edge_num, max_iter, l1_prox, delta, tau, kicked, epsilon, ratio):
    # implement scaled ADMM or kicked ADMM, where rho = 2
    Q = parameters['Q']
    P = parameters['P']
    Pc = parameters['Pc']

    A_tilde = parameters['A_tilde']
    
    Q_pinv = np.linalg.pinv(Q)
    B = np.linalg.solve(np.dot(Q.T, Q), A_tilde.T)
    tem = B.dot(np.linalg.inv(A_tilde.dot(B)))
    D = Q_pinv - tem.dot(A_tilde.dot(Q_pinv))
    
    c = tem[:, -1]
    
    
    y = y0
    lam = lam0
    l2_loss = 1000000 # dummy here
    for i in range(max_iter):
        # update T
        t = np.dot(D, y - lam) + c
        
        # update Y
        z = np.dot(Q, t) + lam
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
        lam += np.dot(Q, t) - y
        
        # primal
        l2_loss_next = np.sum(compute_all_l2(Pc, y))
        if kicked and abs(l2_loss_next - l2_loss) < epsilon:
            tau *= ratio
            lam /= ratio
        l2_loss = l2_loss_next
#        print('L2 loss: ', l2_loss)
    return t

def irls_solver(parameters, num, edge_num, max_iter, delta, use_l1):
    Pc = parameters['Pc']
    Q = parameters['Q']
    w = 1
    tem = np.dot(Pc, Q)
    for i in range(max_iter):
        # update L
        L = np.dot(tem.T * w, tem)
        
        # solve t
        t = qp_solver(L, parameters, num)
        
        # update weight
        Pc_y = np.dot(tem, t)
        if use_l1:
            w = 1 / (np.abs(Pc_y) + delta)
        else:
            squared_norm = Pc_y[0::3] ** 2 + Pc_y[1::3] ** 2 + Pc_y[2::3] ** 2
            w = np.repeat(1 / np.sqrt(squared_norm + delta), 3)
        
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
    t_qp = qp_solver(parameters['L'], parameters, num)
    loss = evaluate_obj(parameters, t_qp)
    print('L2 loss of QP: ', loss)
    T_qp = t_qp.reshape((num, 3))
    
    # ADMM
    kicked = False
    l1_prox = True
    max_iter = 100
    delta = 0.000001
    tau = 0.1 # the smaller, the greater penalty on l1
    ratio = 10
    epsilon = 1e-8
    y0 = np.zeros((3 * edge_num, ))
    lam0 = np.zeros((3 * edge_num, ))    
    t_admm = admm_solver(parameters, y0, lam0, num, edge_num, max_iter, l1_prox, delta, tau, kicked, epsilon, ratio)
    loss = evaluate_obj(parameters, t_admm)
    print('L2 loss of ADMM: ', loss)
    T_admm = t_admm.reshape((num, 3))

    # IRLS
    max_iter = 100
    delta = 0.00001
    use_l1 = False # if False, use l2 as objective
    t_irls = irls_solver(parameters, num, edge_num, max_iter, delta, use_l1)
    loss = evaluate_obj(parameters, t_irls)
    print('L2 loss of IRLS: ', loss)
    T_irls = t_irls.reshape((num, 3))

    T_gt = np.load('./data/T.npy') # ground truth
    t0 = np.mean(T_gt, axis = 0)
    scale = np.dot(parameters['b'], (T_gt - t0).reshape((3 * num, )))
    
    # shift and scale to original frame
    T_qp = T_qp * scale + t0
    T_admm = T_admm * scale + t0
    T_irls = T_irls * scale + t0
    
    # mean absolute error in the original coordinate frame
    MAE_qp = np.mean(np.abs(T_gt - T_qp))
    MAE_admm = np.mean(np.abs(T_gt - T_admm))
    MAE_irls = np.mean(np.abs(T_gt - T_irls))
    print('MAE of QP: ', MAE_qp)
    print('MAE of ADMM: ', MAE_admm)
    print('MAE of IRLS: ', MAE_irls)

    # visualize and compare
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    ax.scatter(T_gt[:, 0], T_gt[:, 1], T_gt[:, 2])
    ax.scatter(T_qp[:, 0], T_qp[:, 1], T_qp[:, 2])    
    ax.scatter(T_admm[:, 0], T_admm[:, 1], T_admm[:, 2])
    ax.scatter(T_irls[:, 0], T_irls[:, 1], T_irls[:, 2])
    