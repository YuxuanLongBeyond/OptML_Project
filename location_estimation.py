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

def ls_solver(L, parameters, num):
#    # implement LS with squared l2 loss
    A_tilde = parameters['A_tilde']
    tem = np.linalg.solve(L, A_tilde.T)
    return tem.dot(np.linalg.solve(A_tilde.dot(tem), [0,0,0,1]))
    

def admm_solver(parameters, y0, lam0, num, edge_num, max_iter, l1_prox, delta, tau):
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
    for i in range(max_iter):
        # update T
        r = y - lam
        t = np.dot(D, r) + c
        
        # update Y
        z = np.dot(Q, t) + lam
        P_z = np.dot(P, z)
        Pc_z = np.dot(Pc, z)
        if l1_prox:
            # default
            y = P_z + np.sign(Pc_z) * np.maximum(Pc_z - 1 / tau, 0)
        else:
            # l2 proximal
            all_l2_norm = compute_all_l2(Pc, z)
            tem = np.kron(all_l2_norm, np.ones((3, ))) + delta
            y = P_z + np.maximum(1 - 1 / (tau * tem), 0) * Pc_z
        
        # update lambda
        lam += np.dot(Q, t) - y
        
        
#        l2_loss_next = np.sum(compute_all_l2(Pc, y))
#        l2_loss = l2_loss_next
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
        t = ls_solver(L, parameters, num)
        
        # update weight
        Pc_y = np.dot(tem, t)
        if use_l1:
            w = 1 / (np.abs(Pc_y) + delta)
        else:
            squared_norm = Pc_y[0::3] ** 2 + Pc_y[1::3] ** 2 + Pc_y[2::3] ** 2
            w = np.repeat(1 / np.sqrt(squared_norm + delta), 3)
        
#        print('Loss: ', evaluate_obj(parameters, t))    
    return t

def main(noise_variance, W, V, show_plot):
    num = W.shape[0]
    edge_num = V.shape[0]    
    # add Gaussian noise to V
    V += np.random.randn(edge_num, 3) * np.sqrt(noise_variance)  
#    V = (V.T / np.sqrt(np.sum(V ** 2, axis = 1))).T
    
    # define parameters such as operators and projectors
    parameters = define_parameters(W, V, num, edge_num)
    
    
    # LS
    t_ls = ls_solver(parameters['L'], parameters, num)
    loss = evaluate_obj(parameters, t_ls)
    print('L2 loss of LS: ', loss)
    T_ls = t_ls.reshape((num, 3))
    
    # ADMM
    l1_prox = True
    max_iter = 100
    delta = 0.00001
    tau = 0.1 # the smaller, the greater penalty on l1
    y0 = np.zeros((3 * edge_num, ))
    lam0 = np.zeros((3 * edge_num, ))    
    t_admm = admm_solver(parameters, y0, lam0, num, edge_num, max_iter, l1_prox, delta, tau)
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
    
    # shift and scale to original coordinate system
    T_ls = T_ls * scale + t0
    T_admm = T_admm * scale + t0
    T_irls = T_irls * scale + t0
    
    # mean absolute error in the original coordinate frame
    MAE_ls = np.mean(np.abs(T_gt - T_ls))
    MAE_admm = np.mean(np.abs(T_gt - T_admm))
    MAE_irls = np.mean(np.abs(T_gt - T_irls))
    print('MAE of LS: ', MAE_ls)
    print('MAE of ADMM: ', MAE_admm)
    print('MAE of IRLS: ', MAE_irls)    
    
    if show_plot:
        # visualize and compare
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        ax.scatter(T_gt[:, 0], T_gt[:, 1], T_gt[:, 2], label = 'GT')
        ax.scatter(T_ls[:, 0], T_ls[:, 1], T_ls[:, 2], label = 'LS')    
        ax.scatter(T_admm[:, 0], T_admm[:, 1], T_admm[:, 2], label = 'ADMM')
        ax.scatter(T_irls[:, 0], T_irls[:, 1], T_irls[:, 2], label = 'IRLS')
        ax.legend(loc = "upper left")
    return MAE_ls, MAE_admm, MAE_irls

if __name__ == "__main__":
    noise_variance = 0.03 # adjust it for experiment
    show_plot = 1
    robustness_test = 0
    var_list = np.linspace(0.0001, 0.0201, 101)
    
    W = np.load('./data/W.npy') # graph adjacency matrix
    V = np.load('./data/V.npy') # collection of direction unit vectors    
    
    MAE_ls, MAE_admm, MAE_irls = main(noise_variance, W, V, show_plot)

    
    if robustness_test:
        show_plot = 0
        err_ls = np.zeros(len(var_list))
        err_admm = np.zeros(len(var_list))
        err_irls = np.zeros(len(var_list))
        for i, var in enumerate(var_list):
            print(i, var)
            W = np.load('./data/W.npy') # graph adjacency matrix
            V = np.load('./data/V.npy') # collection of direction unit vectors
            MAE_ls, MAE_admm, MAE_irls = main(var, W, V, show_plot)
            err_ls[i] = MAE_ls
            err_admm[i] = MAE_admm
            err_irls[i] = MAE_irls
        
        plt.plot(var_list, err_ls, label = 'LS')
        plt.plot(var_list, err_admm, label = 'ADMM')
        plt.plot(var_list, err_irls, label = 'IRLS')
        plt.legend(loc = "upper left")
        plt.xlabel('Noise variance')
        plt.ylabel('MAE')
        plt.show()
    