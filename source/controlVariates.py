#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Authors: Rémi LELUC, François PORTIER and Johan SEGERS
This file contains functions to get matrix H = (h_j(X_i)) of size n x k x d
of the control variates, in Legendre/Hermite/Laguerre/Fourier basis.
It also implements a simple function to draw random samples
'''
#########################################################
# Import libraries
import numpy as np
# Polynomial families for control variates
from numpy.polynomial.hermite_e import hermeval
from numpy.polynomial.legendre import legval
from numpy.polynomial.laguerre import lagval
from scipy.special import factorial
#########################################################

def draw_sample(n,d,law='uniform'):
    ''' Function to draw n samples in dimension d of some distribution
    Params
    @n (int)  : number of samples to draw
    @d (int)  : dimension of the problem
    @law (str): distribution 'uniform' or 'normal'
    Returns
    @X (array n x d): random samples X_1,...,X_n
    '''
    if law == 'uniform': # Uniform law on [0,1]^d
        X = np.random.rand(n, d)
    elif law == 'normal': # Normal law on R^d
        X = np.random.randn(n, d)
    return X

#########################################################
############### CONTROL VARIATES FAMILIES ###############
#########################################################
def legendre(k,t):
    ''' Legendre Polynomial of degree k at point t 
    Params
    @k   (int): order of Legendre polynomial function
    @t (float): point to be evaluated
    Returns: (float) Leg_k(t)
    '''   
    c = np.zeros(k+1)
    c[-1] = 1
    return 2*legval(2*t-1, c)

def hermite(k,t):
    ''' Hermite Polynomial of degree k at point t 
    Params
    @k   (int): order of Hermite polynomial function
    @t (float): point to be evaluated
    Returns: (float) Her_k(t)
    '''
    c = np.zeros(k+1)
    c[-1] = 1
    return hermeval(t, c)/factorial(k)

def laguerre(k,t):
    ''' Laguerre Polynomial of degree k at point t 
    Params
    @k   (int): order of Laguerre polynomial function
    @t (float): point to be evaluated
    Returns: (float) Lag_k(t)
    '''
    c = np.zeros(k+1)
    c[-1] = 1
    return lagval(t, c)

def fourier(k,t):
    ''' Fourier basis function of order k at point t
    Params
    @k   (int): order of basis fourier function
    @t (float): point to be evaluated
    Returns: (float) Fourier_k(t)
    '''
    if k%2==1:
        return np.sqrt(2)*np.cos((k+1)*np.pi*t)
    else:
        return np.sqrt(2)*np.sin(k*np.pi*t)
#########################################################
############# Compute Matrix H of size n x k x d ########
#########################################################
def get_H_nkd(X,k,basis='legendre'):
    ''' Compute Tensor matrix H = (h_j(X_i))of size n x k x d in the given basis
    Params
    @X (array n x d): random samples X_1,...,X_n
    @k (int)        : number of control variates in each direction
    @basis (str)    : name of the basis in ['legendre','hermite','laguerre','fourier']
    Returns
    @H : matrix H = (h_j(X_i)) of size n x k x d in the given basis
    '''
    n,d = X.shape
    H = np.zeros((n,k,d))
    # Fill the matrix H in the given basis
    if basis=='legendre':
        for j in range(1,k+1):
            H[:,j-1,:] = legendre(j,X)
    elif basis=='hermite':
        for j in range(1,k+1):
            H[:,j-1,:] = hermite(j,X)
    elif basis=='laguerre':
        for j in range(1,k+1):
            H[:,j-1,:] = laguerre(j,X)
    elif basis=='fourier':
        for j in range(1,k+1):
            H[:,j-1,:] = fourier(j,X)
    return H
#########################################################
#############   TENSOR PRODUCT MASKS   ##################
#########################################################
def mask_single(k,d):
    ''' Compute mask of tensor products of control functions
    with only one activate coordinate: m = k*d
    Params
    @k (int) : number of control variates in each direction
    @d (int) : dimension of the problem
    Returns
    @mask (array m x d): mask of tensor products combinations
    '''
    I = np.eye(d)
    mask = np.eye(d)
    for i in range(2,k+1):
        mask = np.concatenate((mask,i*I))
    return mask.astype(int)

def mask_pairs(k,d):
    ''' Compute mask of tensor products of control functions
    with only 2 activate coordinates: m = k*d + k*k*d*(d-1)/2
    Params
    @k (int) : number of control variates in each direction
    @d (int) : dimension of the problem
    Returns
    @mask (array m x d): mask of tensor products combinations
    '''
    tab_list = []
    for i in range(d-1):
        for j in range(i+1,d):
            for i_k in range(0,k+1):
                for j_k in range(0,k+1):
                    tab = [0]*d
                    tab[i] = i_k
                    tab[j] = j_k
                    tab_list.append(tab)
    mask = np.delete(arr=sorted(list(set(tuple(row) for row in tab_list)),key=sum),obj=0,axis=0)
    return mask.astype(int)