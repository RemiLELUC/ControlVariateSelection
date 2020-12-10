#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Rémi LELUC, François PORTIER, Johan SEGERS
This file contains the class CVMC which implements the control variates
Monte Carlo estimator with vanilla,olsmc,lassomc,lslassomc,lslassoXmc.
'''
######################################################################
# Import libraries to build control variates
import numpy as np
from time import time
from collections import Counter
from itertools import product
# Tool function to draw random samples and build tensor matrix H size (n,k,d)
from controlVariates import draw_sample, get_H_nkd
# Libraries to perform OLS and LASSO
from scipy.linalg import lstsq
from sklearn.linear_model import LinearRegression,Lasso,LassoCV
from sklearn.preprocessing import StandardScaler,normalize
######################################################################

class CVMC():
    ''' Control Variates Monte Carlo estimator
    Implements all the CVMC estimates for the paper
    "Control Variates Selection for Monte Carlo Integration"
    Attributes:
    @func    (function): function to integrate in dimension d
    @n            (int): number of random nodes
    @d            (int): dimension of the problem
    @k            (int): number of control variates in each direction
    @m            (int): total number of control variates m=(k+1)**d - 1
    @law          (str): distribution 'uniform' or 'normal'
    @basis        (str): name of the basis in ['legendre','hermite','laguerre','fourier']
    @verbose  (boolean): to display some information about computing times
    @X    (array n x d): random samples X_1,...,X_n
    @f_n  (array n x 1): evaluations f(X_1),...,f(X_n)
    @H    (array n x m): matrix H = (h_j(X_i)) of size n x m in the given basis
    @indices     (list): number of control variates whose total degree <= deg
    @mu_van     (float): Naive MC estimate
    @mu_ols     (float): OLSMC estimate
    @mu_lasso   (float): LASSOMC estimate
    @mu_lslassox(float): LSLASSOXMC estimate
    
    Methods:
    @compute_H(self)
    @compute_ols(self,deg)
    @compute_lasso_lslasso(self,deg,c1,c2,cross_val=False)
    @compute_lslassox(self,deg,n_sample,c1,c2,cross_val=False)
    @get_van(self)
    @get_ols(self,deg)
    @get_lasso(self,deg,c1,c2,cross_val=False)
    @get_lslasso(self,deg,c1,c2,cross_val=False)
    @get_lslassox(self,deg,n_sample,c1,c2,cross_val=False)
    '''
    def __init__(self,seed,func,n,d,k,law,basis,verbose=False):
        self.seed = seed   # random seed
        self.func = func   # function to integrate
        self.n = n         # number of random nodes
        self.d = d         # dimension of the problem
        self.k = k         # number of control variates in each direction
        self.law = law     # distribution 'uniform' or 'normal'
        self.basis = basis # name of the basis
        self.verbose = verbose # Boolean to display information
        self.X = draw_sample(n=self.n,d=self.d,law=self.law) # random samples X_1,...,X_n
        self.f_n = np.array([(self.func)(x) for x in self.X]).reshape(self.n, 1) # f(X_1),...,f(X_n)
        self.m = (self.k+1)**(self.d) - 1 # total number of control variates
        self.mu_van = np.mean(self.f_n)   # vanilla MC estimate
        
    def compute_H(self):
        ''' Compute matrix H = (h_j(X_i))of size n x m with m=(k+1)^d - 1 '''
        t_start = time()
        # Draw mask of all possible multidimensional control variates
        # We sort by sum of degree and will increase this (growing m)
        self.mask = np.delete(arr=sorted(list(product(range(self.k + 1), repeat=self.d)), key=sum), obj=0, axis=0)
        self.dico = Counter(self.mask.sum(-1))
        # Indices to slice matrix H to keep only control variates under a certain degree
        self.indices = np.cumsum(list(self.dico.values()))
        # Tools array to build matrix of control variates
        add = np.ones((self.n, 1, self.d))
        col = np.arange(self.d)
        # Compute matrix H of size n x k x d in given basis
        H_temp = get_H_nkd(X=self.X, k=self.k, basis=self.basis)
        # Add [1,...,1] to represent control h_0 = 1
        H_temp = np.concatenate((add, H_temp), axis=1)
        # Build matrix H of size n x m
        m_true = (self.k+1)**(self.d) - 1
        H_total = np.empty((self.n, m_true))
        for i in range(self.n):  # Fill i-th row
            # compute product of separable variables
            H_total[i] = np.prod(a=np.choose(self.mask,H_temp[i]), axis=1)
        if self.verbose:
            print('T compute H   :',time()-t_start)
        # Set the value of the regressor matrix H
        self.H = H_total
        
    def compute_ols(self,deg):
        ''' Compute OLSMC estimate '''
        if not hasattr(self, 'H'):
            self.compute_H()
        # OLSMC esimtate
        ols = LinearRegression(fit_intercept=True, normalize=True, n_jobs=-1)
        m = self.indices[deg-1] # number of control variates with total degree <= deg
        t_start = time()
        ols.fit(X=self.H[:,:m],y=self.f_n)
        if self.verbose:
            print('\nT fit OLS     :',time()-t_start)
        # Set the value of the estimate
        self.mu_ols = (ols.intercept_)[0]
    
    def compute_lasso_lslasso(self,deg,c1,c2,cross_val=False):
        ''' Compute LASSOMC and LSLASSOMC estimates '''
        if not hasattr(self, 'H'):
            self.compute_H()
        t_start = time()
        m = self.indices[deg-1] # number of control variates with total degree <= deg
        if self.verbose:
            print('Current m  : ',m)
        H_current = self.H[:, :m]
        # cross-validation strategy
        if cross_val: 
            lasso = LassoCV(fit_intercept=True, normalize=True, cv=5, n_jobs=-1)
            lasso.fit(X=H_current,y=self.f_n)
            # Time to compute Lasso
            t_lasso = time() - t_start
            if self.verbose:
                print('T fit LASSO   :',t_lasso)
            # Set the value of the estimates
            self.mu_lasso = lasso.intercept_
            # get active set (support of beta)
            active_set = np.nonzero(lasso.coef_)[0]
            # LSLASSOMC part
            # extract useful control variates from H
            H_active = self.H[:, active_set]
            lsl = LinearRegression(fit_intercept=True, normalize=True, n_jobs=-1)
            t_start = time()
            lsl.fit(X=H_active,y=self.f_n)
            t_lslasso = time() - t_start
            if self.verbose:
                print('T fit LSLASSO :',t_lslasso)
            # Set the value of the estimate
            self.mu_lslasso = (lsl.intercept_)[0]
        # dichotomic search strategy
        else: 
            # Compute alpha_max_lasso for which all coefficients are equal to 0
            y_offset = np.average(a=self.f_n.ravel(),axis=0)
            y = self.f_n.ravel()-y_offset
            X_offset = np.average(a=H_current,axis=0)
            X_c = H_current-X_offset
            matX = normalize(X=X_c,axis=0)
            Xy = np.dot(matX.T, y)[:, np.newaxis]
            λ_max_lasso = np.abs(Xy).max() / self.n
            # interval [λ_g,λ_d] to search a good value of λ
            λ_d_lasso = λ_max_lasso 
            λ_g_lasso = 0
            λ_lasso = 1
            # number of selected components with this value of λ
            cv_selected = 0
            lasso_count = 0
            # bounds on the number of selected components 
            lower_bound = int(c1 * np.sqrt(self.n))
            upper_bound = int(c2 * np.sqrt(self.n))
            if self.verbose:
                print('begin Lasso Loop')
            # loop to adjust λ according to number of selected components
            while ((cv_selected < lower_bound) or (cv_selected > upper_bound)) and (λ_lasso > 1e-12):
                λ_lasso = 0.95 * λ_g_lasso + 0.05 * λ_d_lasso
                lasso = Lasso(alpha=λ_lasso,
                              fit_intercept=True,
                              normalize=True,
                              random_state=self.seed)
                lasso.fit(X=H_current, y=self.f_n.ravel())
                # compute number of selected components
                cv_selected = len(np.nonzero(lasso.coef_)[0])
                lasso_count += 1
                if self.verbose:
                    print('Lasso_count:',lasso_count)
                # if all components selected, we stop
                if cv_selected == m:
                    break
                # if not enough control variates selected, we decrease lambda
                if cv_selected < lower_bound:
                    λ_d_lasso = λ_lasso
                # if too much control variates selected, we increase lambda
                if cv_selected > upper_bound:
                    λ_g_lasso = λ_lasso
            # Time to compute Lasso
            t_lasso = time() - t_start
            if self.verbose:
                print('cv_selected: ',cv_selected)
                print('λ_lasso: ',λ_lasso)
                print('T fit LASSO   :',t_lasso)
            # Set the value of the estimates
            self.mu_lasso = lasso.intercept_
            # get active set (support of beta)
            active_set = np.nonzero(lasso.coef_)[0]
            # LSLASSOMC part
            H_active = self.H[:, active_set]
            lsl = LinearRegression(fit_intercept=True, normalize=True, n_jobs=-1)
            t_start = time()
            lsl.fit(X=H_active,y=self.f_n)
            t_lslasso = time() - t_start
            if self.verbose:
                print('T fit LSLASSO :',t_lslasso)
            # Set the value of the estimate
            self.mu_lslasso = (lsl.intercept_)[0]
              
    def compute_lslassox(self,deg,n_sample,c1,c2,cross_val=False):
        ''' Compute LSLASSOXMC estimate '''
        if not hasattr(self, 'H'):
            self.compute_H()
        # Compute alpha_max_lassox for which all coefficients are equal to 0
        t_start = time()
        m = self.indices[deg-1] # number of control variates with total degree <= deg
        H_current = self.H[:n_sample, :m]
        # cross-validation strategy
        if cross_val: 
            lassox = LassoCV(fit_intercept=True, normalize=True, cv=5, n_jobs=-1)
            lassox.fit(X=H_current,y=self.f_n[:n_sample])
            # get active set (support of beta)
            active_setx = np.nonzero(lassox.coef_)[0]
            # extract useful control variates from H
            H_activex = self.H[:, active_setx]
            lslx = LinearRegression(fit_intercept=True, normalize=True, n_jobs=-1)
            t_start = time()
            lslx.fit(X=H_activex,y=self.f_n)
            t_lslassox = time() - t_start
            if self.verbose:
                print('T fit LSLASSOX :',t_lslassox)
            # Set the value of the estimate
            self.mu_lslassox = (lslx.intercept_)[0]
        # dichotomic search strategy    
        else:
            y_offset = np.average(a=self.f_n.ravel()[:n_sample],axis=0)
            y = self.f_n.ravel()[:n_sample]-y_offset
            X_offset = np.average(a=H_current,axis=0)
            X_c = H_current-X_offset
            matX = normalize(X=X_c,axis=0)
            Xy = np.dot(matX.T, y)[:, np.newaxis]
            λ_max_lassox = np.abs(Xy).max() / n_sample
            λ_d_lassox = λ_max_lassox 
            λ_g_lassox = 0
            cvx_selected = 0
            lower_bound = int(c1 * np.sqrt(self.n))
            upper_bound = int(c2 * np.sqrt(self.n))
            lassox_count = 0
            λ_lassox = 1
            if self.verbose:
                print('begin LassoX Loop')
            while ((cvx_selected < lower_bound) or (cvx_selected > upper_bound)) and (λ_lassox > 1e-12):
                λ_lassox = 0.95 * λ_g_lassox + 0.05 * λ_d_lassox
                lassox = Lasso(alpha=λ_lassox,
                               fit_intercept=True,
                               normalize=True,
                               random_state=self.seed)
                lassox.fit(X=H_current, y=self.f_n.ravel()[:n_sample])
                # compute number of selected components
                cvx_selected = len(np.nonzero(lassox.coef_)[0])
                lassox_count += 1
                if self.verbose:
                    print('LassoX_count:',lassox_count)
                # if all components selected, we stop
                if cvx_selected == m:
                    break
                # if not enough control variates selected, we decrease lambda
                if cvx_selected < lower_bound:
                    λ_d_lassox = λ_lassox
                # if too much control variates selected, we increase lambda
                if cvx_selected > upper_bound:
                    λ_g_lassox = λ_lassox
            # get active set (support of beta)
            active_setx = np.nonzero(lassox.coef_)[0]
            # extract useful control variates from H
            H_activex = self.H[:, active_setx]
            lslx = LinearRegression(fit_intercept=True, normalize=True, n_jobs=-1)
            t_start = time()
            lslx.fit(X=H_activex,y=self.f_n)
            t_lslassox = time() - t_start
            if self.verbose:
                print('T fit LSLASSOX :',t_lslassox)
            # Set the value of the estimate
            self.mu_lslassox = (lslx.intercept_)[0]
    
    def get_van(self):
        ''' get Naive MC estimate '''
        return self.mu_van
    
    def get_ols(self,deg):
        ''' get OLSMC estimate '''
        if not hasattr(self, 'mu_ols'):
            self.compute_ols(deg=deg)
        return self.mu_ols

    def get_lasso(self,deg,c1,c2,cross_val=False):
        ''' get LASSOMC estimate '''
        if not hasattr(self, 'mu_lasso'):
            self.compute_lasso_lslasso(deg=deg,c1=c1,c2=c2,
                                       cross_val=cross_val)
        return self.mu_lasso
    
    def get_lslasso(self,deg,c1,c2,cross_val=False):
        ''' get LASSOMC estimate (full data) '''
        if not hasattr(self, 'mu_lslasso'):
            self.compute_lasso_lslasso(deg=deg,c1=c1,c2=c2,
                                       cross_val=cross_val)
        return self.mu_lslasso
    
    def get_lslassox(self,deg,n_sample,c1,c2,cross_val=False):
        ''' get LSLASSOXMC estimate (subsample data)'''
        if not hasattr(self, 'mu_lslassox'):
            self.compute_lslassox(deg=deg,n_sample=n_sample,
                                  c1=c1,c2=c2,cross_val=cross_val)
        return self.mu_lslassox
    
    