#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Authors: Rémi LELUC, François PORTIER and Johan SEGERS
This file contains functions to integrate over [0,1]^d.
'''

# Basic python libraries
import numpy as np
# Python libraries for density functions
from scipy.stats import lognorm,expon
lbda = np.log(2)
# Log-normal density function with mean=0 and variance=1
def f(x):
    return 2*lognorm.pdf(x=x,s=1)
# Exponential density function with parameter lambda = ln(2)
def g(x):
    return 2*expon.pdf(x=x,scale=1/lbda)
# Sinusoidal function
def phi(x):
    return 1 + np.sin(np.pi*(2*np.mean(x) - 1))
#################################################################
# Build functions that takes into account more and more variables
#################################################################
def f1(x):
    return f(x[0])
def f3(x):
    return np.prod(f(x)[:3], axis=0)
def f4(x):
    return np.prod(f(x)[:4], axis=0)
def f5(x):
    return np.prod(f(x)[:5], axis=0)
def f6(x):
    return np.prod(f(x)[:6], axis=0)

def g1(x):
    return g(x[0])
def g3(x):
    return np.prod(g(x)[:3], axis=0)
def g4(x):
    return np.prod(g(x)[:4], axis=0)
def g5(x):
    return np.prod(g(x)[:5], axis=0)
def g6(x):
    return np.prod(g(x)[:6], axis=0)