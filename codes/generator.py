#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 18:59:10 2019

@author: ricky
"""

import numpy as np
from numba import jit

def data_simulation(M, k, V, xi=100, shape=2, scale=1):
    docs = []
    alpha = np.random.gamma(shape=shape, scale=scale, size=k)
    beta = np.random.dirichlet(np.ones(V), k)
    N = np.random.poisson(lam=xi, size=M)
    theta = np.random.dirichlet(alpha, M)

    for d in range(M):
        z = np.random.multinomial(n=1, pvals=theta[d,], size=N[d])
        tmp = z @ beta
        w = np.zeros((N[d], V))
        for n in range(N[d]):
            w[n,] = np.random.multinomial(n=1, pvals=tmp[n,], size=1)
        docs.append(w)
    return docs, alpha, beta

@jit
def data_simulation_numba(M, k, V, xi=100, shape=2, scale=1):
    docs = []
    alpha = np.random.gamma(shape=shape, scale=scale, size=k)
    beta = np.random.dirichlet(np.ones(V), k)
    N = np.random.poisson(lam=xi, size=M)
    theta = np.random.dirichlet(alpha, M)

    for d in range(M):
        z = np.random.multinomial(n=1, pvals=theta[d, ], size=N[d])
        tmp = z @ beta
        w = np.zeros((N[d], V))
        for n in range(N[d]):
            w[n,] = np.random.multinomial(n=1, pvals=tmp[n, ], size=1)
        docs.append(w)
    return docs, alpha, beta



class generator(object):
    def __init__(self, K, V, alpha, kesai):
        self.K = K
        self.V = V
        self.alpha = alpha
        self.beta = np.random.dirichlet(np.ones(V), K)
        self.kesai = kesai

    def make_parameter(self):
        n = self.V//self.K
        for i in range(self.K):
            temp = np.ones(self.V)
            temp[n*i:n*(i+1)] += 10
            self.beta[i] = np.random.dirichlet(temp)
        return self.beta

    def create_one_document(self):
        n = np.random.poisson(self.kesai)
        theta = np.random.dirichlet(self.alpha)
        z = np.random.multinomial(1,theta, n)
        temp = z@self.beta
        doc = np.zeros((n, self.V))
        for i in range(n):
            doc[i, ] = np.random.multinomial(n=1, pvals=temp[i, ], size=1)
        return doc

    def sample(self, m):
        docs = []
        for i in range(m):
            docs.append(self.create_one_document())
        return docs