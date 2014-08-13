# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 16:39:51 2014

@author: Parke
"""
import numpy as np
import my_numpy as mynp
from scipy.interpolate import interp1d

def trend_detect(t,data,rho_min=0.8,Ntrends=None):
    """Detect the systematic trends in the data included as columns in the array
    data."""
    
    a0, b0, c0, d0 = 1e-2, 1e-4, 1e-2, 1e-4
    N, K = data.shape
    K -= 1
    M = 10
    
    #function to find weights that explain each y in terms of all other y
    def basisweights(yarray):
        b, d = K/2.0 + b0, N/2.0 + d0
        weights = np.matrix(np.zeros([K+1,K+1]))
        
        for i in range(K):
            #contrstruct data and basis set matrices
            d = np.matrix(yarray[:,i]).T
            P = np.matrix(np.delete(yarray,i,1))
            
            #iterate to converge on basis set weights
            a, c = a0, c0
            w = np.inf*np.matrix(np.ones(K)).T
            I = np.matrix(np.identity(K))
            while True:
                wold = w
                alpha, beta = a*b, c*d
                S = np.linalg.inv(beta*P.T*P + alpha*I)
                w = (S*beta*P.T*d)
                if all(abs(w - wold)/w < 1e-3): break
                a = 1.0/(w.T*w + S.trace()/2.0 + 1.0/a0)
                c = 1.0/(0.5*(d - P*w).T*(d - P*w) + 0.5*(S*P.T*P).trace() + 1.0/c0)
            
            #store weights
            iw = range(K+1).remove(i)
            weights[iw,i] = w
        
#        trend_candidates = yarray*weights
        return weights
        
        trends = np.array([])
        trends.shape = [N,0]
        yarray = data
        while True:
            #get basis sets
            weights = basisweights(yarray)
            
            #compute shannon entropy of weights for different basis sets
            entropies = np.zeros(K+1)
            for i,w in enumerate(weights.T):
                w = np.delete(w, i)
                p = w**2/(w**2).sum()
                l, U = entropies[i] = -np.sum(p*np.log2(p))
            
            #keep only the M best (highest entropy) sets
            keep = np.argpartition(entropies, -M)[-M:]
            weights = weights[:,keep]
            T = yarray*weights
            l, U = np.linalg.eig(T*T.T)
            normfac = np.linalg.norm(T)
            U = U*normfac
            rho1 = l[0]/l.sum()
            
            if (not Ntrends) and rho1 < rho_min:
                break
            else:
                trend = sift(U[:,0])[:,0]
                trends = np.hstack([trends,trend])
                if Ntrends and len(trends == Ntrends): break
                yarray = trend_remove(data, trends)
            
    return trends

def trend_remove(data, trends):
    pass