# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 16:39:51 2014

@author: Parke
"""
import numpy as np

def arc(data, rho_min=0.8, Ntrends=None, denoise=None):
    """Identify systematic trends in the data. Trends are removed as part
    of the identification process.

    Parameters
    ----------
    data : 2D array-like
        The data (e.g. lightcurves). Should have shape NxM where there are
        N data points for M separate series.
    rho_min : float
        The stopping threshold for the iterative trend identification process.
        This specifies the spectral radius of the first principal component
        of a candidate trend set and quantifies the similarity of the members
        of the trend set. When rho_min, the members of the trend set are
        considered too dissimilar to represent a true underlying trend in the
        data and iteration is stopped.
    Ntrends : int, optional
        Number of trends to identify -- an alternative stopping condition to
        rho_min. If both are specified, iteration will stop when the first
        criterion is met.
    denoise : function, optional
        Denoising function to apply to each trend. For example, Roberts et al.
        (2013, MNRAS 435:3639) use Emprical Mode Decomposition.

    Returns
    -------
    trends : 2D numpy array
        The identified trends. The array will have shape NxK, N data points
        for K trends.
    """

    N, M = data.shape
    trends = np.array([])
    trends.shape = [N,0]
    while True:
        #get basis sets, i.e. the set of weights for each lightcurve that
        #is a best fit of the other lightcurves to that lightcurve
        hhparams = [1e-2, 1e-4, 1e-2, 1e-4]
        weights = np.zeros([M, M])
        for i in range(M):
            others = np.ones(M, bool)
            others[i] = False
            weights[i, others] = basisweights(data[:,i], data[:,others],
                                               hhparams, varweights=False)

        #compute shannon entropy of weights for different basis sets
        entropies = np.zeros(M+1)
        for i,w in enumerate(weights.T):
            w = np.delete(w, i)
            p = w**2/(w**2).sum()
            l, U = entropies[i] = -np.sum(p*np.log2(p))

        #keep only the M best (highest entropy) sets
        keep = np.argpartition(entropies, -M)[-M:]
        weights = weights[:,keep]
        T = data*weights
        l, U = np.linalg.eig(T*T.T)
        normfac = np.linalg.norm(T)
        U = U*normfac
        rho1 = l[0]/l.sum()

        if rho1 < rho_min:
            break

        trend = U[:,0][:,0]
        if denoise is not None:
            trend = denoise(trend)

        trends = np.append(trends, trend)

        if Ntrends and len(trends == Ntrends):
            break

        yarray = trend_remove(data, trends)

    return trends

def basisweights(data, basisset, hhparams, varweights=True, stop=1e-3):
    """function to find weights that explain each y in terms of all other y"""

    # contrstruct data and basis set matrices
    dd = np.asmatrix(data) #vector d in Roberts eq A3
    P = np.matrix(basisset) #Phi in Roberts eq A5
    a0, b0, c0, d0 = hhparams

    # transpose if necessary and check that no. of data points match
    N = len(data)
    if dd.shape[0] == 1:
        dd = dd.T
    if P.shape[0] != N:
        if P.shape[1] != N:
            raise ValueError('series and basisset contain different numbers '
                             'of data points')
        else:
            P = P.T
    K = P.shape[1]

    # iterate to converge on basis set weights by succesively updating weights,
    # hyperparams, and hyperhyperparams

    #initialize for the loop
    wold = np.inf*np.matrix(np.ones([K,1]))
    w = np.matrix(np.zeros([K,1]))
    I = np.matrix(np.identity(K))
    c = c0
    d = N/2.0 + d0

    #make a function to test if stopping condition has been met
    proceed = lambda: np.all(abs((w-wold)/w) > 1e-3)

    #iterate!
    if varweights:
        a = np.matrix([a0]*K).T
        b = 0.5 + b0
        while proceed():
            wold = w
            alpha, beta = a*b, c*d
            S = np.linalg.inv(beta*P.T*P + np.diag(alpha))
            w = (S*beta*P.T*dd)
            E = 0.5*np.multiply(w, w)
            a = 1.0/(E + 0.5*np.diag(S) + 1.0/a0)
            c = float(1.0/(0.5*(dd - P*w).T*(dd - P*w)
                            + 0.5*(S*P.T*P).trace() + 1.0/c0))
    else:
        a = a0
        b = K/2.0 + b0
        while proceed():
            wold = w
            alpha, beta = a*b, c*d
            S = np.linalg.inv(beta*P.T*P + alpha*I)
            w = (S*beta*P.T*dd)
            a = float(1.0/(w.T*w + 0.5*S.trace() + 1.0/a0))
            c = float(1.0/(0.5*(dd - P*w).T*(dd - P*w)
                            + 0.5*(S*P.T*P).trace() + 1.0/c0))

    return w

def trend_remove(data, trends):
    pass