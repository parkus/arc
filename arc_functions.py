# -*- coding: utf-8 -*-
"""
A module for applying the astrophysically robust correction, described by
Roberts et al. (2013 MNRAS 435:3639) to series (namely time-series) data.

Modification History:
2015 April written by R. O. Parke Loyd
"""
import numpy as np
import emd

def arc_emd_choice(t, y, method='spline'):
    """
    Denoise the data in y by returning the intrinsic mode (or residual) with
    the largest variance as found using empirical mode decomposition.

    Parameters
    ----------
    y : 1D array-like
        The data to be denoised.
    method : {'spline'|'saw'}
        Which intrinsic mode identification process to employ.

    Result
    ------
    y_denoised : 1D array
        The denoised data.
    """
    if method == 'spline':
        modes, residual = emd.emd(t, y)
    if method == 'saw':
        modes, residual = emd.saw_emd(t, y)
    choices = np.append(modes, residual[:, np.newaxis], axis=1)
    stds = np.var(choices, axis=0)
    i_choice = np.argmax(stds)
    return choices[:, i_choice]

arc_emd = lambda t, y: arc_emd_choice(t, y, 'spline')
arc_emd_fast = lambda t, y: arc_emd_choice(t, y, 'saw')

def arc(t, data, rho_min=0.8, Ntrends=None, denoise=arc_emd, Nkeep=10):
    """Identify systematic trends in the data.

    Parameters
    ----------
    t : 1D array-like
        The independent data, generally time.
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
    denoise : {None|function}, optional
        Denoising function to apply to each trend. With the accompanying emd
        (empirical mode decomposition) module available at
        https://github.com/parkus/emd, two built in denoising techniques can
        bet used

        - arc_emd : default
            Performs empirical mode decomposition in the "traditional"
            manner of fitting splines to the series extrema and selects
            the resulting mode (or residual) with the largest variance.

        - arc_emd_fast :
            Same as the above, but using the sawtooth transform for speed
            instead of spline fitting. This results in a slightly
            difference set of intrinsic modes.

        User defined denoising functions must accept two 1D arrays as input
        (t and y) and return the "denoised" y.
    Nkeep : int, optional
        Number of candidate trends to use for computing the principal component
        and spectral radius during each iteration.

    Returns
    -------
    trends : 2D numpy array
        The identified trends. The array will have shape NxK, N data points
        for K trends. Trends will be normalized to have zero mean and unit
        standard deviation.
    """
    # groom the input
    N, M = data.shape
    trends = np.empty([N, 0], dtype=data.dtype)
    if denoise is not None:
        dn = denoise
        denoise = lambda t, y: np.reshape(dn(t, y[:, 0]), [N, 1])

    # mean subtract and sigma-normalize the data
    try:
        data = _normalize(data)[0]
    except ValueError:
        raise ValueError("One or more sereis has zero variance. ARC can't "
                         "search for trends in such constant series.")

    original_data = data

    while True:
        # get basis sets, i.e. the set of weights for each lightcurve that
        # is a best fit of the other lightcurves to that lightcurve
        weights = np.zeros([M, M])
        for i in range(M):
            others = np.ones(M, bool)
            others[i] = False
            w = basisweights(data[:,i], data[:,others], varweights=False)
            weights[i, others] = w

        # compute shannon entropy of weights for different basis sets
        entropies = shannon_entropy(weights)

        # keep only the M best (highest entropy) sets
        args = np.argsort(entropies)[::-1]
#        entropy_cut = np.max(entropies)/keep_fac
#        keep = entropies > entropy_cut
#        keep = args[entropies[args] > entropy_cut]
        keep = args[:Nkeep]
        weights = weights[keep, :]

        # compute the leading trends from the weights
        ys = construct(data, weights)

        # find the principle componenet of the retained trends
        trend, rho = principle_component(ys)

        # break out of loop if we've reached a trend that has too low a
        # spectral radius (doesn't explain the data well)
        if rho < rho_min:
            break

        # denoise the trend, if desired
        if denoise is not None:
            trend = denoise(t, trend)

        # normalize the trend
        trend = _normalize(trend)[0]

        # record the trend
        trends = np.append(trends, trend, axis=1)

        if Ntrends and len(trends == Ntrends):
            break

        # fit and subtract all the trends identified so far from the data
        data = [trend_remove(d, trends)[0] for d in original_data.T]
        data = np.array(data).T
        data = _normalize(data)[0]

    return trends

def basisweights(data, basisset, hhparams=[1e-2, 1e-4, 1e-2, 1e-4],
                 varweights=True, stop=1e-3):
    """
    Finds the weights by which the basisset series can be linearly combined
    to best represent the data.

    Parameters
    ----------
    data : 1D array-like
        The data to be modeled as a linear combination fo the basis series,
        length N.
    basisset : 2D array-like
        The set of basis series (i.e. basis functions) to be used to represent
        the data, shape NxM where there are M such series.
    hhparams : list of 4 floats, optional
        Initial values of the hyper-hyper-parameters in the order
        [a0, b0, c0, d0]. See the appendix of Roberts et al. for a description.
        Default values are those used in Roberts et al.
    varweights : {True|False}, optional
        Whether to allow the prior probability distributions for the weights
        to vary. If false, all are held to a Guassian PDF with identical
        variances. If true, each has its own variance that evolves separately
        with each iteration.
    stop : float, optional
        Stopping criterion. When no weight has a relative change greater than
        the stop value over an iteration, the iteration is halted.

    Returns
    -------
    weights : 1D array
        The best-fist basis weights, length M.
    """

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
    if K == 0: raise ValueError('No basis series provided.')

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

    return np.squeeze(np.array(w))

def principle_component(trends):
    """
    Find the principal compenent of a set of trends (principle component
    analysis).

    Parameters
    ----------
    trends : 2D array-like
        A set of trends (or any series) with shape NxM where there are M trends
        of N data points each.

    Returns
    -------
    pc : 1D array
        The principle component of the trends as a len(N) array.
    rho : float
        The spectral radius of the returned trend.
    """
    T = np.matrix(trends)

    # compute principal component
    l, U = np.linalg.eig(T * T.T)
    l, U = np.real(l), np.real(U)
#    normfac = np.linalg.norm(T) # not sure why I tried this initially...
    normfac = np.sqrt(np.sum(np.array(T[:,0])**2)) # this is tested to work
    U = U*normfac
    pc = np.array(U[:, 0])

    # spectral radius
    rho = float(l[0]/l.sum())

    return pc, rho

def construct(basisset, weights):
    """
    Construct a series by linearly combining the basis set series according
    to the weights.

    Parameters
    ----------
    basisset : 2D array-like
        The basis series, supplied as an array with dimensions NxM, where
        M is the number of series (.e.g functions) and N is the number of
        data points in each series.
    weights : 1 or 2D array-like
        The weights to apply to each series before linearly summing to compute
        the output. Either a 1D array of length N may be supplied or an array
        of shape KxM, where M is the number of sereis there are K sets of
        weights to compute output for.

    Returns
    -------
    y : 1D or 2D array
        The data series constructed from the basisset and weights. If
        weights is 2D, y will be 2d with shape NxK, otherwise it will be 1D
        with length N.
    """
    # vet the input
    b, w = map(np.asarray, [basisset, weights])
    if w.ndim == 1:
        w = w.reshape(1, len(w))

    # compute the result using matrix multiplication
    y = np.dot(w, b.T).T
    y = np.squeeze(y)

    return y

def shannon_entropy(weights):
    """
    Compute the Shannon Entropy of a set or sets of weights.

    Parameters
    ----------
    weights : array-like
        The weights for which to compute the shannon entropy. To rapidly
        compute the shannon entropy of several sets of weights, supply
        a 2D array with the sets of weights arranged in rows.

    Returns
    -------
    entropies : float or array
        The entropy or entropies, according to whether input was 1D or 2D.

    Notes
    -----
    Zeros in the weights array will be ignored (otherwise the result is NaN).
    """
    w, axis = _groom_weights(weights)

    # compute the entropies
    p = w**2 / np.sum(w**2, axis=axis)
    # put ones in the diagonal as a trick to ignore those values (log(1) = 0)
    p[np.diag_indices_from(p)] = 1.0
    H = -np.sum(p * np.log2(p), axis=axis)

    return H

def _groom_weights(weights):
    """Groom an input weight array for use with entropy computing functions."""
    w = np.asarray(weights)
    w = np.squeeze(w)
    w = w.T
    axis = 0 if w.ndim > 1 else None
    return w, axis

def trend_remove(data, trends):
    """
    Fit and remove the provided trends from the data.

    Parameters
    ----------
    data : 1D array-like
        The data to fit, length N.
    trends : 2D array-like
        The trends to fit to the data, dimensions NxM where there are M trends.

    Returns
    -------
    detrended : 1D array
        The data with the trend subtracted, length N.
    trendfit : 1D array
        The fit of the trend to the data, length N.

    Notes
    -----
    If daterr or trenderrs is supplied, the other must also be supplied or
    errors cannot be propagated.
    """
    if trends.shape[1] == 0: raise ValueError("No trends supplied.")

    # mean subtract and sigma-normalize the data
    data, restore = _normalize(data)

    # fit the data
    weights = basisweights(data, trends)
    trendfit = construct(trends, weights)
    detrended = data - trendfit

    # undo the median subtract and normalization
    detrended, trendfit = map(restore, [detrended, trendfit])

    return detrended, trendfit

def _normalize(y):
    """
    Mean-subtract and normalize the data in the columns of y. Return the
    normalized data and a function that will return a single (1D array)
    normalized series to the pre-normalized state.
    """
    nd = y.ndim
    if nd == 1:
        y = y.reshape([len(y), 1])

    mn = np.mean(y, axis=0)
    std = np.std(y, axis=0)
    if np.any(std == 0.0):
        raise ValueError("Can't normalize series with std dev of 0.0")
    z = (y - mn) / std

    restore = lambda x: (x * std) + mn

    if nd == 1:
        z = z[:, 0]

    return z, restore
