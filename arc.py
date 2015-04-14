# -*- coding: utf-8 -*-
"""
A module for applying the astrophysically robust correction, described by
Roberts et al. (2013 MNRAS 435:3639) to series (namely time-series) data.

Modification History:
2015 April written by R. O. Parke Loyd
"""
import numpy as np
import emd

def arc(t, data, dataerrs=None, rho_min=0.8, Ntrends=None, denoise=arc_emd,
        keep_fac=2.0):
    """Identify systematic trends in the data.

    Parameters
    ----------
    t : 1D array-like
        The independent data, generally time.
    data : 2D array-like
        The data (e.g. lightcurves). Should have shape NxM where there are
        N data points for M separate series.
    dataerrs : 2D array-like
        Errors on the data that will be propogated into the trends.
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
        Denoising function to apply to each trend. The default is empirical
        mode decomposition as used in Roberts et al.
    keep_fac : float, optional
        Combine all trends with a shannon entropy > max/fac, where max is
        the maximum shannon entropy of all candidate trends. Note that shannon
        entropies generally span orders of magnitude.

    Returns
    -------
    trends : 2D numpy array
        The identified trends. The array will have shape NxK, N data points
        for K trends. Trends will be normalized to have zero mean and unit
        standard deviation.
    trenderrs : 2D numpy array
        Conservatively estimated errors in the trends. NOT EXACT. None if no data
        errors are supplied.
    """

    if dataerrs is not None and denoise is not None:
        raise ValueError('Trend errors cannont be esimated from the data'
                         'errors is a denoising shceme is also used.')

    N, M = data.shape
    def emptyarr():
        x = np.array([])
        x.shape = [N,0]
        return x
    trends = emptyarr()
    trenderrs = None if dataerrs is None else emptyarr()

    # median subtract and mean-normalize the data
    data = data - np.median(data, axis=0)
    normfacs = np.std(data, axis=0)
    if np.any(normfacs == 0.0):
        raise ValueError("One or more sereis has zero variance. ARC can't "
                         "search for trends in such constant series.")
    data = data/normfacs
    if dataerrs is not None:
        dataerrs = dataerrs/normfacs

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
        entropy_cut = np.max(entropies)/keep_fac
        keep = entropies > entropy_cut
        weights = weights[keep, :]

        # compute the leading trends from the weights
        ys, errs = construct(data, weights, dataerrs)

        # find the principle componenet of the retined trends
        trend, rho = principle_component(ys, errs)

        # estimate trend error
        # NOTE: the errors for each trend in ys are HIGHLY CORRELATED
        # as propagating errors through PCA is already confusing, I'll just
        # estimate by taking the minumum error of any given trend.
        trenderr = np.min(errs, axis=1, keepdims=True)

        # break out of loop if we've reached a trend that has too low a
        # spectral radius (doesn't explain the data well)
        if rho < rho_min:
            break

        # denoise the trend, if desired
        if denoise is not None:
            trend = denoise(trend)

        # normalize the trend
        trend -= np.median(trend)
        normfac = np.std(trend)
        trend = trend/normfac

        # record the trend
        trends = np.append(trends, trend, axis=1)
        if dataerrs is not None:
            trenderr = trenderr/normfac
            trenderrs = np.append(trenderrs, trenderr, axis=1)

        if Ntrends and len(trends == Ntrends):
            break

        # fit and subtract all the trends identified so far from the data
        data = [trend_remove(d, trends)[0] for d in original_data.T]
        data = np.array(data).T

    return trends, trenderrs

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

def principle_component(trends, trends_err=None):
    """
    Find the principal compenent of a set of trends (principle component
    analysis).

    Parameters
    ----------
    trends : 2D array-like
        A set of trends (or any series) with shape NxM where there are M trends
        of N data points each.
    trends_err : 2D array-like, optional
        Errors on the trends.

    Returns
    -------
    pc : 1D array
        The principle component of the trends as a len(N) array.
    pc_err : 1D array
        Propagated error on the principal component.
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

def construct(basisset, weights, basiserrs=None):
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
    basiserrs : 2D array-like
        Errors of the series to be propagated.

    Returns
    -------
    y : 1D or 2D array
        The data series constructed from the basisset and weights. If
        weights is 2D, y will be 2d with shape NxK, otherwise it will be 1D
        with length N.
    yerr : 1D or 2D array
        If basiserrs is supplied, yerr will also be output with the same shape
        as y that represents a propagation of the erros in the basisset data
        (assuming the weights to have no error). Otherwise, it will be None.
    """
    b, w = map(np.asarray, [basisset, weights])
    if w.ndim == 1:
        w = w.reshape(1, len(w))

    y = np.dot(w, b.T).T

    if basiserrs is not None:
        sigs = np.asarray(basiserrs)
        sig2s = sigs**2
        ysig2s = np.dot(w**2, sig2s.T).T
        yerr = np.sqrt(ysig2s)
    else:
        yerr = None

    y = np.squeeze(y)
    if basiserrs is not None:
        yerr = np.squeeze(yerr)

    return y, yerr

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
    w = np.asarray(weights)
    w = np.squeeze(w)
    w = w.T
    axis = 0 if w.ndim > 1 else None
    p = w**2/np.sum(w**2, axis=axis)
    H = -np.nansum(p*np.log2(p), axis=axis)
    return H

def trend_remove(data, trends, data_err=None, trend_errs=None):
    """
    Fit and remove the provided trends from the data.

    Parameters
    ----------
    data : 1D array-like
        The data to fit, length N.
    trends : 2D array-like
        The trends to fit to the data, dimensions NxM where there are M trends.
    dataerr = 1D array-like, optional
        Data errors.
    trenderrs : 2D array-like, optional
        Trend errors.

    Returns
    -------
    detrended : 1D array
        The data with the trend subtracted, length N.
    trendfit : 1D array
        The fit of the trend to the data, length N.
    detrended_err : 1D array
        Errors on the detrended data. None if errors aren't supplied.
    trendfit_err : 1D array
        Errors on the trend fit. None if errors aren't supplied.

    Notes
    -----
    If daterr or trenderrs is supplied, the other must also be supplied or
    errors cannot be propagated.
    """
    if ((data_err is None and trend_errs is not None) or
        (data_err is not None and trend_errs is None)):
            raise ValueError('If one of daterr and trenderrs is supplied '
                             'then both must be supplied.')
    err = (data_err is not None)
    if trends.shape[1] == 0: raise ValueError("No trends supplied.")

    # median subtract and mean-normalize the data
    med = np.median(data)
    fac = np.std(data)
    data = (data - med)/fac

    # fit the data
    weights = basisweights(data, trends)
    trendfit, trendfit_err = construct(trends, weights, trend_errs)
    detrended = data - trendfit
    detrended_err = np.sqrt(data_err**2 + trendfit_err**2) if err else None

    # undo the median subtract and normalization
    restore = lambda x: (x * fac) + med
    detrended, trendfit = map(restore, [detrended, trendfit])
    if err:
        detrended_err, trendfit_err = map(restore, [detrended_err, trendfit_err])

    return detrended, trendfit, detrended_err, trendfit_err

def arc_emd(y, min_rel_std=0.9):
    """
    Denoise the data in y by returning the intrinsic mode (or residual) with
    the largest variance, but only if it has a standard deviation at least
    min_rel_std of the original.

    Parameters
    ----------
    y : 1D array-like
        The data to be denoised.
    min_rel_std : float, optional
        The minimum allowable standard deviation of the returned series
        relative to the original. This is intended to prevent the denoising
        step of ARC from removing removing any real trend in the data.

    Result
    ------
    y_denoised : 1D array
        The denoised data, or the original data if attempting to denoise
        resulted in a series with std deviation < min_rel_std.
    """

    modes, residual = emd.emd()