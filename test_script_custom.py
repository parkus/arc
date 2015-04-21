"""
Some commands for testing out the ARC code.
"""

import numpy as np
import numpy.random as r
from math import pi, log10
import arc
import matplotlib.pyplot as plt

# for repeatability, seed the random number generator
r.seed(42)

N = 500 # number of data points
M = 200 # number of data series
t = np.arange(N)


#%% ---------------------------------------------------------------------------
# CREATE SOME FAKE TRENDS

# let's start by creating some fake trends
# polynomial
deg = r.randint(1, 5) # of degree 1-4
coeffs = r.uniform(size=deg+1) #with random coefficients
poly = np.polyval(coeffs, t)

# high frequency sinusoid
phase = r.uniform(0.0, 2*pi)
period = r.uniform(N/50.0, N/5.0)
sine = np.sin(2 * pi * t / period + phase)

# step
loc = r.choice(t)
step = np.zeros(N)
step[loc:] = 1.0

# normalize both to have zero mean and unit std dev
def normit(y):
    z = y - np.mean(y)
    z = z / np.std(z)
    return z

poly, sine = map(normit, [poly, sine])

#%% ---------------------------------------------------------------------------
# CREATE SOME FAKE DATA

# start with uniform series
data0 = np.ones([N, M])

# at random levels spanning three orders of magnitude
levels = 10.0**r.uniform(0.0, 3.0, [M])
data0 = data0*levels[np.newaxis, :]

# randomly replace a few series with special signals. record which ones have
indices = np.arange(M)
choose_series = lambda n: list(r.choice(indices, size=n, replace=False))

# how about some sines to start
Nsines = 2
phases = r.uniform(0.0, 2*pi, Nsines)
periods = 10.0**r.uniform(log10(N/100.0), log10(N), Nsines)
amps = 10.0**r.uniform(-2.0, -1.0, Nsines)
i_sines = choose_series(Nsines)
for i, a, P, phi in zip(i_sines, amps, periods, phases):
    data0[:, i] += levels[i] * a * np.sin(2 * pi * t / P + phi)

# replace a few with boxcar "transits"
Nboxes = 2
amps = 10.0**r.uniform(-3, -1, Nboxes)
widths = r.randint(1, N / 2, Nboxes)
starts = r.randint(0, N - 1, Nboxes)
ends = starts + widths
i_boxes = choose_series(Nboxes)
for i, a, j, k in zip(i_boxes, amps, starts, ends):
    data0[j:k, i] = levels[i] * (1.0 - a)

# and a few with "flares"
Nflares= 2
amps = 10.0**r.uniform(-2, 0.5, Nflares)
decay_consts = r.randint(1, N / 2, Nflares).astype('float')
starts = r.randint(0, N - 1, Nflares)
i_flares = choose_series(Nflares)
for i, a, tau, j in zip(i_flares, amps, decay_consts, starts):
    length = N - j
    flare_x = np.arange(length)
    flare = a * levels[i] * np.exp(-flare_x / tau)
    data0[j:, i] += flare

#%% ---------------------------------------------------------------------------
# INJECT THE TRENDS INTO THE DATA

# function to inject the trends into some fraction of the data
def inject(func, max_amp, inject_frac, data):
    d = np.copy(data)
    K = round(inject_frac * M)
    # I never decided what would be the best way to randomize amplitudes
    amps = r.power(3.0, [1, K]) * max_amp
    trends = np.dot(func.reshape([N, 1]), amps) # gives NxM output array
    trends += 1.0
    i = r.randint(0, M-1, K)
    d[:, i] *= trends
    return d

# try commenting in/out different trends to see the effects
data1 = np.copy(data0)
data1 = inject(poly, 0.2, 0.7, data1)
data1 = inject(sine, 0.1, 0.7, data1)
#data1 = inject(step, 0.3, 1.0, data1)

#%% ---------------------------------------------------------------------------
# ADD SOME NOISE

# NOTE: This is necessary. Otherwise the constant series will have std. dev.
# of zero, which the ARC can't handle.

noise_amps = np.sqrt(levels) * 0.05
data_err = np.outer(np.ones(N), noise_amps) # makes an NxM array of errors
noise = noise_amps[np.newaxis, :] * r.normal(size=[N, M])
data2 = data1 + noise

#%% ---------------------------------------------------------------------------
# IDENTIFY TRENDS WITH THE ARC
trends = arc.arc.arc(t, data2, rho_min=0.5)

# NOTE: A lower rho_min than that recommended by Roberts et al. is necessary
# to detect the trends in these data. Users should experiment with different
# values of rho_min to find what works best with their dataset.

#%% ---------------------------------------------------------------------------
# COMPARE THE RETRIEVED WITH THE INJECTED TRENDS
#plt.figure()

#trendin = (1.0 + poly) * (1.0 + sine * (sineamp/polyamp))

p1, = plt.plot(t, poly, label='injected polynomial')
p2, = plt.plot(t, sine, label='injected sine')
#p3, = plt.plot(t, trendin, label='product of injected signals')

for tr, tre in zip(trends.T, trend_errs.T):
     plt.errorbar(t, tr, tre, fmt='k:', errorevery=N/20)

plt.title ("comparison of injected (solid) and retrieved (dotted) trends")
plt.legend((p1, p2))
plt.show()

#%% ---------------------------------------------------------------------------
# DETREND ALL THE DATA

# detrending all the data gives a length M list of 4-element tuples
results = [arc.arc.trend_remove(d, trends, de, trend_errs)
           for d, de in zip(data2.T, data_err.T)]

# zip into 4 length M tuples of length N arrays
results = zip(*results)

# then turn those into NxM arrays. sorry that this is ugly.
reform = lambda a: np.array(a).T
detrended, trendfits, detrended_errs, trenfit_errs = map(reform, results)

#%% ---------------------------------------------------------------------------
# COMPARE NOISE LEVELS
# compute various errors

# the injected noise level (standard deviation)
std_true_0 = noise_amps
# measured level of injected noise
std_msrd_0 = np.std(noise, axis=0)
# estimated level of noise after detrending (should be a slight overestimate)
std_est_1 = np.sqrt(np.mean(detrended_errs**2, axis=0))
#measured level of noise after detrending
true_errs = detrended - data0
std_msrd_1 = np.std(true_errs, axis=0)

#plot a comparison
plt.figure()
plt.plot(std_true_0, 'o', label='actual std dev of injected noise')
plt.plot(std_msrd_0, 'o', label='measured std dev of injected noise')
plt.plot(std_est_1, 'o', label='estimated std dev of ARC introduced noise')
plt.plot(std_msrd_1, 'o', label='measured std dev of ARC introduced errors')
plt.xlabel('Data Series No.')
plt.ylabel('Standard Deviation')
plt.title('Comparison of Errors Before and After ARC')
plt.legend()
plt.show()

#%% ---------------------------------------------------------------------------
# PLOT THE SPECIAL CASES AND A FEW OF THE CONSTANT-CURVE CASES

i_special = i_sines + i_boxes + i_flares + choose_series(2)
for i in i_special:
    plt.figure()
    plt.plot(t, data0[:, i], label='original data')
    plt.plot(t, data2[:, i], label='data with injected trends and noise')
    plt.plot(t, trendfits[:, i], label='fit trend')
    plt.plot(t, detrended[:, i], label='detrended data')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()

plt.show()