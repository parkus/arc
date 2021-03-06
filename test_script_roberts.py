import arc.arc_functions as arc # so that reload(arc) works, but note that you can also just do "import arc"
import numpy as np
import numpy.random as r
from math import pi

# for repeatability, seed the random number generator
r.seed(42)

## GENERATE DATA

# create data that are sines of random period, amplitude, and phase
N = 1000 # number of data points
M = 200 # number of data series
t = np.arange(N)

# generate random amplitudes, periods, and phases
amps = r.normal(1.0, 0.3, M)
phases = r.uniform(0.0, 2*pi, M)
periods = r.uniform(4.0, 4.0 * N, M)

# compute sine curves and make NxM data array
make_curve = lambda a, P, ph: a * np.sin(2*pi * t / P + ph)
data_list = map(make_curve, amps, periods, phases)
data = np.transpose(data_list)

# add noise
rel_amps = r.uniform(high=0.05, size=M)
rel_noise = r.normal(size=[N, M]) * rel_amps[np.newaxis, :]
abs_noise = rel_noise * amps[np.newaxis, :]
data = data + abs_noise

## CREATE TRENDS

# function to mean subtract and std dev normalize
norm = lambda y: (y - np.mean(y)) / np.std(y)

# exponential decay
exp = norm(np.exp( t / (N * 1.0)))

# quadratic
quad = norm((t - N/2.0)**2)

## INJECT TRENDS

def trend_array(rel_amps, trend):
    abs_amps = rel_amps * amps
    trend_list = [trend * a for a in abs_amps]
    return np.transpose(trend_list)

i = np.arange(M)
decaying_amps = 5.0 * np.exp(-i / (M / 4))
quad_arr = trend_array(decaying_amps, quad)
exp_arr = trend_array(2 * decaying_amps[::-1], exp)
trended = data + quad_arr + exp_arr

# injection function
#def inject(data, trend, amp_mean, amp_std):
#    shape = (amp_mean/amp_std)**2
#    scale = amp_mean/shape
#    rel_amps = r.gamma(shape, scale, M)
#    trends = trend_array(rel_amps, trend)
#    return data + trends
##    rel_amps = r.normal(amp_mean, amp_std, M)
#
#
## inject!
#trended = np.copy(data)
#trended = inject(trended, exp, 2.0, 1.0)
#trended = inject(trended, quad, 1.0, 0.5)

## FIND TRENDS IN SUBSET OF DATA
subset = trended[:, r.choice(M, size=50)]
trends = arc.arc(t, subset, rho_min=0.6, denoise=arc.arc_emd, refine=False)

## PLOT TRENDS
import matplotlib.pyplot as plt
plt.figure()
plt.plot(t, trends)
plt.plot(exp, '--', quad, '--')
plt.title('injected (solid) and retrieved (dashed) trends')

## FIT TRENDS TO A SERIES
i = r.choice(range(M))
y0 = data[:, i]
y1 = trended[:, i]
detrended, trendfit = arc.trend_remove(y1, trends)
plt.figure()
plt.plot(t, np.transpose(y0, y1, trendfit, detrended))
plt.legend(['original data', 'data with trend', 'best-fit trend', 'detrended data'])