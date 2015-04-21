ARC
===
*Astrophysically Robust Correction*

This code can be used to detect and remove "global" trends in an ensemble of identically-sampled data. For example, it can be used to find trends in light curve data when the user has many light curves that have the same time sampling (such as data from the Kepler spacecraft). It is an implementation of the algorithm published by [Roberts et al. (2013 MNRAS 435:3639)](http://adsabs.harvard.edu/abs/2013MNRAS.435.3639R).

The code works by searching for patterns that are present in many different data series. It attempts to separate trends that don't affect the same light curves by the same degree. For example, perhaps there is a slow ramp-up of signal strongly present in light curves from one side of the detector while curves of the opposite side show a strong sinusoid. However, at the moment this doesn't work very well. I'm not sure if I have improperly implemented the algorithm or the algorithm itself is to blame.

## Modifications from Roberts et al.
I have made the "denoising" step flexible (and optional). The user can supply his or her own denoising function if desired. The default is empirical mode decomposition (EMD), which I wrote as a separate module -- https://github.com/parkus/emd. To use the default, you must have my emd module somewhere on your system where Python will find it. 


Currently (2015/04/16) the code behaves consistently with what is described in Roberts et al. I plan to eventually write and test three possible improvements
1. (currently testing) the "refine" option -- a means of removing artifacts from the denoising by iteratively fitting the data to the denoised trend to produce a new trend until the trend converges. 
1. A faster version of EMD. The code bottlenecks at the spline envelope fitting required for EMD.
2. A non-emd means of separating out different trends by fitting and subtracting trends from one another. This idea isn't very well-formed yet. I've found that if I inject two different trends into the data by different amounts, the algorithm ends up finding some trends that are some combination of the two. By fitting them to each other, and then taking those as a new set of candidate trends, I might be able to better separate the injected trends without relying on a "denoting" step (EMD), which I distrust.


## Example Use
(see test_script.py for a heinously complicated example)

```python
import arc
import numpy as np
import numpy.random as r
from math import pi

# for repeatability, seed the random number generator
r.seed(42)

# create data that are sines of random period, amplitude, and phase
N = 500 # number of data points
M = 200 # number of data series
t = np.arange(N)

amps = 10**r.uniform(0.0, 2.0, M) # make the amplitudes span orders of magnitude
phases = r.uniform(0.0, 2*pi, M)
periods = r.uniform(4.0, 4.0 * N, M)
data_list = [a * np.sin(2*pi * t / P + ph) for a, P, ph in zip(amps, periods, phases)]
data = np.transpose(data_list)

trends = arc.arc(
```