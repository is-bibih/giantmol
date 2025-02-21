from scipy import signal as sgl, constants as cst
import matplotlib.pyplot as plt
import numpy as np

import fit_single_peak as utils

data_dir = '../data/20250213_Ca-neutral_doublePeak' # double peak

OVEN_T = 329.9+273.15 # K

# find angle from peaks (and temperature)

def find_angle(f, ct, T, m=utils.CA40_MASS):
    peak_width = 50 
    expected_v = np.sqrt(2*cst.Boltzmann*T / m)
    largest_counts = ct * (ct > 0.8)
    peaks_idx = sgl.find_peaks_cwt(largest_counts, peak_width) 
    peak_dif = f[peaks_idx[1]] - f[peaks_idx[0]]
    theta = np.arccos(peak_dif / (2*k*expected_v))


idx = 1
V, f, ct, = utils.read_preprocess(data_dir, utils.ref_V, utils.ref_f, 0.02)

find_angle(f[idx], ct[idx], OVEN_T)

plt.plot(np.arange(ct[idx].size), ct[idx])
plt.show()

