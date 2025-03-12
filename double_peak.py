from os import listdir, path
from re import X

import utilities as ut
import constants as cst

import numpy as np
import scipy.signal as sgl
import matplotlib.pyplot as plt

data_dir = './data/20250213_Ca-neutral_double-peak'
fnames = [f for f in listdir(data_dir) if path.isfile(path.join(data_dir, f))]
data = [ut.read_counts_freq(path.join(data_dir, f)) for f in fnames]
N = len(fnames)

# store data in usable form
counts = [dat[0] for dat in data]
freqs = [dat[1] for dat in data]
temps = ut.cel2kel(np.array([ut.fname2temp(f) for f in fnames])) # convert to kelvin

# reorder by temperature
idx = np.argsort(temps)
counts = [counts[i] for i in idx]
freqs = [freqs[i] for i in idx]
temps = np.take(temps, idx)

# remove background (it changed a lot between readings)
counts = [c - c.min() for c in counts]

# find center between peaks (unshifted absorbtion freq)
valley_freqs = np.zeros((N,))
for i in range(N):
    c = counts[i]
    third_length = int(len(c)/3)
    center_idx = np.arange(third_length, third_length*2)
    # first valley of width third_length*0.1 (the noise creates narrow spikes in the data) 
    valley_idx = sgl.find_peaks_cwt(-c[center_idx], third_length*0.1)[0] + third_length

    valley_freqs[i] = freqs[i][valley_idx]

mean_trans_freq = valley_freqs.mean()

# distance between peaks

peak_distances = np.zeros((N,))
all_peak_idx = []
for i in range(N):
    c = counts[i]
    all_peak_idx.append(sgl.find_peaks_cwt(c, np.arange(1,400)))
    all_peak_idx[i] = all_peak_idx[i][all_peak_idx[i] > 100]
    print(all_peak_idx[i])
    highest_idx = np.argpartition(c[all_peak_idx[i]], -2)[-2:]
    peak_idx = all_peak_idx[i][highest_idx]
    print('results')
    print(peak_idx)

# ------------------------------------------------------------------------------------------

# print transition frequency info
Hz_scale = 1e6
print(f'mean transition frequency: {mean_trans_freq/Hz_scale:.2f} (MHz)')
print(f'standard deviation: {valley_freqs.std()/Hz_scale:.2f} (MHz)')
print(f'known transition frequency: {cst.TRANSITION_FREQ/Hz_scale} (MHz)')
print(f'relative error: {np.abs(mean_trans_freq - cst.TRANSITION_FREQ)/cst.TRANSITION_FREQ:.2e}')
print(f'absolute error: {np.abs(mean_trans_freq - cst.TRANSITION_FREQ)/Hz_scale:.2f} (MHz)')

# plotting

xaxis_scale = 1e-9

fig, ax = plt.subplots(1,1)
for i in range(N):
    ax.scatter(freqs[i] * xaxis_scale, counts[i], label=f'{temps[i]:.2f} K', marker='.')
    #ax.plot(np.arange(len(counts[i])), counts[i], label=f'{temps[i]:.2f} K')
for i in range(N): ax.scatter(freqs[i][all_peak_idx[i]] * xaxis_scale, counts[i][all_peak_idx[i]], marker='*', c='red', s=80)
ax.set_xlabel('frequency (GHz)')
ax.set_ylabel('counts / ms')
fig.legend()

fig, ax = plt.subplots(1,1)
ax.scatter(temps, valley_freqs * xaxis_scale)
ax.set_xlabel('temperature (K)')
ax.set_ylabel('valley frequency (GHz)')

plt.show()

