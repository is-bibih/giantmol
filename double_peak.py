from os import listdir, path

import utilities as ut
import constants as cst

import numpy as np
import scipy.signal as sgl
from scipy.constants import speed_of_light
import matplotlib.pyplot as plt

data_dir = './data/20250213_Ca-neutral_double-peak'
fnames = [f for f in listdir(data_dir) if path.isfile(path.join(data_dir, f))]
data = [ut.read_counts_freq(path.join(data_dir, f)) for f in fnames]
N = len(fnames)

# store data in usable form
counts = [dat[0] for dat in data]
freqs = [dat[1] for dat in data]
temps = ut.cel2kel(np.array([ut.fname2temp(f) for f in fnames])) # convert to kelvin

# sort by temperature
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
peak_idx = np.zeros((N,2), dtype=int) 

for i in range(N):
    threshold = 0.5
    c = (counts[i]/counts[i].max()).flatten() # normalize
    c = sgl.savgol_filter(c, window_length=int(c.size * 0.07), polyorder=3) # smooth
    # remove lower values for easier peak search
    high_values = c > threshold
    first_high = high_values.nonzero()[0][0]
    first_low = np.nonzero(np.logical_not(high_values[first_high:]))[0][0] + first_high
    high_values[first_low:] = np.zeros((len(high_values)-first_low))
    c = c * high_values
    # find all peaks (there's a lot from the noise)
    all_peak_idx, props = sgl.find_peaks(c, height=(None, None), width=100)
    peak_heights = props['peak_heights']
    # choose the highest two
    highest_idx = np.argpartition(peak_heights, -2)[-2:]
    # store
    this_peak_idx = all_peak_idx[highest_idx]
    peak_idx[i,:] = this_peak_idx
    # calculate distances
    peak_distances[i] = np.abs(freqs[i][this_peak_idx[1]] - freqs[i][this_peak_idx[0]])

peak_speeds = 0.5 * peak_distances * (speed_of_light/cst.TRANSITION_FREQ)

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
for i in range(N): ax.scatter(freqs[i][peak_idx[i]] * xaxis_scale, counts[i][peak_idx[i]], marker='*', c='red', s=80)
ax.set_xlabel('frequency (GHz)')
ax.set_ylabel('counts / ms')
fig.legend()

fig, ax = plt.subplots(1,1)
ax.scatter(temps, valley_freqs * xaxis_scale)
ax.set_xlabel('oven temperature (K)')
ax.set_ylabel('transition frequency (GHz)')

fig, ax = plt.subplots(1,1)
ax.scatter(temps, peak_speeds, label='experimental data')
ax.plot(temps, ut.temp2vel(temps) * np.cos(np.deg2rad(75)),
        label='speed from oven temp,\n$\\theta = 75\\degree$')
ax.plot(temps, ut.temp2vel(temps*0.2) * np.cos(np.deg2rad(75)),
        label='speed from oven temp · 0.2,\n$\\theta = 75\\degree$')
ax.set_xlabel('oven temperature (K)')
ax.set_ylabel(r'$v_p$cos$(\theta_{OL})$ most probable speed (ms⁻¹)')
ax.legend()

plt.show()

