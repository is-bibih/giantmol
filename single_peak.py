from os import listdir, path

import utilities as ut
import constants as cst

import numpy as np
import scipy.signal as sgl
from scipy.constants import speed_of_light
import matplotlib.pyplot as plt

data_dir = './data/20250219_Ca-neutral_trap-axis'
fnames = [f for f in listdir(data_dir) if path.isfile(path.join(data_dir, f))]
data = [ut.read_counts_freq(path.join(data_dir, f), freq_correction=cst.FREQ_CORRECTION) for f in fnames]
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

# width and peak
widths = np.zeros((N,))
heights = np.zeros((N,))
half_maximum_pos = np.zeros((N,2))
peak_freq = np.zeros((N,))
for i in range(N):
    f = freqs[i]
    c = counts[i]
    c = sgl.savgol_filter(c, window_length=int(c.size * 0.07), polyorder=3) # smooth
    c = (c/c.max()).flatten() # normalize
    # find all peaks (there's a lot from the noise)
    all_peak_idx, props = sgl.find_peaks(c, width=100, height=(None, None))
    peak_heights = props['peak_heights']
    # choose the highest
    highest_idx = np.argmax(peak_heights)
    peak_idx = all_peak_idx[highest_idx]
    # get fwhm
    left_idx = props['left_ips']
    right_idx = props['right_ips']
    [left_f, right_f] = np.interp([left_idx[0], right_idx[0]], np.arange(f.size), f)
    fwhm = right_f - left_f
    # store
    widths[i] = fwhm
    heights[i] = c[peak_idx] * counts[i].max()
    half_maximum_pos[i] = np.array([left_f, right_f])
    peak_freq[i] = f[peak_idx]

# convert to speeds
vp_from_width = (cst.TRANSITION_WAVELENGTH_AIR * cst.n_AIR) * widths / (2 * np.sqrt(np.log(2)))
vp_from_peak = np.abs(peak_freq - cst.TRANSITION_FREQ) * (cst.TRANSITION_WAVELENGTH_AIR * cst.n_AIR)
print(peak_freq - cst.TRANSITION_FREQ)

# calculated temperatures

# -------------------------

freq_scale = 1e9
freq_units = 'GHz'

# counts
fig, ax = plt.subplots(1,1)
for i in range(N):
    ax.scatter(freqs[i] / freq_scale, counts[i], marker='.')
    ax.scatter(half_maximum_pos[i] / freq_scale, [heights[i]/2, heights[i]/2], marker='*', s=100)
ax.set_xlabel(f'frequency ({freq_units})')
ax.set_ylabel('counts / ms')

# v_p cos(theta) from peaks
fig, ax = plt.subplots(1,1)
ax.set_title('from peak position')
ax.scatter(temps, vp_from_peak, label='experimental data')
ax.plot(temps, ut.temp2vel(temps) * np.cos(np.deg2rad(75)),
        label='speed from oven temp,\n$\\theta = 75\\degree$')
ax.set_xlabel('oven temperature (K)')
ax.set_ylabel(r'$v_p$cos$(\theta_{OL})$ most probable speed (ms⁻¹)')
ax.legend()

# v_p cos(theta) from width
fig, ax = plt.subplots(1,1)
ax.set_title('from width')
ax.scatter(temps, vp_from_width, label='experimental data')
ax.plot(temps, ut.temp2vel(temps) * np.cos(np.deg2rad(75)),
        label='speed from oven temp,\n$\\theta = 75\\degree$')
ax.plot(temps, ut.temp2vel(temps) * np.cos(np.deg2rad(79)),
        label='speed from oven temp,\n$\\theta = 79\\degree$')
ax.set_xlabel('oven temperature (K)')
ax.set_ylabel(r'$v_p$cos$(\theta_{OL})$ most probable speed (ms⁻¹)')
ax.legend()

plt.show()
