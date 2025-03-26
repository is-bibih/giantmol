from os import path

import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, signal as sgl

from utilities import get_filelist

# read

datadir = './data/power-monitoring'
filelist = get_filelist(datadir)
N = len(filelist)

sample_spacing = 1 # s

datalist = []
timelist = []
labellist = []

for fname in filelist:
    # the files are edited to remove the header, assumes 1 measurement per second
    data = np.loadtxt(path.join(datadir, fname))
    labellist.append(fname[:2])
    datalist.append(data if len(data.shape) == 1 else data[:,1])
    timelist.append(np.arange(0.0, datalist[-1].size))

# variation

means = np.array([d.mean() for d in datalist])
minmax = np.array([d.max() - d.min() for d in datalist])
stdevs = np.array([d.std() for d in datalist])
rel_minmax = minmax / means
rel_stdevs = stdevs / means

# periodogram

results = [sgl.periodogram(d, sample_spacing) for d in datalist]
freqs = [f for f, _ in results]
Ps = [p for _, p in results]

# fourier transforms

fts = [np.array(fft.fft(d)) for d in datalist]
ftfreqs = [fft.fftfreq(d.size, sample_spacing) for d in datalist]

# largest contributions

plot_fts = [np.abs(fft.fftshift(ft)) for ft in fts]
plot_freqs = [fft.fftshift(freq) for freq in ftfreqs]

"""
for i in range(N):

    # we only care about the smaller frequencies (slower variations) and no duplicates
    small_idx = (plot_freqs[i] < 1) * (plot_freqs[i] > 0)
    plot_fts[i] = plot_fts[i][small_idx]
    plot_freqs[i] = plot_freqs[i][small_idx]

    # keep highest for plotting
    n_highest = 100
    idx = np.argpartition(plot_fts[i], -n_highest)[-n_highest:]
    plot_fts[i] = plot_fts[i][idx]
    plot_freqs[i] = plot_freqs[i][idx]

    # 10 highest relative weights
    total = np.abs(fts[i]).sum()
    rel_weights = plot_fts[i] / total
    highest = np.argsort(plot_fts[i])[-10:]

    print(f'\nfor {filelist[i]}:')
    print(f'\tstandard deviation:\t{stdevs[i] : .4f} V ({rel_stdevs[i] * 100 : .2f}% rel. to mean)')
    print(f'\tmax - min:\t\t{minmax[i] : .4f} V ({rel_minmax[i] * 100 : .2f}% rel. to mean)')
    print(f'\tperiods with highest contributions:')
    for idx in highest: print(f'\t{1/plot_freqs[i][idx] / 60 : .2f} min' +
                              f' \t\t{rel_weights[idx] * 100 : .2f}%')
"""

# plot

half_hour_freq = 1/(60*30)
minute_freq = 1/60

xlabel = 'time (s)'
ylabel = 'voltage (V)'

fig, ax = plt.subplots(1,1)
for i in range(N):
    ax.plot(timelist[i], datalist[i], label=labellist[i])
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.legend()

"""
xlabel = 'frequency (Hz)'
ylabel = 'F{voltage}'

fig, ax = plt.subplots(1,1)
#ax.plot([half_hour_freq, half_hour_freq], [-1, 10], label='30 minutes', alpha=0.7)
#ax.plot([minute_freq, minute_freq], [-1, 10], label='1 minute', alpha=0.7)
for i in range(1):
    ax.scatter(plot_freqs[i], plot_fts[i], label=labellist[i], marker='.')
#ax.set_ylim(-1, 10)
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.semilogx()
ax.legend()
"""

xlabel = 'frequency (Hz)'
ylabel = 'F{voltage} (V²/Hz)'

fig, ax = plt.subplots(1,1)
for i in range(N):
    ax.semilogx(freqs[i], Ps[i], label=labellist[i])
ax.legend()
plt.show()
