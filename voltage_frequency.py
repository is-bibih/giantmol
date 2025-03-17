from os import listdir, path

import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import speed_of_light

data_dir = './data/20250313_Ca-neutral_freqMeasurements'
ignore = ['test.lta']
header = 'Time  [ms]	Signal 1 Power  [µW]	Signal 2 Power  [µW]	Signal 1  Wavelength, vac.  [nm]	Signal 2  Wavelength, vac.  [nm]'

# read file names
fnames = [f for f in listdir(data_dir) \
          if path.isfile(path.join(data_dir, f)) \
          and f[-4:] == '.lta' \
          and not f in ignore]

def lta_to_dat(data_dir=data_dir, fnames=fnames, header=header):
    # read and write to dat files
    for fn in fnames:
        out_fname = fn[:-4] + '.dat'
        reading_data = False
        with open(path.join(data_dir, fn), 'r', encoding='latin-1') as read_file:
            with open(path.join(data_dir, out_fname), 'w+') as out_file:
                for line in read_file:
                    if reading_data:
                        out_file.write(' '.join(line.split()) + '\n')
                    if line.strip() == header.strip():
                        reading_data = True

def read_static_voltages(data_dir=data_dir, fname='static.txt'):
    data = {}
    with open(path.join(data_dir, fname), 'r') as file:
        for line in file:
            details = line.split()
            data[details[0]] = {
                'filename': details[0],
                'V_low': float(details[1]),
                'freq_low': float(details[2]),
                'V_zero': float(details[3]),
                'freq_zero': float(details[4]),
                'V_high': float(details[5]),
                'freq_high': float(details[6]),
                'timestep': details[7],
            }           
    return data

def read_dat(fname, data_dir=data_dir):
    data = np.loadtxt(path.join(data_dir, fname))
    time_ms = data[:,0]
    wl_vacuum_nm = data[:,2]
    freq = speed_of_light / (wl_vacuum_nm * 1e-9)
    return time_ms, freq

def compare(static_data, freq):
    fmin = freq.min()
    fmax = freq.max()
    gain = (static_data['V_high'] - static_data['V_low']) / (fmax - fmin)
    low_dif = fmin - static_data['freq_low']
    high_dif = fmax - static_data['freq_high']
    return gain, low_dif, high_dif

if __name__ == '__main__':
    static_data = read_static_voltages()
    times = []
    freqs = []
    gains = []
    low_difs = []
    high_difs = []
    freq_ranges = [static_data[name]['freq_high']-static_data[name]['freq_low'] \
                  for name in [name[:-4] for name in fnames]]
    volt_ranges = [static_data[name]['V_high']-static_data[name]['V_low'] \
                  for name in [name[:-4] for name in fnames]]
    timesteps = [static_data[name]['timestep'] \
                  for name in [name[:-4] for name in fnames]]

    for fname in fnames:
        time, freq = read_dat(fname[:-4] + '.dat')
        gain, low_dif, high_dif = compare(static_data[fname[:-4]], freq)

        times.append(time)
        freqs.append(freq)
        gains.append(gain)
        low_difs.append(low_dif)
        high_difs.append(high_dif)

    print(f'gain:\t\tmean = {np.average(gains) * 1e9} V/MHz \tstdev = {np.std(gains) * 1e9} V/MHz')
    print(f'lower gap:\tmean = {np.average(low_difs) * 1e-9} MHz \t\tstdev = {np.std(low_difs) * 1e-9} MHz')
    print(f'upper gap:\tmean = {np.average(high_difs) * 1e-9} MHz \tstdev = {np.std(high_difs) * 1e-9} MHz')

    """
    gain:		mean = 0.9370261250701449 V/MHz, 	stdev = 0.05979943360828218 V/MHz
    lower gap:	mean = 0.17876736585227274 MHz, 	stdev = 0.0999835948907794 MHz
    upper gap:	mean = -0.10634362604545455 MHz, 	stdev = 0.04986648303810322 MHz
    """

    # ~~~~~~~~~~ plots ~~~~~~~~~~

    # gain vs. range
    fig, ax = plt.subplots(1,2)
    ax[0].set_xlabel('frequency range (Hz)')
    ax[0].set_ylabel('gain (Hz/V)')
    ax[1].set_xlabel('voltage range (V)')
    ax[1].set_ylabel('gain (Hz/V)')
    for i in range(len(fnames)):
        ax[0].scatter(freq_ranges[i], gains[i])
        ax[1].scatter(volt_ranges[i], gains[i])

    # gain vs. time step
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel('time step (ms)')
    ax.set_ylabel('gain (Hz/V)')
    ax.scatter(timesteps, gains)

    # lower and upper gap vs. range
    fig, ax = plt.subplots(1,2)
    ax[0].set_xlabel('frequency range (Hz)')
    ax[0].set_ylabel('gap (Hz)')
    ax[0].scatter(freq_ranges, low_difs, label='lower')
    ax[0].scatter(freq_ranges, high_difs, label='higher')
    ax[0].legend()
    ax[1].set_xlabel('voltage range (V)')
    ax[1].set_ylabel('gap (Hz)')
    ax[1].scatter(volt_ranges, low_difs, label='lower')
    ax[1].scatter(volt_ranges, high_difs, label='higher')
    ax[1].legend()

    plt.show()

