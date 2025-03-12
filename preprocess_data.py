import numpy as np
from os import path

from utilities import read_raw_data

# reads data and writes it as a function of frequency and with counts / ms

# data for double peak
data_dir = './data-voltage/20250213_Ca-neutral_double-peak'
save_dir = './data/20250213_Ca-neutral_double-peak'
data = read_raw_data(data_dir)
for freq, ct, fname in zip(data['frequencies'], data['counts'], data['filenames']):
    data = np.array([freq, ct]).T
    np.savetxt(path.join(save_dir, fname), data)

# data for laser along trap axis
data_dir = './data-voltage/20250219_Ca-neutral_trap-axis'
save_dir = './data/20250219_Ca-neutral_trap-axis'
data = read_raw_data(data_dir)
for freq, ct, fname in zip(data['frequencies'], data['counts'], data['filenames']):
    data = np.array([freq, ct]).T.flatten()
    np.savetxt(path.join(save_dir, fname), data)

