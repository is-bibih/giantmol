import re
from os import listdir, path

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgl, scipy.constants as cst, scipy.optimize as opt

from constants import *

get_alpha = lambda th, om0=PEAK_FREQ*2*np.pi, theta_L=0: - cst.c/(om0*np.cos(th - theta_L))
get_alphainv = lambda th, om0=PEAK_FREQ*2*np.pi, theta_L=0: - om0*np.cos(th - theta_L)/cst.c
temp2vel = lambda T, m=CA40_MASS: np.sqrt(2*cst.Boltzmann*T/m)
vel2temp = lambda v, m=CA40_MASS: m*v**2 / (2*cst.Boltzmann)
cel2kel = lambda c: c+273.15
kel2cel = lambda k: k-273.15
freq2om = lambda f: 2*np.pi*f
om2freq = lambda om: 0.5*om/np.pi

fname2temp = lambda fname: float(f'{m.group(1)}.{m.group(2)}') \
    if (m := re.match(r"(\d+)-(\d)C", fname)) else None

TIME_STEP_FILENAME = 'timestep.txt'
VOLTAGE_FREQUENCY_FILENAME = 'voltage_frequency.dat'

def read_raw_data(data_dir):
    fnames = [f for f in listdir(data_dir) \
              if path.isfile(path.join(data_dir, f)) \
              and not (f == TIME_STEP_FILENAME or f == VOLTAGE_FREQUENCY_FILENAME)]
    volt_freq = np.loadtxt(path.join(data_dir, VOLTAGE_FREQUENCY_FILENAME))
    timestep = np.loadtxt(path.join(data_dir, TIME_STEP_FILENAME))

    data = [np.loadtxt(path.join(data_dir, fname)) for fname in fnames]

    ref_V = volt_freq[:,0]
    ref_f = volt_freq[:,1]

    get_freq = lambda V: (ref_f[1] - ref_f[0]) / (ref_V[1] - ref_V[0]) * (V - ref_V[0]) + ref_f[0]

    voltages = [d[:,4] for d in data] 
    counts = [d[:,1] / timestep for d in data] 
    freqs = [np.array(get_freq(V)) for V in voltages]
    temps = [fname2temp(t) for t in fnames]

    return {
        'voltages': voltages, 
        'counts': counts,
        'frequencies': freqs,
        'timestep': timestep,
        'temperatures': temps,
        'filenames': fnames,
    }

def read_counts_freq(fname):
    data = np.loadtxt(fname)
    freq = data[:,0]
    counts = data[:,1]
    return counts, freq

def clean(ct):
    """normalize and set background to zero"""
    ct = ct - np.min(ct)
    ct = ct/np.max(ct)
    return ct

def remove_small(ct, threshold, *other_quantities):
    """remove small values in base"""
    idx = ct >= threshold
    ct = ct[idx]
    return ct, *[qty[idx] for qty in other_quantities]

def read_preprocess(data_dir, threshold=0.0):
    V, f, ct = read_raw_data(data_dir)
    ct = [clean(c) for c in ct]
    if threshold:
        for i in range(len(ct)):
            ct[i], f[i], V[i] = remove_small(ct[i], threshold, f[i], V[i]) 
    return V, f, ct


def get_FWHM(x, y):
    """
    gets FWHM from function
    """
    peak_height = y.max()
    peak_idx = y.argmax()
    left_pos = np.argmin(np.abs(y[0:peak_idx] - 0.5*peak_height))
    right_pos = peak_idx + np.argmin(np.abs(y[peak_idx:-1] - 0.5*peak_height))
    left_x, right_x = x[left_pos], x[right_pos]
    fwhm = np.abs(right_x - left_x)
    return fwhm, (left_x, right_x)

