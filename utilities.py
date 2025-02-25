from math import nan
from os import error, listdir, path

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgl, scipy.constants as cst, scipy.optimize as opt

from constants import *

get_alpha = lambda th, om0=PEAK_FREQ*2*np.pi, theta_L=0: - cst.c/(om0*np.cos(th - theta_L))
get_alphainv = lambda th, om0=PEAK_FREQ*2*np.pi, theta_L=0: - om0*np.cos(th - theta_L)/cst.c
temp2vel = lambda T, m=CA40_MASS: np.sqrt(2*cst.Boltzmann*T/m)
vel2temp = lambda v, m=CA40_MASS: m*v**2 / (2*cst.Boltzmann)
freq2om = lambda f: 2*np.pi*f
om2freq = lambda om: 0.5*om/np.pi

def read_data(data_dir, ref_V, ref_f):
    fnames = [f for f in listdir(data_dir) if path.isfile(path.join(data_dir, f))]
    data = [np.loadtxt(path.join(data_dir, fname)) for fname in fnames]
    get_freq = lambda V: (ref_f[1] - ref_f[0]) / (ref_V[1] - ref_V[0]) * (V - ref_V[0]) + ref_f[0]

    voltages = [d[:,4] for d in data] 
    counts = [d[:,1] for d in data] 
    freqs = [np.array(get_freq(V)) for V in voltages]

    return voltages, freqs, counts

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

def read_preprocess(data_dir, ref_V, ref_f, threshold=0.0):
    V, f, ct = read_data(data_dir, ref_V, ref_f)
    ct = [clean(c) for c in ct]
    if threshold:
        for i in range(len(ct)):
            ct[i], f[i], V[i] = remove_small(ct[i], threshold, f[i], V[i]) 
    return V, f, ct

