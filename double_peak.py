from math import nan
from os import error, listdir, path

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgl, scipy.constants as cst, scipy.optimize as opt

data_dir = '../data/20250213_Ca-neutral_doublePeak'
single_peak_dir = '../data/20250212_Ca-neutral_ImproveFluoSignal'

# constants
TAU = 4.6e-9 # 423 nm transition lifetime [s]
CA40_MOLAR_MASS = 39.9625908 # g/mol https://pubchem.ncbi.nlm.nih.gov/compound/Calcium-40
CA40_MASS = CA40_MOLAR_MASS * 1e-3 / cst.Avogadro
PEAK_FREQ = 709.07855e12 # Hz

# values to define voltage - freq conversion
ref_V = [-0.1, 1.9]
ref_f = [709.07726e12, 709.07979e12] # double peak
ref_fs = [709.07717e12, 709.07984e12] # single peak

# read count data
def read_data(data_dir, ref_V, ref_f):
    fnames = [f for f in listdir(data_dir) if path.isfile(path.join(data_dir, f))]
    data = [np.loadtxt(path.join(data_dir, fname)) for fname in fnames]
    get_freq = lambda V: (ref_f[1] - ref_f[0]) / (ref_V[1] - ref_V[0]) * (V - ref_V[0]) + ref_f[0]

    voltages = [d[:,4] for d in data] 
    counts = [d[:,1] for d in data] 
    freqs = [get_freq(V) for V in voltages]

    return voltages, freqs, counts

# get temperatures from file names
temps = [float(fname[0:2] + '.' + fname[4]) for fname in listdir(data_dir)]

# normalize and set background to 0
def clean(ct):
    ct = ct - np.min(ct)
    ct = ct/np.max(ct)
    return ct

# find FWHM
def get_FWHM(f, ct):
    """
    gets FWHM from normalized counts
    """
    peak_idx = ct.argmax()
    _, _, left_pos, right_pos = sgl.peak_widths(ct, [peak_idx])
    [left_f, right_f] = np.interp([left_pos[0], right_pos[0]], np.arange(f.size), f)
    fwhm = right_f - left_f
    return fwhm

# doppler broadening
def displaced_mb(om, om0, theta, T, m):
    result = np.zeros(om.size)
    idx = om > om0
    result[idx] = (om[idx] - om0)**2 * np.exp(-0.5*m/(cst.Boltzmann*T) * (cst.speed_of_light * (om[idx] - om0) / (om0*np.cos(theta)))**2 )
    max_val = result.max()
    return result / max_val if max_val != 0 else result

def atomic_beam(om, om0, theta, T, m):
    result = (om)**3 * np.exp(-0.5*m/(cst.Boltzmann*T) * (cst.speed_of_light * (om - om0) / (om0*np.cos(theta)))**2 )
    max_val = result.max()
    return result / max_val if max_val != 0 else result

def doppler(om, om0, theta, T, m):
    result = np.exp(-0.5*m/(cst.Boltzmann*T) * (cst.speed_of_light * (om - om0) / (om0*np.cos(theta)))**2 )
    max_val = result.max()
    return result / max_val if max_val != 0 else result

# read and clean
V, f, ct = read_data(data_dir, ref_V, ref_f) # double peak
clean_ct = [clean(c) for c in ct]
Vs, fs, cts = read_data(single_peak_dir, ref_V, ref_fs) # single peak
clean_cts = [clean(c) for c in cts]

# doppler fit
#fit_f = lambda nu, nu0, theta, T, alpha, phi: atomic_beam(0.5*nu/np.pi, 0.5*nu0/np.pi, theta, T, alpha, phi, CA40_MASS)
def fit_f(nu, nu0_dif, theta, T100):
    """
    nu0_dif: [GHz] such that nu0 = 709.07855 Hz + nu0_dif
    theta: angle between atomic beam and laser beam
    T100: [1e2 K] such that T = T100*100
    alpha: angle subtended by oven opening 
    phi: viewing angle w.r.t. laser beam
    """
    om = 0.5*nu/np.pi
    om0 = 0.5/np.pi * (nu0_dif*1e9 + PEAK_FREQ)
    T = T100*100
    return atomic_beam(om, om0, theta, T, CA40_MASS)

fit_results = opt.curve_fit(fit_f, fs[0], clean_cts[0],
                            p0=[0, 0, 2],
                            bounds=([0, 0.0, 0],
                                    [np.inf, 2*np.pi, np.inf]))
print(fit_results)
fit_counts = fit_f(fs[0], *fit_results[0])
#fit_counts = fit_f(fs[0], 0, 0.1, 0.2)

# compare fwhm with natural linewidth
fwhm = get_FWHM(fs[0], clean_cts[0])
lw = 1 / (2*np.pi * TAU)

#for freq, counts in zip(f, clean_ct):
#    plt.plot(freq, counts)
#for freq, counts in zip(fs, clean_cts):
#    plt.plot(freq, counts)
#for counts in test_counts:
#    plt.plot(fs[0], counts)
plt.plot(fs[0], clean_cts[0])
plt.plot(fs[0], fit_counts)
plt.show()
