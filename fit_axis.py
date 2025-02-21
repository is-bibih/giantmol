from os import listdir

from scipy import constants as cst, optimize as opt
import matplotlib.pyplot as plt
import numpy as np

from fit_single_peak import CA40_MASS, PEAK_FREQ, TAU
from fit_single_peak import read_preprocess
from oven_dist import signal_f

data_dir = '../data/20250219_Ca-neutral_TrapAxis'
temps = np.array([float(fname[0:2] + '.' + fname[4]) for fname in listdir(data_dir)]) + 273.15
ref_V = [-1.2, 0.5]
ref_f = [709.07714e12, 709.07902e12]

temp2vel = lambda T, m=CA40_MASS: np.sqrt(2*cst.Boltzmann*T/m)
vel2temp = lambda v, m=CA40_MASS: m*v**2 / (2*cst.Boltzmann)
freq2om = lambda f: 2*np.pi*f
om2freq = lambda om: 0.5*om/np.pi

PEAK_OM = freq2om(PEAK_FREQ)

def transform_params(vp, theta_L, om0):
    om0_dif = (om0 - PEAK_OM)*1e-9
    vp100 = vp/100
    return vp100, theta_L, om0_dif

def untransform_params(vp100, theta_L, om0_dif):
    om0 = om0_dif*1e9 + PEAK_OM
    vp = vp100*100
    return vp, theta_L, om0

def fit_func(f, vp100, theta_L, om0_dif, th=np.array([0])):
    """
    vp100: [1e2 K] most probable speed such that vp = vp100*100
    theta_L: angle between atomic beam and trap axis
    om0_dif: [rad GHz] such that om0 = 2pi * 709.07855 Hz + om0_dif
    """
    vp, theta_L, om0 = untransform_params(vp100, theta_L, om0_dif)
    res = (signal_f(f, th, vp, theta_L, om0, tau=TAU)).flatten()
    maxval = res.max()
    return res if maxval == 0 else res/maxval

if __name__ == '__main__':
    V, f, ct = read_preprocess(data_dir, ref_V, ref_f, 0)

    idx = 0 # index of dataset
    f = f[idx]
    ct = ct[idx]

    # take subset for fit
    n_fit = 100
    sample_delta = np.astype(np.floor(f.size / n_fit), int)
    f_sample = f[0:-1:sample_delta]
    ct_sample = ct[0:-1:sample_delta]

    thL0 = np.deg2rad(75)
    vp0 = temp2vel(temps[idx])
    om0 = freq2om(PEAK_FREQ)

    initial_params = transform_params(vp0, thL0, om0)
    param_bounds = ( \
        [0, -np.pi/2, -np.inf], # lower bounds
        [np.inf, np.pi/2, np.inf]) # upper bounds

    #fit_results = opt.curve_fit(fit_func, f_sample, ct_sample,
    #                            p0=initial_params, bounds=param_bounds)
    #vp_fit, thL_fit, om_fit = fit_results[0]
    #fit_counts = fit_func(f_sample, vp_fit, thL_fit, om_fit)
    fit_counts = fit_func(f_sample, *initial_params)
    fit_counts = fit_counts / fit_counts.max()

    plt.plot(f_sample, fit_counts)
    plt.plot(f, ct)
    plt.show()
