from math import nan
from os import error, listdir, path

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgl, scipy.constants as cst, scipy.optimize as opt

#data_dir = '../data/20250212_Ca-neutral_ImproveFluoSignal' # single peak
data_dir = '../data/20250213_Ca-neutral_doublePeak' # double peak

# constants
TAU = 4.6e-9 # 423 nm transition lifetime [s]
GAMMA = 1/TAU
CA40_MOLAR_MASS = 39.9625908 # g/mol https://pubchem.ncbi.nlm.nih.gov/compound/Calcium-40
CA40_MASS = CA40_MOLAR_MASS * 1e-3 / cst.Avogadro
PEAK_FREQ = 709.07855e12 # Hz
I_SAT = 45 # W/m²

# values to define voltage - freq conversion
ref_V = [-0.1, 1.9]
#ref_f = [709.07726e12, 709.07979e12] # double peak
ref_f = [709.07717e12, 709.07984e12] # single peak

# laser intensity
I_LASER = 10.73e-3 / np.pi*(5e-3)**2 # W/m² (power/detector area)

# useful quantities

vw2 = lambda T, m=CA40_MASS: 2*cst.Boltzmann*T / m
vw_to_T = lambda vw, m=CA40_MASS: m*vw**2 / (2*cst.Boltzmann**2)
om_to_f = lambda om: 0.5*om / np.pi
f_to_om = lambda f: 2*np.pi * f

def normal_to_fit_params(f0, theta, T):
    f0_dif = (f0 - PEAK_FREQ)*1e-9
    T100 = T/100
    return f0_dif, theta, T100

def fit_params_to_normal(f0_dif, theta, T100):
    f0 = f0_dif*1e9 + PEAK_FREQ
    T = T100*100
    return f0, theta, T

def get_FWHM(f, ct):
    """
    gets FWHM from normalized counts
    """
    peak_idx = ct.argmax()
    _, _, left_pos, right_pos = sgl.peak_widths(ct, [peak_idx])
    [left_f, right_f] = np.interp([left_pos[0], right_pos[0]], np.arange(f.size), f)
    fwhm = right_f - left_f
    return fwhm

# preprocessing

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

# expected distribution

def maxwell_boltzmann(om, om0, T, theta, m=CA40_MASS):
    return (om-om0)**2 * np.exp(-(cst.speed_of_light * (om - om0)/(om0 * np.cos(theta)))**2 * 1/vw2(T, m=m))

def positive_mb(om, om0, T, theta, m=CA40_MASS):
    return (om > om0) * maxwell_boltzmann(om, om0, T, theta, m)

def lorentzian(om, om0, I=I_LASER, I_S=I_SAT, gam=GAMMA):
    return 1/(1 + I/I_S + 4*((om-om0)/gam)**2)

def P_om(om, om0, T, theta, m=CA40_MASS, I=I_LASER, I_S=I_SAT, gam=GAMMA):
    unnorm = lorentzian(om, om0, I, I_S, gam) * maxwell_boltzmann(om, om0, T, theta, m)
    max_val = unnorm.max()
    return unnorm / max_val if max_val != 0 else unnorm 

# fit to distribution

def fit_func(f, f0_dif, theta, T100):
    """
    f0_dif: [GHz] such that f0 = 709.07855 Hz + f0_dif
    theta: angle between atomic beam and laser beam
    T100: [1e2 K] such that T = T100*100
    alpha: angle subtended by oven opening 
    phi: viewing angle w.r.t. laser beam
    """
    om = f_to_om(f)
    f0, theta, T = fit_params_to_normal(f0_dif, theta, T100)
    return P_om(om, f_to_om(f0), T, theta)

if __name__ == "__main__":
    idx = 1
    V, f, ct = read_preprocess(data_dir, ref_V, ref_f, 0.02)
    
    fit_results = opt.curve_fit(fit_func, f[idx], ct[idx],
                                p0=[-0.25, 0.1, 6],
                                bounds=([-np.inf, 0, 4],
                                        [np.inf, 2*np.pi, 8]))
    fit_f0, fit_theta, fit_T = fit_params_to_normal(*fit_results[0])
    fit_thetad = np.rad2deg(fit_theta)
    fit_counts = fit_func(f[idx], *fit_results[0])
    #fit_counts = fit_func(f[idx], 0, np.pi/4, 1)
    
    print(f'f0 = {fit_f0*1e-12} THz\ntheta = {fit_thetad}°\nT = {fit_T-273.15} °C' )
    
    # compare fwhm with natural linewidth
    fwhm = get_FWHM(f[idx], ct[idx])
    lw = 1 / (2*np.pi * TAU)
    #print(f'lw/FWHM = {lw/fwhm}')
    
    plt.plot(f[idx], ct[idx])
    plt.plot(f[idx], fit_counts)
    plt.show()
