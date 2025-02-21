from scipy import constants as cst, integrate as intg
import matplotlib.pyplot as plt
import numpy as np

from fit_single_peak import CA40_MASS, PEAK_FREQ, TAU
from fit_single_peak import read_preprocess
PEAK_FREQ = PEAK_FREQ-0.5e9

V_MAX = np.sqrt(2*cst.Boltzmann*1e3/CA40_MASS)
V_P = np.sqrt(2*cst.Boltzmann*(318+273)/CA40_MASS)

get_alpha = lambda th, om0=PEAK_FREQ*2*np.pi, theta_L=0: - cst.c/(om0*np.cos(th - theta_L))
get_alphainv = lambda th, om0=PEAK_FREQ*2*np.pi, theta_L=0: - om0*np.cos(th - theta_L)/cst.c

def maxwell_boltzmann_3d(x, a=1.0):
    return 1/(a**3 * np.pi**(1.5)) * np.exp(-(x/a)**2)

def natural_linewidth(x, gam=1.0):
    return gam/(2*np.pi) / (x**2 + (gam/2)**2)

def excited_population(x, sat=1.0, gam=1.0):
    return sat/2 / (1 + sat + (2*x/gam)**2)

def P_v(v, th, vp=V_P, theta_L=0, om0=2*np.pi*PEAK_FREQ,
        tau=TAU, num_theta=2**8+1, num_v=2**8+1, v_max=V_MAX,
        integration_method='romb'):
    """
    v: speed
    theta_L: angle between laser beam (k-vector) and the center of the atom beam
    om0: transition frequency
    num_theta: number of elements to use for integral along theta (odd)
    num_v: number of elements to use for integral along v (odd)
    """

    alphainv = lambda th: get_alphainv(th, om0, theta_L)
    gam = 1/tau

    # resize arrays to have dimensions corresponding to (v, th, vpr, thpr)
    # shape = [v.size, th.size, num_v, num_theta]
    v = v.reshape((v.size,1,1,1)) 
    th = th.reshape((1,th.size,1,1))
    vpr = np.linspace(0, v_max, num=num_v).reshape((1,1,num_v,1)) # v prime
    thpr = np.linspace(-np.pi/2, np.pi/2, num=num_theta).reshape((1,1,1,num_theta)) # theta prime

    # do integration
    dth = thpr[0,0,0,1] - thpr[0,0,0,0]
    dv = vpr[0,0,1,0] - vpr[0,0,0,0]
    integration_array = vpr**2 * np.abs(np.sin(thpr)) * maxwell_boltzmann_3d(vpr, vp) * natural_linewidth(v*alphainv(th) - vpr*alphainv(thpr), gam)
    integrate = intg.romb
    if integration_method == 'trapezoid':
        integrate = intg.trapezoid
    elif integration_method == 'simpson':
        integrate = intg.simpson
    integration_array = integrate(integration_array, dx=dth, axis=-1) # along theta
    integration_array = integrate(integration_array, dx=dv, axis=-1) # along v
    return integration_array

def signal_v(v, th, vp=V_P, theta_L=0, om0=2*np.pi*PEAK_FREQ,
             sat=1.0, tau=TAU, num_theta=2**8+1, num_v=2**8+1, v_max=V_MAX,
             integration_method='romb'):

    alphainv = get_alphainv(th, om0, theta_L)
    P = P_v(v, th, vp, theta_L, om0, tau, num_theta,
            num_v, v_max, integration_method)
    v = v.reshape((v.size, 1))
    N = excited_population(v*alphainv, sat=sat, gam=1/tau)

    return N*P

def signal_f(f, th, vp=V_P, theta_L=0, om0=2*np.pi*PEAK_FREQ,
             sat=1.0, tau=TAU, num_theta=2**8+1, num_v=2**8+1, v_max=V_MAX,
             integration_method='romb'):
    f = f.reshape((f.size, 1))
    th = th.reshape((1, th.size))
    alpha = get_alpha(th, om0, theta_L)
    v = alpha * (2*np.pi*f - om0)
    return signal_v(v, th, vp, theta_L, om0, sat, tau, num_theta, num_v, v_max)

if __name__ == '__main__':
    ref_V = [-1.2, 0.5]
    ref_f = [709.07714e12, 709.07902e12]
    data_dir = '../data/20250219_Ca-neutral_TrapAxis'
    V, f, ct = read_preprocess(data_dir, ref_V, ref_f, 0)
    idx = 0
    f_ct = f[idx]
    ct = ct[idx]

    #v = np.linspace(0, V_P*1.5, num=300)
    freq_delta = 3e-7
    f = np.linspace(PEAK_FREQ*(1-freq_delta), PEAK_FREQ*(1+freq_delta), num=300)
    #f = np.linspace(f_ct[0], f_ct[-1], num=300)
    th = np.array([0])
    vp = 500
    P = signal_f(f, th, vp=vp, integration_method='trapezoid')
    P = P.flatten()/P.max()

    fig, ax = plt.subplots(1,2)
    ax[0].plot((f-PEAK_FREQ)*1e-9, P)
    ax[1].plot((f_ct-PEAK_FREQ)*1e-9, ct)
    ax[0].set_xlabel('f-f0 [GHz]')
    ax[1].set_xlabel('f-f0 [GHz]')
    plt.show()

