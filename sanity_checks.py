import matplotlib.pyplot as plt
import scipy.integrate as intg
import numpy as np

from fit_single_peak import CA40_MASS, PEAK_FREQ, TAU
from fit_single_peak import read_preprocess
import oven_dist as od

if __name__ == '__main__':
    num = 201
    vt = od.temp2vel(300+273)
    v = np.linspace(-vt, vt, num=num)    
    th = np.array([0])

    # check for normalization
    # distribution
    P = od.signal_v(v, th, vp=vt, theta_L=np.deg2rad(75), sat=1.0, tau=TAU,
                    integration_method='trapezoid', asym=False).flatten()
    print(P.shape)
    # get integral
    Pnorm = intg.simpson(P, v)
    print(Pnorm)

    plt.plot(v, P)
    plt.show()
