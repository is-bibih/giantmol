import matplotlib.pyplot as plt
import scipy.integrate as intg
import numpy as np

from constants import TAU
import distributions as dist
import utilities as ut

if __name__ == '__main__':
    num = 201
    vt = ut.temp2vel(300+273)
    v = np.linspace(-vt, vt, num=num)    
    th = np.array([0])

    # check for normalization
    # distribution
    P = dist.signal_v(v, th, vp=vt, theta_L=np.deg2rad(75), sat=1.0, tau=TAU,
                    integration_method='trapezoid', asym=False).flatten()
    print(P.shape)
    # get integral
    Pnorm = intg.simpson(P, v)
    print(Pnorm)

    plt.plot(v, P)
    plt.show()
