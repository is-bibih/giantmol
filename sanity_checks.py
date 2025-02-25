import matplotlib.pyplot as plt
import scipy.integrate as intg
import numpy as np

from constants import TAU, PEAK_FREQ
import distributions as dist
import utilities as ut

def dist1d(v, vt):
    dv = v[1] - v[0]

    # distributions
    maxboltz = dist.maxwell_boltzmann_1d(v, vt)
    natlinwidth = np.abs(ut.get_alphainv(0)) * dist.natural_linewidth(v*ut.get_alphainv(0), 1/TAU)
    voigt = dist.gas_dist1d(v, 0, vp=vt, theta_L=0.0, tau=TAU,
                        om0=2*np.pi*PEAK_FREQ, v_max=vt*3, num_v=2**10+1).flatten() # voigt

    # integrals of distributions
    int_mb = intg.trapezoid(maxboltz, dx=dv)
    int_nlw = intg.trapezoid(natlinwidth, dx=dv)
    int_voi = intg.trapezoid(voigt, dx=dv)
    print(f'Maxwell-Boltzmann scaling speed: {vt} ms⁻¹')
    print('Maxwell-Boltzmann integral: \t' + str(int_mb))
    print('natural linewidth integral: \t' + str(int_nlw))
    print('Voigt profile integral: \t' + str(int_voi))

    # plots

    fig, ax = plt.subplots(1,2)
    ax[0].plot(v/vt, maxboltz)
    ax[0].set_title('Maxwell-Boltzmann')
    ax[1].plot(v/vt, natlinwidth)
    ax[1].set_title('Natural linewidth')
    ax[0].set_xlabel('v/vp')
    ax[1].set_xlabel('v/vp')

    fig, ax = plt.subplots(1,1)
    ax.plot(v/vt, voigt, label='Voigt')
    ax.plot(v/vt, maxboltz, label='Maxwell-Boltzmann')
    #ax.plot(v, natlinwidth, label='Natural linewidth')
    ax.set_xlabel('v/vp')
    ax.legend()
    ax.set_title('Voigt profile')

    plt.show()

if __name__ == '__main__':
    num = 501
    vt = ut.temp2vel(300+273)
    v = np.linspace(-vt*3, vt*3, num=num)    
    dist1d(v, vt)

