import matplotlib.pyplot as plt
from scipy import constants as cst, integrate as intg
import numpy as np

from constants import TAU, PEAK_FREQ
import distributions as dist
import utilities as ut

def dist1d():
    om0 = 2*np.pi*PEAK_FREQ
    k = om0/cst.speed_of_light

    num = 501
    vt = ut.temp2vel(300+273)
    v = np.linspace(-vt*3, vt*3, num=num)    
    dv = v[1] - v[0]

    omt = ut.temp2vel(300+273) * k
    om = np.linspace(-omt, omt, num=num)    
    dom = om[1] - om[0]

    omv_max = omt * 3
    omv = np.linspace(om0-omv_max, om0+omv_max, num=num)
    domv = omv[1] - omv[0]

    # distributions
    maxboltz = dist.maxwell_boltzmann_1d(v, vt)
    natlinwidth = dist.natural_linewidth(om, 1/TAU)
    voigt = dist.gas_dist1d(omv, 0, vp=vt, theta_L=0.0, tau=TAU,
                        om0=2*np.pi*PEAK_FREQ, num_om=2**10+1).flatten() # voigt

    # integrals of distributions
    int_mb = intg.trapezoid(maxboltz, dx=dv)
    int_nlw = intg.trapezoid(natlinwidth, dx=dom)
    int_voi = intg.trapezoid(voigt, dx=domv)
    print(f'Maxwell-Boltzmann scaling speed: {vt} ms⁻¹')
    print('Maxwell-Boltzmann integral: \t' + str(int_mb))
    print('natural linewidth integral: \t' + str(int_nlw))
    print('Voigt profile integral: \t' + str(int_voi))

    # FWHM checks

    # plots

    fig, ax = plt.subplots(1,2)
    ax[0].plot(v/vt, maxboltz)
    ax[0].set_title('Maxwell-Boltzmann')
    ax[1].plot(om, natlinwidth)
    ax[1].set_title('Natural linewidth')
    ax[0].set_xlabel('v/vp')
    ax[1].set_xlabel('ω - ω₀')

    fig, ax = plt.subplots(1,1)
    ax.plot(omv-om0, voigt, label='Voigt')
    ax.set_xlabel('ω - ω₀')
    ax.set_title('Voigt profile')

    plt.show()

def dist3d():
    num = 501
    vt = ut.temp2vel(300+273)
    v = np.linspace(0, vt*3, num=num)    
    dv = v[1] - v[0]

    # distributions
    maxboltz = v**2 * dist.maxwell_boltzmann_3d(v, vt)
    natlinwidth = 2 * np.abs(ut.get_alphainv(0)) * dist.natural_linewidth(v*ut.get_alphainv(0), 1/TAU)

    # integrals of distributions
    int_mb = intg.trapezoid(maxboltz, dx=dv)
    int_nlw = intg.trapezoid(natlinwidth, dx=dv)
    print(f'Maxwell-Boltzmann scaling speed: {vt} ms⁻¹')
    print('Maxwell-Boltzmann integral: \t' + str(int_mb))
    print('natural linewidth integral: \t' + str(int_nlw))

    # plots

    #fig, ax = plt.subplots(1,2)
    #ax[0].plot(v/vt, maxboltz)
    #ax[0].set_title('Maxwell-Boltzmann')
    #ax[1].plot(v/vt, natlinwidth)
    #ax[1].set_title('Natural linewidth')
    #ax[0].set_xlabel('v/vp')
    #ax[1].set_xlabel('v/vp')

    fig, ax = plt.subplots(1,1)
    ax.plot(v/vt, voigt, label='Voigt')
    ax.plot(v/vt, maxboltz, label='Maxwell-Boltzmann')
    #ax.plot(v, natlinwidth, label='Natural linewidth')
    ax.set_xlabel('v/vp')
    ax.legend()
    ax.set_title('Voigt profile')

    plt.show()

if __name__ == '__main__':
    dist1d()
    #dist3d()

