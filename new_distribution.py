import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as cst

from constants import TAU, PEAK_FREQ
import distributions as dist
import utilities as ut

if __name__ == '__main__':
    om0 = 2*np.pi*PEAK_FREQ
    k = om0/cst.speed_of_light
    T = 500
    sat = 0.1

    num = 301
    vp = ut.temp2vel(T)
    v = np.linspace(-vp*3, vp*3, num=num)    

    omt = vp * k
    om_max = omt * 3
    om = np.linspace(om0-om_max, om0+om_max, num=num)

    th = np.array([0])
    test_angles = np.deg2rad(np.linspace(0, 90, num=7)).astype(float)
    test_openings = np.deg2rad(np.linspace(1, 89, num=7)).astype(float)

    max_angle = np.atan(3/11.7) # ~14.38°
    #max_angle = np.pi/2 * 0.3
    params = {
        'vp': vp,
        'om0': om0,
        'sat': sat,
        'tau': TAU,
        'theta_OL': np.deg2rad(15),
        'num_om': 2**8+1,
        'om_max': None,
        'num_theta': 2**8+1,
        'theta_min': -max_angle,
        'theta_max': +max_angle,
    }
    signal = dist.new_signal(om, th, **params)
    signal_perp = dist.new_signal(om, th, **{**params, 'theta_OL': np.pi/2})
    angle_signals = np.array([dist.new_signal(om, th, **{**params, 'theta_OL': thOL}) \
                              for thOL in test_angles]).reshape((test_angles.size, num))
    opening_signals = np.array([dist.new_signal(om, th, **{**params, 'theta_min': -to, 'theta_max': +to}) \
                              for to in test_openings]).reshape((test_openings.size, num))
    voigt_gas = dist.voigt3d_gas(om, th, **params)
    voigt_beam = dist.voigt3d_beam(om, th, **params)

    fig, ax = plt.subplots(1,1)
    #ax.plot(om-om0, signal / signal.max(), label='convolution 0')
    #ax.plot(om-om0, signal_perp / signal_perp.max(), label='convolution 90')
    #ax.plot(om-om0, voigt_gas / voigt_gas.max(), label='voigt gas')
    #ax.plot(om-om0, voigt_beam / voigt_beam.max(), label='voigt beam')
    ax.plot(om-om0, signal, label=r'truncated convolution ($\theta_{OL}$ = ' + str(params['theta_OL']) + ')')
    ax.plot(om-om0, angle_signals[0,:], label=r'truncated convolution ($\theta_{OL}$ = 0)')
    ax.plot(om-om0, signal_perp, label=r'truncated convolution ($\theta_{OL}$ = $\pi/2$)')
    ax.plot(om-om0, voigt_gas, label='voigt gas')
    ax.plot(om-om0, voigt_beam, label='voigt beam')
    ax.set_xlabel('ω - ω₀')
    ax.set_title('signal')
    ax.legend()

    fig, ax = plt.subplots(1,1)
    ax.set_xlabel('ω - ω₀')
    ax.set_title('signal for different opening angles')
    for i in range(test_angles.size):
        #ax.plot(om-om0, angle_signals[i,:]/angle_signals[i,:].max(),
        ax.plot(om-om0, opening_signals[i,:],
                label=r'$\theta_{max} = $' + str(np.rad2deg(test_openings[i])))
    ax.legend()

    fig, ax = plt.subplots(1,1)
    ax.set_xlabel('ω - ω₀')
    ax.set_title('signal for different oven-laser angles')
    for i in range(test_angles.size):
        #ax.plot(om-om0, angle_signals[i,:]/angle_signals[i,:].max(),
        ax.plot(om-om0, angle_signals[i,:],
                label=r'$\theta_{OL} = $' + str(np.rad2deg(test_angles[i])))
    ax.legend()

    plt.show()

