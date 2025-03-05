import matplotlib.pyplot as plt
from scipy import constants as cst, integrate as intg
import numpy as np

from constants import TAU, PEAK_FREQ
import distributions as dist
import utilities as ut

def dist1d():
    om0 = 2*np.pi*PEAK_FREQ
    k = om0/cst.speed_of_light
    T = 500
    sat = 0.1

    num = 501
    vt = ut.temp2vel(T)
    v = np.linspace(-vt*3, vt*3, num=num)    
    dv = v[1] - v[0]

    omt = vt * k
    om_max = omt
    om = np.linspace(-om_max, om_max, num=num)    
    dom = om[1] - om[0]

    omv_max = omt * 3
    #omv_max = omt * 20
    omv = np.linspace(om0-omv_max, om0+omv_max, num=num)
    domv = omv[1] - omv[0]

    # distributions
    maxboltz = dist.maxwell_boltzmann_1d(v, vt)
    natlinwidth = dist.excited_population(om, sat, 1/TAU)
    voigt = dist.voigt1d(omv, vp=vt, sat=sat, tau=TAU,
                        om0=2*np.pi*PEAK_FREQ, num_om=2**10+1).flatten() # voigt
    compare_pop = dist.excited_population(omv-om0, sat, 1/TAU)
    compare_pop = compare_pop * voigt.max() / compare_pop.max()

    # integrals of distributions
    int_mb = intg.trapezoid(maxboltz, dx=dv)
    print(f'Maxwell-Boltzmann scaling speed: {vt} ms⁻¹')
    print('Maxwell-Boltzmann integral: \t' + str(int_mb))

    # FWHM checks
    fwhm_mb, (mb_l, mb_r) = ut.get_FWHM(v, maxboltz)
    fwhm_nl, (nl_l, nl_r) = ut.get_FWHM(om, natlinwidth)
    fwhm_vg, (vg_l, vg_r) = ut.get_FWHM(omv, voigt)
    # theoretical
    fwhm_mb_t = 2*np.sqrt(np.log(2))*vt
    fwhm_nl_t = np.sqrt(1+sat)/TAU
    fwhm_vg_t = fwhm_mb_t * k
    # rel error
    er_vg = np.abs((fwhm_vg-fwhm_vg_t)/fwhm_vg_t)
    er_nl = np.abs((fwhm_nl-fwhm_nl_t)/fwhm_nl_t)
    er_mb = np.abs((fwhm_mb-fwhm_mb_t)/fwhm_mb_t)
    print(f'Maxwell-Boltzmann FWHM: \t{fwhm_mb:.5e} ms⁻¹\t\t(theoretical: {fwhm_mb_t:.5e} ms⁻¹, \trel error: {er_mb})')
    print(f'excited population FWHM: \t{fwhm_nl:.5e} rad·s⁻¹\t\t(theoretical: {fwhm_nl_t:.5e} rad·s⁻¹, \trel error: {er_nl})')
    print(f'Voigt profile FWHM: \t\t{fwhm_vg:.5e} rad·s⁻¹\t\t(MB theoretical: {fwhm_vg_t:.5e} rad·s⁻¹, \trel error: {er_vg})')

    # plots

    fig, ax = plt.subplots(1,2)
    ax[0].plot(v/vt, maxboltz)
    ax[0].set_title('Maxwell-Boltzmann')
    #ax[0].plot([mb_l/vt, mb_l/vt], [0, 0.0012])
    #ax[0].plot([mb_r/vt, mb_r/vt], [0, 0.0012])
    ax[1].plot(om, natlinwidth)
    ax[1].set_title('Natural linewidth')
    #ax[1].plot([nl_l, nl_l], [0, 3e-9])
    #ax[1].plot([nl_r, nl_r], [0, 3e-9])
    ax[0].set_xlabel('v/vp')
    ax[1].set_xlabel('ω - ω₀')

    fig, ax = plt.subplots(1,1)
    ax.plot(omv-om0, voigt, label='Voigt')
    ax.plot(omv-om0, compare_pop, label='static atom')
    ax.set_xlabel('ω - ω₀')
    ax.legend()
    ax.set_title('Voigt profile')

    plt.show()

def dist3d(voigt_fun):
    om0 = 2*np.pi*PEAK_FREQ
    k = om0/cst.speed_of_light
    T = 1000
    #T = 1e-5
    temps = np.linspace(50, T, num=5)
    angles = np.linspace(0, np.pi/2, num=7)
    sat = 0.1

    num = 501
    vt = ut.temp2vel(T)
    vmax = vt*6
    v = np.linspace(0, vmax, num=num)    
    dv = v[1] - v[0]

    omt = vt * k
    om_max = omt / 5
    om = np.linspace(-om_max, om_max, num=num)    
    dom = om[1] - om[0]

    omv_max = omt * 3
    #omv_max = om0 * 1e-7
    omv = np.linspace(om0-omv_max, om0+omv_max, num=num)
    domv = omv[1] - omv[0]

    theta = np.array([0])

    # distributions
    maxboltz = dist.maxwell_boltzmann_3d(v, vt)
    beam = dist.beam(v, vt)
    population = dist.excited_population(om, sat, 1/TAU)
    voigt = voigt_fun(omv, theta, vp=vt, om0=om0, sat=sat, tau=TAU)

    # for comparison
    compare_pop = dist.excited_population(omv-om0, sat, 1/TAU)
    compare_pop = compare_pop * voigt.max() / compare_pop.max()
    compare_temps = [voigt_fun(omv, theta, vp=ut.temp2vel(T),
                                      om0=om0, sat=sat, tau=TAU) for T in temps]
    #compare_temps = [voigt / voigt.max() for voigt in compare_temps]
    compare_angles = [voigt_fun(omv, theta, vp=vt,
                                      om0=om0, sat=sat, tau=TAU) for theta in angles]

    # integrals of distributions
    int_mb = intg.trapezoid(maxboltz, dx=dv)
    int_bm = intg.trapezoid(beam, dx=dv)
    print(f'Maxwell-Boltzmann scaling speed: {vt} ms⁻¹')
    print('gas integral: \t' + str(int_mb))
    print('beam integral: \t' + str(int_mb))

    # peaks
    peak_mb = v[maxboltz.argmax()]
    peak_bm = v[beam.argmax()]
    # theoretical
    mb_t = vt
    bm_t = np.sqrt(3/2) * vt
    # rel error
    er_mb = np.abs((peak_mb - mb_t)/mb_t)
    er_bm = np.abs((peak_bm - bm_t)/bm_t)
    print(f'gas most probable v: \t{peak_mb:.5e} ms⁻¹\ttheoretical: {mb_t:5e} ms⁻¹\trel error: {er_mb}')
    print(f'beam most probable v: \t{peak_bm:.5e} ms⁻¹\ttheoretical: {bm_t:5e} ms⁻¹\trel error: {er_bm}')

    # mean
    mean_mb = np.average(v, weights=maxboltz)
    mean_bm = np.average(v, weights=beam)
    # theoretical
    mb_t = 2 * vt / np.sqrt(np.pi)
    bm_t = 0.75 * np.sqrt(np.pi) * vt
    # rel error
    er_mb = np.abs((mean_mb - mb_t)/mb_t)
    er_bm = np.abs((mean_bm - bm_t)/bm_t)
    print(f' gas mean v: \t{mean_mb:.5e} ms⁻¹\ttheoretical: {mb_t:5e} ms⁻¹\trel error: {er_mb}')
    print(f'beam mean v: \t{mean_bm:.5e} ms⁻¹\ttheoretical: {bm_t:5e} ms⁻¹\trel error: {er_bm}')

    # plots

    fig, ax = plt.subplots(1,2)
    ax[0].plot(v/vt, maxboltz, label='gas')
    ax[0].plot(v/vt, beam, label='beam')
    ax[0].set_title('Maxwell-Boltzmann')
    ax[0].legend()
    ax[0].set_xlabel('v/vp')
    ax[1].plot(om, population)
    ax[1].set_title('excited population')
    ax[1].set_xlabel('ω - ω₀')

    fig, ax = plt.subplots(1,1)
    ax.plot(omv-om0, voigt, label='Voigt')
    ax.plot(omv-om0, compare_pop, label='static atom')
    ax.set_xlabel('ω - ω₀')
    ax.legend()
    ax.set_title('Voigt profile')

    fig, ax = plt.subplots(1,1)
    for voigt, T in zip(compare_temps, temps):
        ax.plot(omv-om0, voigt, label=f'Voigt (T={T} K)')
    ax.set_xlabel('ω - ω₀')
    ax.legend()
    ax.set_title('Voigt profile for different temperatures')

    fig, ax = plt.subplots(1,1)
    for voigt, angle in zip(compare_angles, angles):
        ax.plot(omv-om0, voigt, label=f'Voigt ($\\theta$={np.rad2deg(angle)}°)')
    ax.set_xlabel('ω - ω₀')
    ax.legend()
    ax.set_title('Voigt profile for different atom beam-laser angles')

    plt.show()

def dist3d_comp():
    om0 = 2*np.pi*PEAK_FREQ
    k = om0/cst.speed_of_light
    T = 300
    #T = 1e-5
    angles = np.linspace(0, 3/4 * np.pi/2, num=3)
    sat = 0.1

    num = 501
    vt = ut.temp2vel(T)
    omt = vt * k
    om_max = omt * 3
    om = np.linspace(om0-om_max, om0+om_max, num=num)

    beams = [dist.voigt3d_beam(om, theta, vp=vt,
                                      om0=om0, sat=sat, tau=TAU) for theta in angles]
    gases = [dist.voigt3d_gas(om, theta, vp=vt,
                                      om0=om0, sat=sat, tau=TAU) for theta in angles]

    fig, ax = plt.subplots(1,1)
    for beam, angle in zip(beams, angles):
        ax.plot(om-om0, beam, label=f'Beam ($\\theta$={np.rad2deg(angle)}°)')
    for gas, angle in zip(gases, angles):
        ax.plot(om-om0, gas, label=f'Gas ($\\theta$={np.rad2deg(angle)}°)')
    ax.set_xlabel('ω - ω₀')
    ax.legend()
    ax.set_title('Voigt profile for different atom beam-laser angles')
    plt.show()

if __name__ == '__main__':
    dist1d()
    #dist3d(dist.voigt3d_gas)
    #dist3d_comp()

