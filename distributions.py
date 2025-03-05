from scipy import constants as cst, integrate as intg
import matplotlib.pyplot as plt
import numpy as np

from constants import CA40_MASS, PEAK_FREQ, TAU
from utilities import get_alpha, get_alphainv, temp2vel, vel2temp, freq2om, om2freq

V_MAX = np.sqrt(2*cst.Boltzmann*1e3/CA40_MASS)
V_P = np.sqrt(2*cst.Boltzmann*(318+273)/CA40_MASS)

def maxwell_boltzmann_1d(x, a=1.0):
    return 1/(a * np.sqrt(np.pi)) * np.exp(-(x/a)**2)

def maxwell_boltzmann_3d(x, a=1.0):
    return 4 * x**2 /(a**3 * np.sqrt(np.pi)) * np.exp(-(x/a)**2) # × v²dv, from ramsey

def beam(x, a=1.0):
    return 2 * x**3 / a**4 * np.exp(-(x/a)**2) # × v³dv, from ramsey

def natural_linewidth(x, gam=1.0, P0=1.0):
    return P0 * gam/(2*np.pi) / (x**2 + (gam/2)**2)

def excited_population(x, sat=1.0, gam=1.0):
    return sat/2 / (1 + sat + (2*x/gam)**2)

def voigt1d(om, vp=V_P, om0=2*np.pi*PEAK_FREQ, sat=1.0, tau=TAU, num_om=2**8+1, om_max=None):
    om = om.reshape((om.size, 1))
    k = om0/cst.speed_of_light
    om_max = vp*k * 5
    gam = 1/tau

    # the integral is from 0 to (om_max-om0)
    # ompr is the integration variable
    # ompr_mb is centered around 0 so that the maxwell-boltzmann distribution is different from 0 in the integration domain
    ompr_mb = np.linspace(-om_max, om_max, num=num_om).reshape((1, num_om)) # (ω₀-ω') for integration
    ompr = om0 - ompr_mb # ω' for integration
    dom = np.abs(ompr[0,1] - ompr[0,0])
    integration_array = (1/k) * maxwell_boltzmann_1d(ompr_mb/k, vp) \
        * excited_population(om - ompr, sat, gam)
    integration_array = intg.trapezoid(integration_array, dx=dom, axis=-1)

    return integration_array

def voigt3d_common(vel_dist, om, theta, vp=V_P, om0=2*np.pi*PEAK_FREQ, sat=1.0, tau=TAU, num_om=2**8+1, om_max=None):
    om = np.reshape(om, (np.size(om), 1))
    theta = np.reshape(theta, (1, np.size(theta)))
    k = om0 * np.cos(theta) / cst.speed_of_light
    om_max = vp*k * 3
    gam = 1/tau

    # the integral is from om0 to (om_max+om0) (positive v)
    # ompr is the integration variable
    ompr = np.linspace(om0, om_max+om0, num=num_om).reshape((1, num_om)) # ω' for integration
    ompr_mb = ompr - om0 # (ω₀-ω') for integration on maxwell-boltzmann
    dom = np.abs(ompr[0,1] - ompr[0,0])
    integration_array = np.abs(1/k) * vel_dist(ompr_mb/k, vp) \
        * excited_population(om - ompr, sat, gam)
    integration_array = intg.trapezoid(integration_array, dx=dom, axis=-1)

    return integration_array

def voigt3d_gas(om, theta, vp=V_P, om0=2*np.pi*PEAK_FREQ, sat=1.0, tau=TAU, num_om=2**8+1, om_max=None):
    return voigt3d_common(maxwell_boltzmann_3d, om, theta, vp, om0, sat, tau, num_om, om_max)

def voigt3d_beam(om, theta, vp=V_P, om0=2*np.pi*PEAK_FREQ, sat=1.0, tau=TAU, num_om=2**8+1, om_max=None):
    return voigt3d_common(beam, om, theta, vp, om0, sat, tau, num_om, om_max)

def new_signal(om, theta, vp=V_P, om0=2*np.pi*PEAK_FREQ, sat=1.0, tau=TAU, theta_OL=0.0,
               num_om=2**8+1, om_max=None, num_theta=2**8+1, theta_min=-np.pi/2, theta_max=np.pi/2):
    """
    om: laser frequency (rad/s)
    theta: angle between laser wave-number and velocity vector (rad)
    vp: most probable speed (m/s)
    om0: transition frequency at rest (rad/s)
    sat: saturation parameter
    tau: transition lifetime
    theta_OL: angle between laser wave-number and oven most probable velocity (rad)
    num_om: number of integration points along omega
    om_max: number to use as infinity
    num_theta: number of integration points along theta
    theta_min, theta_max: integration limits for theta (as velocity coordinate)
    """
    # converting between velocity coordinate, and angle w.r.t. laser
    thetav2thetal = lambda th: theta_OL - th

    # constants
    k = om0 / cst.speed_of_light
    om_max = vp*k * 3 if not om_max else om_max
    gam = 1/tau

    # reshaped arrays
    # integration array will have shape (om.size, theta.size, num_om, num_theta)
    om = np.reshape(om, (np.size(om), 1, 1, 1))
    theta = np.reshape(theta, (1, np.size(theta), 1, 1))

    # integration variables
    ompr = np.linspace(om0, om_max+om0, num=num_om).reshape((1, 1, num_om, 1)) # ω'
    thetapr = np.linspace(theta_min, theta_max, num=num_theta).reshape((1, 1, 1, num_theta)) # θ'
    k_proj = k * np.cos(thetav2thetal(thetapr)) # projection of k along v
    dom = ompr[0,0,0,1] - ompr[0,0,0,0]
    dth = thetapr[0,0,1,0] - thetapr[0,0,0,0]

    integration_array = np.pi * np.sin(2*thetapr) / k_proj \
        * excited_population(om * np.cos(theta) - ompr, sat=sat, gam=gam) \
        * maxwell_boltzmann_3d((om0 - ompr)/(k * np.cos(thetapr)), a=vp)
    integration_array = intg.trapezoid(integration_array, dx=dom, axis=2) # integrate along ω'
    integration_array = intg.trapezoid(integration_array, dx=dth, axis=-1) # integrate along θ'
    return integration_array

def P_v(v, th, vp=V_P, theta_L=0.0, om0=2*np.pi*PEAK_FREQ,
        tau=TAU, num_theta=2**8+1, num_v=2**8+1, v_max=V_MAX,
        integration_method='romb', asym=True):
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
    v_min = 0 if asym else -v_max
    v = v.reshape((v.size,1,1,1)) 
    th = th.reshape((1,th.size,1,1))
    vpr = np.linspace(v_min, v_max, num=num_v).reshape((1,1,num_v,1)) # v prime
    thpr = np.linspace(-np.pi/2, np.pi/2, num=num_theta).reshape((1,1,1,num_theta)) # theta prime

    # do integration
    dth = thpr[0,0,0,1] - thpr[0,0,0,0]
    dv = vpr[0,0,1,0] - vpr[0,0,0,0]

    integrate = intg.romb
    if integration_method == 'trapezoid':
        integrate = intg.trapezoid
    elif integration_method == 'simpson':
        integrate = intg.simpson

    if asym:
        integration_array = vpr**2 * np.abs(np.sin(thpr)) * maxwell_boltzmann_3d(vpr, vp) * natural_linewidth(v*alphainv(th) - vpr*alphainv(thpr), gam)
        integration_array = integrate(integration_array, dx=dth, axis=-1) # along theta
    else:
        integration_array = vpr**2 * maxwell_boltzmann_3d(vpr, vp) * natural_linewidth(v*alphainv(th) - vpr*alphainv(th), gam)
        integration_array = integration_array.reshape((v.size, th.size, num_v))
    integration_array = integrate(integration_array, dx=dv, axis=-1) # along v
    return integration_array


def signal_v(v, th, vp=V_P, theta_L=0.0, om0=2*np.pi*PEAK_FREQ,
             sat=1.0, tau=TAU, num_theta=2**8+1, num_v=2**8+1, v_max=V_MAX,
             integration_method='romb', asym=True):

    alphainv = get_alphainv(th, om0, theta_L)
    P = P_v(v, th, vp, theta_L, om0, tau, num_theta,
            num_v, v_max, integration_method, asym)
    v = v.reshape((v.size, 1))
    N = excited_population(v*alphainv, sat=sat, gam=1/tau)

    return N*P

def signal_f(f, th, vp=V_P, theta_L=0.0, om0=2*np.pi*PEAK_FREQ,
             sat=1.0, tau=TAU, num_theta=2**8+1, num_v=2**8+1, v_max=V_MAX,
             integration_method='romb', asym=True):
    f = f.reshape((f.size, 1))
    th = th.reshape((1, th.size))
    alpha = get_alpha(th, om0, theta_L)
    v = alpha * (2*np.pi*f - om0)
    return signal_v(v, th, vp, theta_L, om0, sat, tau, num_theta, num_v, v_max, asym)

