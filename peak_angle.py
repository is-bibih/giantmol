import matplotlib.pyplot as plt
import numpy as np

from fit_single_peak import CA40_MASS, PEAK_FREQ, TAU
from fit_single_peak import read_preprocess
from oven_dist import temp2vel, signal_f, signal_v

if __name__ == '__main__':
    savedir = 'arrays/'
    read = True

    th = np.array([0])
    vp = temp2vel(318+273) # ~500 ms⁻¹
    fp = PEAK_FREQ

    if read:
        angles = np.load(savedir + 'angles.npy')
        v = np.load(savedir + 'v.npy')
        max_vs = np.load(savedir + 'max_vs.npy')
        Ps = [[0]] * len(angles)
        for i in range(len(angles)):
            Ps[i] = np.load(f'{savedir}P_{i}.npy')
    else:
        angles = np.linspace(0, 2*np.pi, num=300)
        v = np.linspace(-vp/5, vp/5, num=251)
        Ps = [signal_v(v, th, vp=vp, theta_L=ang, integration_method='trapezoid') for ang in angles]
        Ps = [P / P.max() for P in Ps]
        max_vs = np.array([v[np.argmax(P)] for P in Ps]).flatten()

        freq_delta = 2e-7
        f = np.linspace(fp*(1-freq_delta), fp*(1+freq_delta), num=100)
        Pfs = [signal_f(f, th, vp=vp, theta_L=ang, integration_method='trapezoid') for ang in angles]

        np.save(savedir + 'angles.npy', angles)
        np.save(savedir + 'v.npy', v)
        np.save(savedir + 'max_vs.npy', max_vs)
        for i in range(len(Ps)):
            np.save(f'{savedir}P_{i}.npy', Ps[i])

    # fit for angle-vp
    fit_f = lambda th, n=1: -np.abs(np.sin(th)**n / np.cos(th))

    # plots for multiple angles
    num = 5
    colors = plt.cm.jet(np.linspace(0, 1, num=num))
    angles_compare = np.pi/2 * np.linspace(0.0, 0.97, num=num)
    idx = np.array([np.argmin(np.abs(angles - th)) for th in angles_compare]).astype(int)
    angles_compare = angles[idx]
    max_vs_compare = max_vs[idx]
    P_compare = [Ps[i] for i in idx]

    fig, ax = plt.subplots(1,1)
    ax.scatter(np.rad2deg(angles), max_vs, label='numerical fit value')
    ax.plot(np.rad2deg(angles), fit_f(angles, n=1), c='red', label='-|tanθ|')
    ax.plot(np.rad2deg(angles), fit_f(angles, n=10), c='green', label='-|sin¹⁰θ / cosθ|')
    ax.set_ylim(-50, 10)
    ax.set_ylabel('speed for signal peak v / ms⁻¹')
    ax.set_xlabel(r'laser-atom beam angle θ / °')
    ax.legend()

    fig, ax = plt.subplots(1,1)
    for i in range(num):
        ax.plot(v, P_compare[i], color=colors[i], label=f'{np.rad2deg(angles_compare[i])}°')
        ax.scatter(fit_f(angles_compare[i], n=10), 1, marker='*', color=colors[i])
    fig.legend()
    plt.show()
