import sys, os
from decay_vc import solver as solver_unscaled

def solver_scaled(beta, T, dt, theta=0.5):
    """
    Solve u'=-u+1, u(0)=beta for (0,T]
    with step dt and theta method.
    """
    print 'Computing the numerical solution'
    return solver_unscaled(
        I=beta, a=lambda t: 1, b=lambda t: 1,
        T=T, dt=dt, theta=theta)

import joblib
disk_memory = joblib.Memory(cachedir='temp')
solver_scaled = disk_memory.cache(solver_scaled)

def unscale(u_scaled, t_scaled, d, mu, rho, rho_b, V):
    a, b = ab(d, mu, rho, rho_b, V)
    return (b/a)*u_scaled, a*t_scaled

def ab(d, mu, rho, rho_b, V):
    g = 9.81
    a = 3*pi*d*mu/(rho_b*V)
    b = g*(rho/rho_b - 1)
    return a, b

from math import pi
import matplotlib.pyplot as plt
import numpy as np

def main(dt=0.075, # Time step, scaled problem
         T=7.5,    # Final time, scaled problem
         d=0.001,  # Diameter (unscaled problem)
         I=0,      # Initial velocity (unscaled problem)
         ):
    # Set parameters, solve and plot
    rho = 0.00129E+3  # air
    rho_b = 1E+3      # density of water
    mu = 0.001        # viscosity of water
    # Asumme we have list or similar for d
    if not isinstance(d, (list,tuple,np.ndarray)):
        d = [d]

    legends1 = []
    legends2 = []
    plt.figure(1)
    plt.figure(2)
    betas = []     # beta values already computed (for plot)

    for d_ in d:
        V = 4*pi/3*(d_/2.)**3  # volume
        a, b = ab(d_, mu, rho, rho_b, V)
        beta = I*a/b
        # Restrict to 3 digits in beta
        beta = abs(round(beta, 3))

        print 'beta=%.3f' % beta
        u_scaled, t_scaled = solver_scaled(beta, T, dt)

        # Avoid plotting curves with the same beta value
        if not beta in betas:
            plt.figure(1)
            plt.plot(t_scaled, u_scaled)
            plt.hold('on')
            legends1.append('beta=%g' % beta)
        betas.append(beta)

        plt.figure(2)
        u, t = unscale(u_scaled, t_scaled, d_, mu, rho, rho_b, V)
        plt.plot(t, u)
        plt.hold('on')
        legends2.append('d=%g [mm]' % (d_*1000))
    plt.figure(1)
    plt.xlabel('scaled time'); plt.ylabel('scaled velocity')
    plt.legend(legends1, loc='lower right')
    plt.savefig('tmp1.png');  plt.savefig('tmp1.pdf')

    plt.figure(2)
    plt.xlabel('t [s]');  plt.ylabel('u [m/s]')
    plt.legend(legends2, loc='lower right')
    plt.savefig('tmp2.png');  plt.savefig('tmp2.pdf')
    plt.show()

if __name__ == '__main__':
    main(d=[0.001, 0.002, 0.003])
