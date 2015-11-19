import sys, os
# Enable loading modules in ../src-scaling
sys.path.insert(0, os.path.join(os.pardir, 'src-scaling'))
from decay_vc import solver as solver_unscaled
from math import pi
import matplotlib.pyplot as plt
import numpy as np

def solver_scaled(gamma, T, dt, theta=0.5):
    """
    Solve u'=-a*u, u(0)=1 for (0,T] with step dt and theta method.
    a=1 for t < gamma and 2 for t > gamma.
    """
    print 'Computing the numerical solution'
    return solver_unscaled(
        I=1, a=lambda t: 1 if t < gamma else 5,
        b=lambda t: 0, T=T, dt=dt, theta=theta)

import joblib
disk_memory = joblib.Memory(cachedir='temp')
solver_scaled = disk_memory.cache(solver_scaled)

def unscale(u_scaled, t_scaled, d, I):
    return I*u_scaled, d*t_scaled

def main(d,
         I,
         t_1,
         dt=0.04, # Time step, scaled problem
         T=4,     # Final time, scaled problem
         ):

    legends1 = []
    legends2 = []
    plt.figure(1)
    plt.figure(2)

    gamma = t_1*d
    print 'gamma=%.3f' % gamma
    u_scaled, t_scaled = solver_scaled(gamma, T, dt)

    plt.figure(1)
    plt.plot(t_scaled, u_scaled)
    legends1.append('gamma=%.3f' % gamma)

    plt.figure(2)
    u, t = unscale(u_scaled, t_scaled, d, I)
    plt.plot(t, u)
    legends2.append('d=%.2f [1/s], t_1=%.2f s' % (d, t_1))
    plt.figure(1)
    plt.xlabel('scaled time'); plt.ylabel('scaled velocity')
    plt.legend(legends1, loc='upper right')
    plt.savefig('tmp1.png');  plt.savefig('tmp1.pdf')

    plt.figure(2)
    plt.xlabel('t [s]');  plt.ylabel('u')
    plt.legend(legends2, loc='upper right')
    plt.savefig('tmp2.png');  plt.savefig('tmp2.pdf')
    plt.show()

if __name__ == '__main__':
    main(d=1/120., I=1, t_1=100)
