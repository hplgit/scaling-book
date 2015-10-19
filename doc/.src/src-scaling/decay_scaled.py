from decay import solver_unscaled
import numpy as np
import matplotlib.pyplot as plt

def solver_scaled(T, dt, theta):
    """
    Solve u'=-u, u(0)=1 for (0,T] with step dt and theta method.
    """
    print 'Computing the numerical solution'
    return solver_unscaled(I=1, a=1, T=T, dt=dt, theta=theta)

import joblib
disk_memory = joblib.Memory(cachedir='temp')
solver_scaled = disk_memory.cache(solver_scaled)

from decay import unscale, read_command_line_argparse

def main():
    # Read parameters, solve and plot
    I, a, T, theta, dt_values = read_command_line_argparse()
    dt = dt_values[0]  # use only the first dt value
    u_scaled, t_scaled = solver_scaled(T, dt, theta)
    u, t = unscale(u_scaled, t_scaled, I, a)

    plt.figure()
    plt.plot(t_scaled, u_scaled)
    plt.xlabel('scaled time'); plt.ylabel('scaled velocity')
    plt.title('Universial solution of scaled problem')
    plt.savefig('tmp1.png');  plt.savefig('tmp1.pdf')

    plt.figure()
    plt.plot(t, u)
    plt.xlabel('t'); plt.ylabel('u')
    plt.title('I=%g, a=%g, theta=%g' % (I, a, theta))
    plt.savefig('tmp2.png'); plt.savefig('tmp2.pdf')
    plt.show()

if __name__ == '__main__':
    main()
