import sys, os
# Enable loading modules in ../src-scaling
sys.path.insert(0, os.path.join(os.pardir, 'src-scaling'))
from decay_vc import solver as solver_unscaled
from math import pi
import matplotlib.pyplot as plt
import numpy as np

def solver_scaled(alpha, beta, t_stop, dt, theta=0.5):
    """
    Solve T' = -T + 1 + alha*sin(beta*t), T(0)=0
    for (0,T] with step dt and theta method.
    """
    print 'Computing the numerical solution'
    return solver_unscaled(
        I=0, a=lambda t: 1,
        b=lambda t: 1 + alpha*np.sin(beta*t),
        T=t_stop, dt=dt, theta=theta)

import joblib
disk_memory = joblib.Memory(cachedir='temp')
solver_scaled = disk_memory.cache(solver_scaled)

def main(alpha,
         beta,
         t_stop=50,
         dt=0.04
         ):

    T, t = solver_scaled(alpha, beta, t_stop, dt)
    plt.plot(t, T)
    plt.xlabel(r'$\bar t$');  plt.ylabel(r'$\bar T$')
    plt.title(r'$\alpha=%g,\ \beta=%g$' % (alpha, beta))
    filestem = 'tmp_%s_%s' % (alpha, beta)
    plt.savefig(filestem + '.png');  plt.savefig(filestem + '.pdf')
    plt.show()

if __name__ == '__main__':
    import sys
    alpha = float(sys.argv[1])
    beta = float(sys.argv[2])
    t_stop = float(sys.argv[3])
    main(alpha, beta, t_stop)
