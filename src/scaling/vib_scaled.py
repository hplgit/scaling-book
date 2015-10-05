import sys, os
sys.path.insert(0, os.path.join(
    os.pardir, os.pardir, 'vib', 'src-vib'))
from vib import solver as solver_unscaled

def solver_scaled(alpha, beta, gamma, delta, T, dt):
    """
    Solve u'' + b*u + u = gamma*cos(delta*t),
    u(0)=alpha, u'(1)=beta, for (0,T] with step dt.
    """
    print 'Computing the numerical solution'
    return solver_unscaled(I=1, a=1, T=T, dt=dt, theta=theta)

import joblib
disk_memory = joblib.Memory(cachedir='temp')
solver_scaled = disk_memory.cache(solver_scaled)
