from vib import solver as solver_unscaled

def solver_scaled(alpha, beta, gamma, delta, Q, T, dt):
    """
    Solve u'' + (1/Q)*u' + u = gamma*cos(delta*t),
    u(0)=alpha, u'(1)=beta, for (0,T] with step dt.
    """
    print 'Computing the numerical solution'
    from math import cos
    return solver_unscaled(I=alpha, V=beta, m=1, b=1./Q,
                           s=lambda u: u,
                           F=lambda t: gamma*cos(delta*t),
                           dt=dt, T=T, damping='linear')

import joblib
disk_memory = joblib.Memory(cachedir='temp')
solver_scaled = disk_memory.cache(solver_scaled)

import matplotlib.pyplot as plt
from math import pi

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--delta', type=float, default=0.5)
    parser.add_argument('--Q', type=float, default=10.0)
    parser.add_argument('--n', type=int, default=160)
    parser.add_argument('--P', type=int, default=40)
    parser.add_argument('--ylabel', type=str, default='u')
    parser.add_argument('--plot_F', action='store_true')

    a = parser.parse_args()

    u, t = solver_scaled(alpha=a.alpha, beta=a.beta,
                         gamma=a.gamma, delta=a.delta,
                         Q=a.Q, T=2*pi*a.P, dt=2*pi/a.n)
    plt.plot(t, u)
    if a.plot_F:
        from numpy import cos
        plt.plot(t, a.gamma*cos(a.delta*t), 'k--')
        plt.legend([a.ylabel, '$F(t)$'], loc='lower right')
    plt.xlabel('t');  plt.ylabel(a.ylabel)
    plt.title(r'$\alpha=%g,\ \beta=%g,\ \gamma=%g,\ \delta=%g,\ Q=%g,\ n=%d$' %
              (a.alpha, a.beta, a.gamma, a.delta, a.Q, a.n))

if __name__ == '__main__':
    main()
    plt.show()
    raw_input()
