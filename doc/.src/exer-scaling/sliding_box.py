import scitools.std as plt
import matplotlib.pyplot as plt
import numpy as np

def simulate(alpha, beta=0,
             num_periods=8, time_steps_per_period=60):
    # Use oscillations without friction to set dt and T
    P = 2*np.pi
    dt = P/time_steps_per_period
    T = num_periods*P
    t = np.linspace(0, T, time_steps_per_period*num_periods+1)
    import odespy

    def f(u, t, alpha):
        # Note the sequence of unknowns: v, u (v=du/dt)
        v, u = u
        return [-alpha*np.sign(v) - u, v]

    solver = odespy.RK4(f, f_args=[alpha])
    solver.set_initial_condition([beta, 1])  # sequence must match f
    uv, t = solver.solve(t)
    u = uv[:,1]  # recall sequence in f: v, u
    v = uv[:,0]
    return u, t


if __name__ == '__main__':
    alpha_values = [0, 0.05, 0.1]
    for alpha in alpha_values:
        u, t = simulate(alpha, 0, 6, 60)
        plt.plot(t, u)
        plt.hold('on')
    plt.legend([r'$\alpha=%g$' % alpha for alpha in alpha_values])
    plt.xlabel(r'$\bar t$');  plt.ylabel(r'$\bar u$')
    plt.savefig('tmp.png');   plt.savefig('tmp.pdf')
    plt.show()
    raw_input()  # for scitools' matplotlib engine
