import scitools.std as plt
import matplotlib.pyplot as plt
import numpy as np

def simulate(Theta, num_periods=8, time_steps_per_period=60,
             scaling=1):
    # Use oscillations for small Theta to set dt and T
    P = 2*np.pi
    dt = P/time_steps_per_period
    T = num_periods*P
    t = np.linspace(0, T, time_steps_per_period*num_periods+1)
    import odespy

    def f(u, t, Theta):
        # Note the sequence of unknowns: omega, theta
        # omega = d(theta)/dt, angular velocity
        omega, theta = u
        return [-Theta**(-1)*np.sin(Theta*theta), omega]

    solver = odespy.RK4(f, f_args=[Theta])
    solver.set_initial_condition([0, 1]) # sequence must match f
    u, t = solver.solve(t)
    theta = u[:,1]  # recall sequence in f: omega, theta
    return theta, t


if __name__ == '__main__':
    Theta_values_degrees = [1, 20, 45, 60]
    for Theta_degrees in Theta_values_degrees:
        Theta = Theta_degrees*np.pi/180
        theta, t = simulate(Theta, 6, 60)
        plt.plot(t, theta)
        plt.hold('on')
    plt.legend([r'$\Theta=%g$' % Theta
                for Theta in Theta_values_degrees],
               loc='lower left')
    plt.xlabel(r'$\bar t$');  plt.ylabel(r'$\bar\theta$')
    plt.savefig('tmp.png');   plt.savefig('tmp.pdf')
    plt.show()
    raw_input()
