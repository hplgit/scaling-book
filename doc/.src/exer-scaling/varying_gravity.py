import odespy, numpy as np
import matplotlib.pyplot as plt

def varying_gravity(epsilon):
    def ode_a(u, t):
        h, v = u
        return [v, -epsilon**(-2)/(1 + h)**2]

    def ode_b(u, t):
        h, v = u
        return [v, -1.0/(1 + h)**2]

    def ode_c(u, t):
        h, v = u
        return [v, -1.0/(1 + epsilon**2*h)**2]

    problems = [ode_a, ode_b, ode_c]     # right-hand sides
    ics = [[0, 1], [0, epsilon], [0, 1]] # initial conditions
    for problem, ic, legend in zip(problems, ics, ['a', 'b', 'c']):
        solver = odespy.RK4(problem)
        solver.set_initial_condition(ic)
        t = np.linspace(0, 5, 5001)
        # Solve ODE until h < 0 (h is in u[:,0])
        u, t = solver.solve(t, terminate=lambda u, t, n: u[n,0] < 0)
        h = u[:,0]

        plt.figure()
        plt.plot(t, h)
        plt.legend(legend, loc='upper left')
        plt.title(r'$\epsilon^2=%g$' % epsilon**2)
        plt.xlabel(r'$\bar t$');  plt.ylabel(r'$\bar h(\bar t)$')
        plt.savefig('tmp_%s.png' % legend)
        plt.savefig('tmp_%s.pdf' % legend)

if __name__ == '__main__':
    import sys, os
    # Give empsilon**2 (V**2/g/R) on the command-line
    epsilon = np.sqrt(float(sys.argv[1]))
    varying_gravity(epsilon)
    for ext in 'pdf', 'png':
        cmd = 'doconce combine_images -3 tmp_a.%s tmp_b.%s tmp_c.%s varying_gravity_%g.%s' % (ext, ext, ext, epsilon**2, ext)
        print cmd
        os.system(cmd)
    plt.show()
