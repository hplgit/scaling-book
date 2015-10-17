import odespy
import numpy as np
import matplotlib.pyplot as plt
import sys

def solver(R0, alpha, gamma, delta, T, dt=0.1):
    def f(u, t):
        S, I, R, V = u
        return [
            -R0*S*I - delta(t)*S + gamma*R,
            R0*S*I - I,
            I - gamma*R,
            delta(t)*S]

    Nt = int(round(T/dt))
    t_mesh = np.linspace(0, Nt*dt, Nt+1)

    solver = odespy.RK4(f)
    solver.set_initial_condition([1-alpha, alpha, 0, 0])
    u, t = solver.solve(t_mesh)
    S = u[:,0]
    I = u[:,1]
    R = u[:,2]
    V = u[:,3]
    # Consistency check
    N = 1
    tol = 1E-13
    for i in range(len(S)):
        if abs(S[i] + I[i] + R[i] + V[i] - N) > tol:
            print 'Consistency error: S+I+R+V=%g != %g' % \
                  (S[i] + I[i] + R[i] + V[i], N)
    return S, I, R, V, t

def demo():
    alpha = 0.02
    R0 = 5
    gamma = 0.05
    delta = 0.5
    T = 60
    dt = float(sys.argv[1]) if len(sys.argv) >= 2 else 0.01
    S, I, R, V, t = solver(R0, alpha, gamma, lambda t: delta if 7 <= t <= 15 else 0, T, dt)
    plt.plot(t, S, t, I, t, R, t, V)
    plt.legend(['S', 'I', 'R', 'V'], loc='upper right')
    plt.title(r'$R_0=%g$, $\alpha=%g$, $\gamma=%g$, $\delta=%g$' %
              (R0, alpha, gamma, delta))
    plt.savefig('tmp.png');  plt.savefig('tmp.pdf')
    plt.show()

if __name__ == '__main__':
    demo()
