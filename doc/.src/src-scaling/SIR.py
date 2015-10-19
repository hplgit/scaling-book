import odespy
import numpy as np
import matplotlib.pyplot as plt
import sys

def solver(R0, alpha, T, dt=0.1):
    def f(u, t):
        S, I, R = u
        return [
            -R0*S*I,
            R0*S*I - I,
            I]

    Nt = int(round(T/dt))
    t_mesh = np.linspace(0, Nt*dt, Nt+1)

    solver = odespy.RK4(f)
    solver.set_initial_condition([1-alpha, alpha, 0])
    u, t = solver.solve(t_mesh)
    S = u[:,0]
    I = u[:,1]
    R = u[:,2]
    # Consistency check
    N = 1
    tol = 1E-15
    for i in range(len(S)):
        if abs(S[i] + I[i] + R[i] - N) > tol:
            print 'Consistency error: S+I+R=%g != %g' % \
                  (S[i] + I[i] + R[i], N)
    return S, I, R, t

def demo():
    alpha = 0.02
    R0 = 5
    T = 8
    dt = float(sys.argv[1]) if len(sys.argv) >= 2 else 0.1
    S, I, R, t = solver(R0, alpha, T, dt)
    plt.plot(t, S, t, I, t, R)
    plt.legend(['S', 'I', 'R'], loc='lower right')
    plt.title('R0=%g, alpha=%g' % (R0, alpha))
    plt.savefig('tmp.png');  plt.savefig('tmp.pdf')
    plt.show()

if __name__ == '__main__':
    demo()
