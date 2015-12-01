#import scitools.std as plt
import matplotlib.pyplot as plt
import odespy
import numpy as np

def solver(alpha, ic, T, dt=0.05):
    def f(u, t):
        x, vx, y, vy = u
        v = np.sqrt(vx**2 + vy**2)  # magnitude of velocity
        system = [
            vx,
            -alpha*np.abs(v)*vx,
            vy,
            -2 - alpha*np.abs(v)*vy,
            ]
        return system

    Nt = int(round(T/dt))
    t_mesh = np.linspace(0, Nt*dt, Nt+1)

    solver = odespy.RK4(f)
    solver.set_initial_condition(ic)
    u, t = solver.solve(t_mesh,
                        terminate=lambda u, t, n: u[n][2] < 0)
    x = u[:,0]
    y= u[:,2]
    return x, y, t

def demo_soccer_ball():
    import math
    theta_degrees = 45
    theta = math.radians(theta_degrees)
    ic = [0, 2/math.tan(theta), 0, 2]
    g = 9.81
    v0_s = 8.3    # soft kick
    v0_h = 33.3   # hard kick
    # Length scales
    L_s = 0.5*(v0_s**2/g)*math.sin(theta)**2
    L_h = 0.5*(v0_h**2/g)*math.sin(theta)**2
    print 'L:', L_s, L_h

    m = 0.43
    a = 0.11
    A = math.pi*a**2
    rho = 1.2
    C_D = 0.4
    alpha_s = C_D*rho*A*v0_s**2*math.cos(theta)**2/(4*m*g)
    alpha_h = C_D*rho*A*v0_h**2*math.cos(theta)**2/(4*m*g)
    print 'alpha:', alpha_s, alpha_h
    x_s, y_s, t = solver(alpha=alpha_s, ic=ic, T=6, dt=0.01)
    x_h, y_h, t = solver(alpha=alpha_h, ic=ic, T=6, dt=0.01)
    plt.plot(x_s, y_s, x_h, y_h)
    plt.legend(['soft, L=%.2f' % L_s, 'hard, L=%.2f' % L_h],
                loc='upper left')
    # Let the y range be [-0.2,2] so we have space for legends
    plt.axis([x_s[0], x_s[-1], -0.2, 2])
    plt.axes().set_aspect('equal') # x and y axis have same scaling
    plt.title(r'$\theta=%d$ degrees' % theta_degrees)
    plt.savefig('tmp.png')
    plt.savefig('tmp.pdf')
    plt.show()

demo_soccer_ball()
