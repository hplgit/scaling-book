import numpy as np

def cooling(T0, k, T_s, t_end, dt, theta=0.5):
    """
    Solve T'=-k(T-T_s(t)), T(0)=T0,
    for t in (0,t_end] with steps of dt.
    T_s(t) is a Python function of t.
    theta=0.5 means Crank-Nicolson, 1 is Backward
    Euler, and 0 is Forward Euler scheme.
    """
    dt = float(dt)                  # avoid integer division
    Nt = int(round(t_end/dt))       # no of time intervals
    t_end = Nt*dt                   # adjust to fit time step dt
    T = np.zeros(Nt+1)              # array of T[n] values
    t = np.linspace(0, t_end, Nt+1) # time mesh
    T[0] = T0                       # set initial condition
    for n in range(0, Nt):          # n=0,1,...,Nt-1
        T[n+1] = ((1 - dt*(1 - theta)*k)*T[n] + \
        dt*k*(theta*T_s(t[n+1]) + (1 - theta)*T_s(t[n])))/ \
        (1 + dt*theta*k)
    return T, t

def cooling_scaled(alpha, beta, t_end, dt, theta=0.5):
    return cooling(
        T0=0, k=1, T_s=T_s, t_end=t_end, dt=dt, theta=theta)

from numpy import pi, sin

def T_s(t):
    return 1 + alpha*sin(beta*t)

def demo():
    Tm = 25
    a = 2.5
    P_values = [3600, 600, 3600*6]  # P: period of unscaled T_s
    k = 0.05/60
    T0 = 5
    T0 = 24.9

    global alpha, beta   # needed in T_s(t)
    alpha = a/(Tm - T0)

    import matplotlib.pyplot as plt
    legends = []
    for P in P_values:
        omega = 2*pi/P
        beta = omega/k
        dt = min(P_values)/40*k  # 40 steps per shortest period
        t_end = 1.6*max(P_values)*k
        T, t = cooling_scaled(alpha, beta, t_end, dt, theta=0.5)
        plt.plot(t, T)
        legends.append(r'$\beta = %.2f$' % beta)
    plt.plot(t, T_s(t), 'k--')  # T_s for largest P to show amplitude
    legends.append(r'$T_s$, $\beta = %.2f$' % beta)
    plt.legend(legends, loc='lower right')
    plt.xlabel('t'); plt.ylabel('T')
    plt.savefig('tmp.png');  plt.savefig('tmp.pdf')
    plt.show()

if __name__ == '__main__':
    demo()
