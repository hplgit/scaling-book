import numpy as np
import matplotlib.pyplot as plt

def u(t, A, w, module=np):
    return A*module.sin(w*t)

def a():
    """Plot u."""
    w = 2*np.pi
    A = 1.0
    t = np.linspace(0, 8*np.pi/w, 1001)
    plt.figure()
    plt.plot(t, u(t, A, w))
    plt.xlabel('t');  plt.ylabel('u')
    plt.axis([t[0], t[-1], -1.1, 1.1])
    plt.title(r'$u=A\sin (\omega t)$, $A=%g$, $\omega = %g$'
              % (A, w))
    plt.savefig('tmp1.png');  plt.savefig('tmp1.pdf')

def u_damped(t, A, w, a, module=np):
    return A*module.exp(-a*t)*module.sin(w*t)

def c():
    """Plot damped u."""
    w = 2*np.pi
    A = 1.0
    a = 0.5
    t = np.linspace(0, 8*np.pi/w, 100001)
    plt.figure()
    plt.plot(t, u_damped(t, A, w, a))
    plt.xlabel('t');  plt.ylabel('u')
    plt.title(r'$u=Ae^{-at}\sin (\omega t)$,'
              '$a=%g$, $A=%g$, $\omega = %g$' % (a, A, w))
    plt.savefig('tmp2.png');  plt.savefig('tmp2.pdf')

    u_max = []
    u_ = u_damped(t, A, w, a)
    for i in range(1, len(t)-1):
        if u_[i-1] < u_[i] > u_[i+1]:
            u_max.append((t[i], u_[i]))
    print u_max
    for i in range(len(u_max)-1):
        print 'P=', u_max[i+1][0] - u_max[i][0]

def d():
    """Find the period of u."""
    # Method 1: the sine function has period 2*pi
    import sympy as sym
    t, w, P, A, a = sym.symbols('t w P A a')
    # w*t + 2*pi = w*(t+P)
    dudt = sym.diff(u_damped(t, A, w, a, module=sym), t)
    s = sym.solve(dudt, t)
    print 'du/dt=0:', s
    print 'Inserted in u:'
    for s_ in s:
        print 'root:', s_, 'u:', \
              sym.simplify(u_damped(s_, A, w, a, module=sym))

#a()
c()
#d()
#plt.show()
