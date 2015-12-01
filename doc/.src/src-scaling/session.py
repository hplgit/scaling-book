from sympy import *

# Solutions of various differential equations

def generalized_exponential_decay():
    print "u' -a*u + b"
    t, a, b, I = symbols('t a b I', real=True, positive=True)
    u = symbols('u', cls=Function)
    eq = diff(u(t), t) + a*u(t) - b
    # or
    eq = Eq(diff(u(t), t), -a*u(t) + b)
    sol = dsolve(eq, u(t))
    print sol
    u = sol.rhs
    C1 = symbols('C1')
    eq = Eq(u.subs(t, 0), I)
    sol = solve(eq, C1)
    print sol
    u = u.subs(C1, sol[0])
    print u
    u = simplify(expand(u))
    print u
    return u

def cooling_sine_Ts_hand_calc():
    print '--- Cooling with sinusoidal variations in the surroundings ---'
    t, k, T_m, a, w = symbols('t k T_m a w', real=True, positive=True)
    T_s = T_m + a*sin(w*t)
    I = exp(k*t)*T_s
    I = integrate(I, (t, 0, t))
    print I
    Q = k*exp(-k*t)*I
    Q = simplify(expand(Q))
    print Q
    return Q

def cooling_sine_Ts_dsolve():
    print '--- Cooling with sinusoidal variations in the surroundings ---'
    t, k, T_0, T_m, a, w = symbols('t k T_0 T_m a w',
                                   real=True, positive=True)
    T = symbols('T', cls=Function)
    T_s = T_m + a*sin(w*t)
    eq = Eq(diff(T(t), t), -k*(T(t) - T_s))
    sol = dsolve(eq, T(t))
    T = sol.rhs
    C1 = symbols('C1')
    eq = Eq(T.subs(t, 0), T_0)
    sol = solve(eq, C1)
    print sol
    T = T.subs(C1, sol[0])
    # Demonstrate here what T is by default and how it can be simplified
    T = simplify(expand(T))  # Yes, this is the way to rewrite!
    print T
    return T

def free_vibrations():
    print '--- Free vibrations ---'
    u = symbols('u', cls=Function)
    w, t = symbols('w t', real=True, positive=True)
    I, V, C1, C2 = symbols('I V C1 C2', real=True)
    #diffeq = Eq(u(t).diff(t,t) + w**2*u(t), 0)
    def ode(u):
        return diff(u, t, t) + w**2*u

    diffeq = ode(u(t))
    s = dsolve(diffeq, u(t))
    u_sol = s.rhs
    print u_sol
    #A, B = symbols('A B')
    # The solution contains C1 and C2 but these are not symbols,
    # substitute them by symbols
    u_sol = u_sol.subs('C1', C1).subs('C2', C2)
    print u_sol
    eqs = [u_sol.subs(t, 0) - I, u_sol.diff(t).subs(t, 0) - V]
    print eqs
    s = solve(eqs, [C1, C2])
    print s
    u_sol = u_sol.subs(C1, s[C1]).subs(C2, s[C2])
    print u_sol
    # Check the solution
    checks = dict(ODE=simplify(ode(u_sol)),
                  IC1=simplify(u_sol.subs(t, 0) - I),
                  IC2=simplify(diff(u_sol, t).subs(t, 0) - V))
    for check in checks:
        msg = '%s residual: %s' % (check, checks[check])
        assert checks[check] == 0, msg
    return u_sol

def forced_vibrations():
    print "--- Forced vibrations ---"
    u = symbols('u', cls=Function)
    t, w, A, A1, m, psi = symbols('t w A A1 m psi',
                                  positive=True, real=True)
    C1, C2, V, I = symbols('C1 C2 V I', real=True)
    #diffeq = Eq(u(t).diff(t,t) + w**2*u(t), A/m*sin(psi*t))
    # Note: SymPy often gets confused if we have multiple symbols
    # in coefficients. Use just one symbol in the coefficients.
    # A1 = A/m.

    def ode(u):
        return diff(u, t, t) + w**2*u - A1*cos(psi*t)

    diffeq = ode(u(t))
    u_sol = dsolve(diffeq, u(t))
    u_sol = u_sol.rhs
    u_sol = u_sol.subs('C1', C1).subs('C2', C2)
    eqs = [u_sol.subs(t, 0) - I, u_sol.diff(t).subs(t, 0) - V]
    s = solve(eqs, [C1, C2])
    print s
    u_sol = u_sol.subs(C1, s[C1]).subs(C2, s[C2])
    print u_sol
    # Check the solution
    checks = dict(ODE=simplify(ode(u_sol)),
                  IC1=simplify(u_sol.subs(t, 0) - I),
                  IC2=simplify(diff(u_sol, t).subs(t, 0) - V))
    for check in checks:
        msg = '%s residual: %s' % (check, checks[check])
        assert checks[check] == 0, msg

    # Rewrite for I=V=0 to check that special formula
    u_sol2 = A1/(w**2 - psi**2)*(cos(psi*t) - cos(w*t))
    print simplify(ode(u_sol2))
    print simplify(u_sol2.subs(t, 0))
    print simplify(diff(u_sol2, t).subs(t, 0))

    u_sol = simplify(expand(u_sol.subs(A1, A/m)))
    print 'u=', u_sol
    print latex(u_sol).replace('w', r'\omega')
    return u_sol

def damped_forced_vibrations_v1():
    # Demonstrate how dsolve solves this problem
    u = symbols('u', cls=Function)
    t, w, b, A, A1, m, psi = symbols('t w b A A1 m psi',
                                     positive=True, real=True)
    diffeq = diff(u(t), t, t) + b/m*diff(u(t), t) + w**2*u(t)
    s = dsolve(diffeq, u(t))
    print s.rhs
    # Problem: general complex exponential solution, not
    # expressed by sin/cos

def damped_forced_vibrations():
    print "--- Damped forced vibrations ---"
    u = symbols('u', cls=Function)
    t, w, B, A, A1, m, psi = symbols('t w B A A1 m psi',
                                     positive=True, real=True)
    # dsolve does not work well for this more complicated case
    # Run manual procedure

    def ode(u, homogeneous=True):
        #h = diff(u, t, t) + b/m*diff(u, t) + w**2*u
        #f = A/m*cos(psi*t)
        # Use just one symbol in each coefficient
        h = diff(u, t, t) + 2*B*diff(u, t) + w**2*u
        f = A1*cos(psi*t)
        return h if homogeneous else h - f

    # Find coefficients in polynomial (in r) for exp(r*t) ansatz
    r = symbols('r')
    ansatz = exp(r*t)
    poly = simplify(ode(ansatz)/ansatz)
    # Convert to polynomial to extract coefficients
    poly = Poly(poly, r)
    # Extract coefficients in poly: a_*t**2 + b_*t + c_
    a_, b_, c_ = poly.coeffs()
    print 'a:', a_, 'b:', b_, 'c:', c_
    # Assume b**2 - 4*a*c < 0
    d = -b_/(2*a_)
    print 'a_ == 1', a_ == 1, a_ == sympify(1)
    if a_ == 1:
        omega = sqrt(c_ - (b_/2)**2)  # nicer formula
    else:
        omega = sqrt(4*a_*c_ - b_**2)/(2*a_)
    print 'omega:', omega
    # The homogeneous solution is a linear combination of a
    # cos term (u1) and a sin term (u2)
    u1 = exp(d*t)*cos(omega*t)
    u2 = exp(d*t)*sin(omega*t)
    C1, C2, V, I = symbols('C1 C2 V I', real=True)
    u_h = simplify(C1*u1 + C2*u2)
    print 'u_h:', u_h
    # Check that the constructed h_h fits the ODE
    assert simplify(ode(u_h)) == 0

    # Particular solution
    C3, C4 = symbols('C3 C4')
    u_p = C3*cos(psi*t) + C4*sin(psi*t)
    eqs = simplify(ode(u_p, homogeneous=False))
    # Collect cos(omega*t) terms
    print 'eqs:', eqs
    eq_cos = simplify(eqs.subs(sin(psi*t), 0).subs(cos(psi*t), 1))
    eq_sin = simplify(eqs.subs(cos(psi*t), 0).subs(sin(psi*t), 1))
    s = solve([eq_cos, eq_sin], [C3, C4])
    u_p = simplify(u_p.subs(C3, s[C3]).subs(C4, s[C4]))
    # Check that u_p fulfills the ode
    assert simplify(ode(u_p, homogeneous=False)) == 0

    # Total solution
    u_sol = u_h + u_p
    print u_sol
    # Initial conditions
    eqs = [u_sol.subs(t, 0) - I, u_sol.diff(t).subs(t, 0) - V]
    print 'IC:', eqs
    # Determine C1 and C2 from the initial conditions
    s = solve(eqs, [C1, C2])
    print 'solution C1, C2:', s
    u_sol = u_sol.subs(C1, s[C1]).subs(C2, s[C2])
    print 'ode(u_h) with C1, C2 2:', simplify(ode(u_h))
    u_h = u_h.subs(C1, s[C1]).subs(C2, s[C2])
    print 'Final u_sol:', u_sol
    print 'ode(u_h):', simplify(ode(u_h))
    # Check the solution
    checks = dict(
        ODE=simplify(expand(ode(u_sol, homogeneous=False))),
        IC1=simplify(u_sol.subs(t, 0) - I),
        IC2=simplify(diff(u_sol, t).subs(t, 0) - V))
    for check in checks:
        msg = '%s residual: %s' % (check, checks[check])
        assert checks[check] == sympify(0), msg

    # Remember that A1 should be A/m
    u_sol = simplify(expand(u_sol.subs(A1, A/m)))
    print 'u=', latex(u_sol).replace('w', r'\omega')
    return u_sol

    """
    # Too complicated eqs for solve...but we know it is a linear system
    #s = solve(eqs, [C1, C2])
    eqs_coeff = [collect(e, [C1, C2], evaluate=False) for e in eqs]
    # - I + d*(-psi**2 + w**2)/(4*A**2*psi**2 + (psi**2 - w**2)**2)
    print 'eqs_coeff:', eqs_coeff
    # Note: eqs_coeff[i] is a dict with 1, C1, C2 as keys, but 1 is not
    # ordinary 1, but instead sympy.core.numbers.One.
    from sympy.core.numbers import One

    # Linear system with C1 and C2
    # Define simple symbols as coefficients and right-hand side
    # (subst these symbols by long expressions afterwards)
    from sympy.core.symbol import Symbol
    import numpy as np
    c = np.empty((2,2), dtype=Symbol)
    rhs = np.empty(2, dtype=Symbol)
    for i in range(2):
        for j in range(2):
            c[i,j] = Symbol('c_%d%d' % (i, j))
        rhs[i] = Symbol('rhs_%d' % i)
    eqs = [rhs[0] + c[0,0]*C1 + c[0,1]*C2,
           rhs[1] + c[1,0]*C1 + c[1,1]*C2]
    print 'eqs with symbols:', eqs
    s = solve(eqs, [C1, C2])
    print 's:', s
    print
    # Substitute original coefficints for c[i,j] and rhs[i]
    h = [C1, C2]  # help list for indexing
    for key in s:
        for i in range(2):
            for j in range(2):
                s[key] = s[key].subs(c[i,j], eqs_coeff[i].get(h[j], 0))
            s[key] = s[key].subs(rhs[i], eqs_coeff[i].get(One(), 0))
            s[key] = simplify(s[key])
    print 's after subst:', s
    """

def plot_forced_vibrations():
    import numpy as np
    t = np.linspace(0, 10*np.pi, 1001)
    w = 5
    psi = 1
    u = 1./(w**2 - psi**2)*np.sin((w+psi)/2.*t)*np.sin((psi-w)/2.*t)
    u2 = 1./(w**2 - psi**2)*np.sin(w*t)
    import matplotlib.pyplot as plt
    plt.plot(t, u)
    plt.plot(t, u2)
    plt.xlabel('t'); plt.ylabel('u')
    plt.savefig('tmp.png'); plt.savefig('tmp.pdf')
    plt.show()

def simulate_forced_vibrations1():
    # Scaling with u_c based on resonance amplitude
    from vib import solver, visualize
    from math import pi
    delta = 0.99
    alpha = 1 - delta**2
    u, t = solver(I=alpha, V=0, m=1, b=0, s=lambda u: u,
                  F=lambda t: (1-delta**2)*cos(delta*t),
                  dt=2*pi/160, T=2*pi*160)
    visualize(u, t)
    raw_input()

def simulate_forced_vibrations2():
    # Scaling with u_c based on gamma=1
    from vib import solver, visualize
    from math import pi
    delta = 0.5
    delta = 0.99
    alpha = 1
    u, t = solver(I=alpha, V=0, m=1, b=0, s=lambda u: u,
                  F=lambda t: cos(delta*t),
                  dt=2*pi/160, T=2*pi*160)
    visualize(u, t)
    raw_input()

def simulate_forced_vibrations3():
    # Scaling with u_c based on large delta
    from vib import solver, visualize
    from math import pi
    delta = 10
    alpha = 0.05*delta**2
    u, t = solver(I=alpha, V=0, m=1, b=0, s=lambda u: delta**(-2)*u,
                  F=lambda t: cos(t),
                  dt=2*pi/160, T=2*pi*20)
    visualize(u, t)
    raw_input()

def simulate_Gaussian_and_incoming_wave():
    from wave1D_dn import solver, viz
    from math import pi, sin
    from numpy import exp
    alpha = 0.1
    beta = 10
    gamma = 2*pi*3

    def I(x):
        return alpha*exp(-beta**2*(x - 0.5)**2)

    def U_0(t):
        return sin(gamma*t) if t <= 2*pi/gamma else 0

    L = 1
    c = 1
    Nx = 80; dx = L/float(Nx); dt = dx/c
    #solver(I=I, V=0, f=0, U_0=U_0, U_L=None, L=L, dt=dt, C=1, T=4,
    #       user_action=myplotter)
    viz(I=I, V=0, f=0, c=c, U_0=U_0, U_L=None, L=L, dt=dt, C=1,
        T=4, umin=-(alpha+1), umax=(alpha+1),
        version='vectorized', animate=True)

import odespy

def biochemical_solver(alpha, beta, epsilon, T, dt=0.1):
    def f(u, t):
        Q, P, S, E = u
        # Consistency checks
        conservation1 = abs(Q/(alpha*epsilon) + E - 1)
        conservation2 = abs(alpha*S + Q + P - alpha)
        tol = 1E-14
        if conservation1 > tol or conservation2 > tol:
            print 't=%g *** conservations:' % t, \
                  conservation1, conservation2
        if Q < 0:
            print 't=%g *** Q=%g < 0' % (t, Q)
        if P < 0:
            print 't=%g *** P=%g < 0' % (t, P)
        if S < 0 or S > 1:
            print 't=%g *** S=%g' % (t, S)
        if E < 0 or S > 1:
            print 't=%g *** E=%g' % (t, E)

        return [
            alpha*(E*S - Q),
            beta*Q,
            -E*S + (1-beta/alpha)*Q,
            (-E*S + Q)/epsilon,
            ]

    import numpy as np
    Nt = int(round(T/dt))
    t_mesh = np.linspace(0, Nt*dt, Nt+1)

    solver = odespy.RK4(f)
    solver.set_initial_condition([0, 0, 1, 1])
    u, t = solver.solve(t_mesh)
    Q = u[:,0]
    P = u[:,1]
    S = u[:,2]
    E = u[:,3]
    return Q, P, S, E, t

def simulate_biochemical_process():
    alpha = 1.5
    beta = 1

    epsilon = 0.1
    T = 8
    dt = 0.01

    # Very small epsilon:
    #epsilon = 0.005
    #dt = 0.001
    #T = 0.05

    Q, P, S, E, t = biochemical_solver(alpha, beta, epsilon, T, dt)

    import matplotlib.pyplot as plt
    plt.plot(t, Q, t, P, t, S, t, E)
    plt.legend(['complex', 'product', 'substrate', 'enzyme'],
               loc='center right')
    plt.title('alpha=%g, beta=%g, epsilon=%g' % (alpha, beta, epsilon))
    if epsilon < 0.05:
        plt.axis([t[0], t[-1], -0.05, 1.1])
    plt.savefig('tmp.png');  plt.savefig('tmp.pdf')
    plt.show()

def boundary_layer1D():
    import sympy as sym
    x, Pe = sym.symbols('x Pe')

    def u(x, Pe, module):
        return (1 - module.exp(x*Pe))/(1 - module.exp(Pe))

    u_formula = u(x, Pe, sym)
    ux_formula = sym.diff(u_formula, x)
    uxx_formula = sym.diff(ux_formula, x)
    print ux_formula, uxx_formula
    print 'u_x:', sym.simplify(ux_formula.subs(x, 1))
    print sym.simplify(ux_formula.subs(x, 1)).series(Pe, 0, 3)
    print 'u_xx:', sym.simplify(uxx_formula.subs(x, 1))
    print sym.simplify(uxx_formula.subs(x, 1)).series(Pe, 0, 3)

    import matplotlib.pyplot as plt
    import numpy as np
    x = np.linspace(0, 1, 10001)
    Pe = 1
    u_num1 = u(x, Pe, np)
    Pe = 50
    u_num2 = u(x, Pe, np)
    plt.plot(x, u_num1, x, u_num2)
    plt.legend(['Pe=1', 'Pe=50'], loc='upper left')
    plt.savefig('tmp.png');  plt.savefig('tmp.pdf')
    plt.axis([0, 1, -0.1, 1])
    plt.show()

def boundary_layer1D_scale2():
    import sympy as sym
    x, Pe = sym.symbols('x Pe')

    def u(x, Pe, module):
        return (1 - module.exp(x))/(1 - module.exp(Pe))

    u_formula = u(x, Pe, sym)
    ux_formula = sym.diff(u_formula, x)
    uxx_formula = sym.diff(ux_formula, x)
    print ux_formula, uxx_formula
    print 'u_x:', sym.simplify(ux_formula.subs(x, Pe))
    print sym.simplify(ux_formula).series(Pe, 0, 3)
    print 'u_xx:', sym.simplify(uxx_formula.subs(x, Pe))
    print sym.simplify(uxx_formula).series(Pe, 0, 3)

    import matplotlib.pyplot as plt
    import numpy as np
    Pe_values = [1, 10, 25, 50]
    for Pe in Pe_values:
        x = np.linspace(0, Pe, 10001)
        u_num = u(x, Pe, np)
        plt.plot(x, u_num)
    plt.legend(['Pe=%d' % Pe for Pe in Pe_values], loc='lower left')
    plt.savefig('tmp.png');  plt.savefig('tmp.pdf')
    plt.axis([0, max(Pe_values), -0.4, 1])
    plt.show()


def solver_diffusion_FE(
    I, a, f, L, Nx, F, T, U_0, U_L, h=None, user_action=None):
    """
    Forward Euler scheme for the diffusion equation
    u_t = a*u_xx + f, u(x,0)=I(x).
    If U_0 is a function of t: u(0,t)=U_0(t)
    If U_L is a function of t: u(L,t)=U_L(t)
    If U_0 is None: du/dx(0,t)=0
    If U_L is None: du/dx(L,t)=0
    If U_0 is a number: Robin condition -a*du/dn(0,t)=h*(u-U_0)
    If U_L is a number: Robin condition -a*du/dn(L,t)=h*(u-U_0)
    """
    import numpy as np
    version = 'scalar'
    x = np.linspace(0, L, Nx+1)   # mesh points in space
    dx = x[1] - x[0]
    dt = F*dx**2/a
    Nt = int(round(T/float(dt)))
    t = np.linspace(0, T, Nt+1)   # mesh points in time

    if f is None:
        f = lambda x, t: 0 if isinstance(x, (float,int)) else np.zero_like(x)

    u   = np.zeros(Nx+1)   # solution array
    u_1 = np.zeros(Nx+1)   # solution at t-dt
    u_2 = np.zeros(Nx+1)   # solution at t-2*dt

    # Set initial condition
    for i in range(0,Nx+1):
        u_1[i] = I(x[i])

    if user_action is not None:
        user_action(u_1, x, t, 0)

    for n in range(0, Nt):
        # Update all inner points
        if version == 'scalar':
            for i in range(1, Nx):
                if callable(f):  # f(x,t)
                    u[i] = u_1[i] + \
                           F*(u_1[i-1] - 2*u_1[i] + u_1[i+1])\
                           + f(x[i], t[n])
                elif isinstance(f, (float,int)):
                    # f = f*(u-1)
                    u[i] = u_1[i] + \
                           F*(u_1[i-1] - 2*u_1[i] + u_1[i+1])\
                           + f*(u_1[i] - 1)

        elif version == 'vectorized':
            if callable(f):
                u[1:Nx] = u_1[1:Nx] +  \
                          F*(u_1[0:Nx-1] - 2*u_1[1:Nx] + u_1[2:Nx+1])\
                          + f(x[1:Nx], t[n])
            elif isinstance(f, (float,int)):
                # f = f*(u-1)
                u[1:Nx] = u_1[1:Nx] +  \
                          F*(u_1[0:Nx-1] - 2*u_1[1:Nx] + u_1[2:Nx+1])\
                          + f*(u_1[1:Nx] - 1)

        # Insert boundary conditions
        if callable(U_0):
            u[0] = U_0(t[n+1])
        elif U_0 is None:
            # Homogeneous Neumann condition
            i = 0
            u[i] = u_1[i] + F*(u_1[i+1] - 2*u_1[i] + u_1[i+1])
        elif isinstance(U_0, (float,int)):
            # Robin condition
            # u_-1 = u_1 + 2*dx/a*(u[i] - U_0)
            i = 0
            u[i] = u_1[i] + F*(u_1[i+1] + 2*dx*h/a*(u[i] - U_0)
                               - 2*u_1[i] + u_1[i+1])
        if callable(U_L):
            u[Nx] = U_L(t[n+1])
        elif U_L is None:
            # Homogeneous Neumann condition
            i = Nx
            u[i] = u_1[i] + F*(u_1[i-1] - 2*u_1[i] + u_1[i-1])
        elif isinstance(U_0, (float,int)):
            # Robin condition
            # u_Nx+1 = u_Nx-1 - 2*dx/a*(u[i] - U_0)
            i = Nx
            u[i] = u_1[i] + F*(u_1[i-1] - 2*u_1[i] +
                               u_1[i-1] - 2*dx*h/a*(u[i] - U_0))

        if user_action is not None:
            user_action(u, x, t, n+1)

        # Update u_1 before next step
        #u_1[:] = u  # safe, but slow
        u_1, u = u, u_1  # just switch references

def diffusion_oscillatory_BC():
    import scitools.std as plt

    def plot(u, x, t, n):
        plt.plot(x, u, 'r-', legend=['t=%.2f' % t[n]],
                 axis=[x[0], x[-1], -1.1, 1.1],
                 xlabel='$x$', ylabel='$u$',
                 savefig='tmp_%04d.png' % n)

    from math import sin, pi
    solver_diffusion_FE(
        I=lambda x: 0,
        a=0.5,
        f=None,
        L=4,
        Nx=30,
        F=0.5,
        T=8*pi,
        U_0=lambda t: sin(t),
        U_L=None,  # du/dx=0, x=L
        h=None,
        user_action=plot)

def diffusion_two_metal_pieces():
    import scitools.std as plt

    def plot(u, x, t, n):
        plt.plot(x, u, 'r-', legend=['t=%.5f' % t[n]],
                 axis=[x[0], x[-1], -0.1, 2.1],
                 xlabel='$x$', ylabel='$u$',
                 savefig='tmp_%04d.png' % n)

    beta = 0.1
    gamma = 2
    Nu = 0.1
    solver_diffusion_FE(
        I=lambda x: 0 if x < 0.5 else gamma,
        a=0.5,
        f=-beta,
        L=1,
        Nx=50,
        F=0.25,
        T=0.015,
        U_0=1,  # Robin condition U_s=1
        U_L=1,
        h=Nu,
        user_action=plot)

if __name__ == '__main__':
    #damped_forced_vibrations()
    #cooling_sine_Ts_dsolve()
    #generalized_exponential_decay()
    #forced_vibrations()
    #simulate_forced_vibrations1()
    #simulate_forced_vibrations3()
    #simulate_Gaussian_and_incoming_wave()
    #simulate_biochemical_process()
    #boundary_layer1D()
    #boundary_layer1D_scale2()
    #diffusion_oscillatory_BC()
    diffusion_two_metal_pieces()
