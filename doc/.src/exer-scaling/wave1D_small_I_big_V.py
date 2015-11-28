import sys, os
sys.path.insert(0, os.path.join(os.pardir, 'src-scaling'))

def simulate_I_and_V():
    from wave1D_dn import solver, viz
    from math import pi
    from numpy import exp
    alpha = 0.0001

    def I(x):
        return alpha*exp(-50*(x - 0.5)**2)

    def V(x):
        # scaled V = 2*(orig V)/mean(V)
        # mean of exp(-50*(x - 0.3)**2) is 1/4
        return -8*exp(-50*(x - 0.3)**2)

    L = 1
    c = 1
    Nx = 80; dx = L/float(Nx); dt = dx/c
    #solver(I=I, V=V, f=0, U_0=None, U_L=None, L=1, dt=dt, C=1, T=4,
    #       user_action=myplotter)
    viz(I=I, V=V, f=0, c=1,
        #U_0=None, U_L=None,  # gives growing amplitude for V!=0
        U_0=0, U_L=0,
        L=L, dt=dt, C=1,
        T=4, umin=-(alpha+2), umax=(alpha+2),
        version='vectorized', animate=True)

simulate_I_and_V()
