import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import time, sys


def solver(I, a, f, L, Nx, D, T, theta=0.5, u_L=1, u_R=0,
           user_action=None):
    """
    The a variable is an array of length Nx+1 holding the values of
    a(x) at the mesh points.

    Method: (implicit) theta-rule in time.

    Nx is the total number of mesh cells; mesh points are numbered
    from 0 to Nx.
    D = dt/dx**2 and implicitly specifies the time step.
    T is the stop time for the simulation.
    I is a function of x.

    user_action is a function of (u, x, t, n) where the calling code
    can add visualization, error computations, data analysis,
    store solutions, etc.
    """
    import time
    t0 = time.clock()

    x = np.linspace(0, L, Nx+1)   # mesh points in space
    dx = x[1] - x[0]
    dt = D*dx**2
    #print 'dt=%g' % dt
    Nt = int(round(T/float(dt)))
    t = np.linspace(0, T, Nt+1)   # mesh points in time

    if isinstance(a, (float,int)):
        a = np.zeros(Nx+1) + a
    if isinstance(u_L, (float,int)):
        u_L_ = float(u_L)  # must take copy of u_L number
        u_L = lambda t: u_L_
    if isinstance(u_R, (float,int)):
        u_R_ = float(u_R)  # must take copy of u_R number
        u_R = lambda t: u_R_

    u   = np.zeros(Nx+1)   # solution array at t[n+1]
    u_1 = np.zeros(Nx+1)   # solution at t[n]

    """
    Basic formula in the scheme:

    0.5*(a[i+1] + a[i])*(u[i+1] - u[i]) -
    0.5*(a[i] + a[i-1])*(u[i] - u[i-1])

    0.5*(a[i+1] + a[i])*u[i+1]
    0.5*(a[i] + a[i-1])*u[i-1]
    -0.5*(a[i+1] + 2*a[i] + a[i-1])*u[i]
    """
    Dl = 0.5*D*theta
    Dr = 0.5*D*(1-theta)

    # Representation of sparse matrix and right-hand side
    diagonal = np.zeros(Nx+1)
    lower    = np.zeros(Nx)
    upper    = np.zeros(Nx)
    b        = np.zeros(Nx+1)

    # Precompute sparse matrix (scipy format)
    diagonal[1:-1] = 1 + Dl*(a[2:] + 2*a[1:-1] + a[:-2])
    lower[:-1] = -Dl*(a[1:-1] + a[:-2])
    upper[1:]  = -Dl*(a[2:] + a[1:-1])
    # Insert boundary conditions
    diagonal[0] = 1
    upper[0] = 0
    diagonal[Nx] = 1
    lower[-1] = 0

    A = scipy.sparse.diags(
        diagonals=[diagonal, lower, upper],
        offsets=[0, -1, 1],
        shape=(Nx+1, Nx+1),
        format='csr')
    #print A.todense()

    # Set initial condition
    for i in range(0,Nx+1):
        u_1[i] = I(x[i])

    if user_action is not None:
        user_action(u_1, x, t, 0)

    # Time loop
    for n in range(0, Nt):
        b[1:-1] = u_1[1:-1] + Dr*(
            (a[2:] + a[1:-1])*(u_1[2:] - u_1[1:-1]) -
            (a[1:-1] + a[0:-2])*(u_1[1:-1] - u_1[:-2])) + \
            dt*theta*f(x[1:-1], t[n+1]) + \
            dt*(1-theta)*f(x[1:-1], t[n])
        # Boundary conditions
        b[0]  = u_L(t[n+1])
        b[-1] = u_R(t[n+1])
        # Solve
        u[:] = scipy.sparse.linalg.spsolve(A, b)

        if user_action is not None:
            user_action(u, x, t, n+1)

        # Switch variables before next step
        u_1, u = u, u_1

    t1 = time.clock()
    return t1-t0

def run(gamma, beta=10, delta=40, scaling=1, animate=False):
    """Run the scaled model for welding."""
    if scaling == 1:
        v = gamma
        a = 1
        L = 1.0
        b = 0.5*beta**2
    elif scaling == 2:
        v = 1
        a = 1.0/gamma
        L = 1.0
        b = 0.5*beta**2
    elif scaling == 3:
        v = 1
        a = beta/gamma
        L = beta
        b = 0.5

    ymin = 0
    # Need global ymax to be able change ymax in closure process_u
    global ymax
    ymax = 1.2

    I = lambda x: 0
    f = lambda x, t: delta*np.exp(-b*(x - v*t)**2)

    import time
    import scitools.std as plt
    plot_arrays = []

    def process_u(u, x, t, n):
        global ymax
        if animate:
            plt.plot(x, u, 'r-',
                     x, f(x, t[n])/delta, 'b-',
                     axis=[0, L, ymin, ymax], title='t=%f' % t[n],
                     xlabel='x', ylabel='u and f/%g' % delta)
        if t[n] == 0:
            time.sleep(1)
            plot_arrays.append(x)
        dt = t[1] - t[0]
        tol = dt/10.0
        if abs(t[n] - 0.2) < tol or abs(t[n] - 0.5) < tol:
            plot_arrays.append((u.copy(), f(x, t[n])/delta))
            if u.max() > ymax:
                ymax = u.max()

    Nx = 100
    D = 10
    T = 0.5
    u_L = u_R = 0
    theta = 1.0
    cpu = solver(
        I, a, f, L, Nx, D, T, theta, u_L, u_R, user_action=process_u)
    x = plot_arrays[0]
    plt.figure()
    for u, f in plot_arrays[1:]:
        plt.plot(x, u, 'r-', x, f, 'b--', axis=[x[0], x[-1], 0, ymax],
                 xlabel='$x$', ylabel=r'$u, \ f/%g$' % delta)
        plt.hold('on')
    plt.legend(['$u,\\ t=0.2$', '$f/%g,\\ t=0.2$' % delta,
                '$u,\\ t=0.5$', '$f/%g,\\ t=0.5$' % delta])
    filename = 'tmp1_gamma%g_s%d' % (gamma, scaling)
    if scaling == 1:
        s = 'diffusion'
    elif scaling == 2:
        s = 'source'
    elif scaling == 3:
        s = 'sigma'
    plt.title(r'$\beta = %g,\ \gamma = %g,\ $' % (beta, gamma)
              + 'scaling=%s' % s)
    plt.savefig(filename + '.pdf');  plt.savefig(filename + '.png')
    return cpu

def investigate():
    """Do scienfic experiments with the run function above."""
    # Clean up old files
    import glob, os
    for filename in glob.glob('tmp1_gamma*') + \
            glob.glob('welding_gamma*'):
        os.remove(filename)

    scaling_values = 1, 2, 3
    gamma_values = 1, 40, 5, 0.2, 0.025
    delta_values = {}  # delta_values[scaling][gamma]
    delta_values[1] = {0.025: 140, 0.2: 60,  1: 20, 5: 40, 40: 800}
    delta_values[2] = {0.025: 700, 0.2: 100, 1: 20, 5: 8,  40: 5}
    delta_values[3] = {0.025: 350, 0.2: 40,  1: 12, 5: 5,  40: 2}
    for gamma in gamma_values:
        for scaling in scaling_values:
            run(gamma=gamma, beta=10,
                delta=delta_values[scaling][gamma],
                scaling=scaling)

    # Combine images
    for gamma in gamma_values:
        for ext in 'pdf', 'png':
            cmd = 'doconce combine_images -' + str(len(scaling_values)) + ' '
            for s in scaling_values:
                cmd += ' tmp1_gamma%(gamma)g_s%(s)d.%(ext)s ' % vars()
            cmd += ' welding_gamma%(gamma)g.%(ext)s' % vars()
            os.system(cmd)
            # pdflatex doesn't like a dot (as in 0.2) in filenames...
            if '.' in str(gamma):
                os.rename(
                'welding_gamma%(gamma)g.%(ext)s' % vars(),
                ('welding_gamma%(gamma)g' % vars()).replace('.', '_')
                + '.' + ext)

if __name__ == '__main__':
    #run(gamma=1/40., beta=10, delta=40, scaling=2)
    investigate()
