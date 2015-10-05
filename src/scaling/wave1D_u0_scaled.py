import sys, os
# Enable loading modules in the wave eq solver and softeng2 dirs
sys.path.insert(0, os.path.join(
    os.pardir, os.pardir, 'wave', 'src-wave', 'wave1D'))
sys.path.insert(0, os.path.join(
    os.pardir, os.pardir, 'softeng2', 'src-softeng2'))
from wave1D_u0 import solver as solver_unscaled
from Storage import Storage

def solver_scaled(I, dt, C, T):
    """
    Solve 1D wave equation in dimensionless form.
    """
    # Make a hash of the arguments
    import inspect, hashlib
    data = inspect.getsource(I) + '_' + str(dt) + '_' + \
           str(C) + '_' + str(T)
    # Not fool proof: if x0 changes value, I source is the same...
    hashed_input = hashlib.sha1(data).hexdigest()

    cachedir = 'tmp_%s' % hashed_input
    is_computed = os.path.isdir(cachedir)
    print 'cachedir:', cachedir, is_computed
    storage = Storage(cachedir, verbose=0)

    def action(u, x, t, n):
        if n == 0:
            storage.save('x', x)
            storage.save('t', t)
        storage.save('u%d' % n, u)

    if is_computed:
        print 'No need to compute the numerical solution'
        return storage
    else:
        print 'Computing the numerical solution'
        solver_unscaled(
            I=I, V=0, f=0, c=1, L=1, dt=dt, C=C, T=T,
            user_action=action)
        return storage


def unscale_u(u, I_max, c, L):
    return I_max*u

def unscale_x(x, I_max, c, L):
    return x*L

def unscale_t(t, I_max, c, L):
    return t*L**2/float(c)

def guitar_scaled(C, animate=True):
    """Triangular wave (pulled guitar string)."""
    L = 1.0
    x0 = 0.8*L
    T = 2
    Nx = 50; dx = L/float(Nx)
    dt = dx/1  # Choose dt at the stability limit

    def I(x):
        return x/x0 if x < x0 else (L-x)/(L-x0)

    storage = solver_scaled(I, dt, C, T)
    if not animate:
        return storage

    from scitools.std import plot
    x = storage.retrieve('x')
    t = storage.retrieve('t')
    for n in range(len(t)):
        u = storage.retrieve('u%d' %n)
        plot(x, u, 'r-', label='t=%.2f' % t[n],
             axis=[x[0], x[-1], -1.2, 1.2])
    return storage

def guitar(C, I_max, c, L):
    """Triangular wave (pulled guitar string). Unscaled version."""
    storage = guitar_scaled(C, animate=False)
    x = storage.retrieve('x')
    t = storage.retrieve('t')
    x = unscale_x(x, I_max, c, L)
    t = unscale_t(t, I_max, c, L)

    from scitools.std import plot
    for n in range(len(t)):
        u = storage.retrieve('u%d' %n)
        u = unscale_u(u, I_max, c, L)
        plot(x, u, 'r-', label='t=%.2f' % t[n],
             axis=[x[0], x[-1], -I_max*1.2, 1.2*I_max])

if __name__ == '__main__':
    guitar_scaled(0.6, animate=False)
    guitar(C=0.6, I_max=.005, c=0.5, L=0.5)
    guitar(C=0.6, I_max=.005, c=0.5, L=2)
    guitar(C=0.8, I_max=.005, c=0.5, L=0.2)
