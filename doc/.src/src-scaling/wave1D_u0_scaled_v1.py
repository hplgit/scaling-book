import sys, os
# Enable loading modules in ../../decay/src-decay
sys.path.insert(0, os.path.join(os.pardir, os.pardir,
                                'wave', 'src-wave', 'wave1D'))
from wave1D_u0 import solver as solver_unscaled

def solver_scaled_prev(I, dt, C, T):
    """
    Solve 1D wave equation in dimensionless form.
    """
    # store solution in files? joblib? or just list in memory?
    # joblib doesn't handle I... Store u in joblib? retrieve/store Yes!
    def action(u, x, t, n):
        if n == 0:
            save('x', x)
            save('t', t)
        save('u%d' % n, u)

    # Make a hash of the arguments
    import inspect, hashlib
    data = inspect.getsource(I) + '_' + str(dt) + '_' + \
           str(C) + '_' + str(T)
    hashed_input = hashlib.sha1(data).hexdigest()

    print 'hash:', hashed_input
    cachedir = 'tmp_%s' % hashed_input
    import joblib
    memory = joblib.Memory(cachedir=cachedir, verbose=1)

    @memory.cache(ignore=['data'])
    def retrieve(name, data=None):
        print 'joblib save of', name
        return data

    save = retrieve

    if os.path.isdir(cachedir):
        return retrieve
    else:
        print 'Computing the numerical solution'
        solver_unscaled(
            I=I, V=0, f=0, c=1, L=1, dt=dt, C=C, T=T,
            user_action=action)
        return retrieve

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

    import joblib
    memory = joblib.Memory(cachedir=cachedir, verbose=1)

    def retrieve(name, data=None):
        print 'joblib save of', name
        return data

    retrieve = memory.cache(retrieve, ignore=['data'])
    save = retrieve

    def action(u, x, t, n):
        if n == 0:
            save('x', x)
            save('t', t)
        save('u%d' % n, u)

    if is_computed:
        print 'No need to compute the numerical solution'
        return retrieve
    else:
        print 'Computing the numerical solution'
        solver_unscaled(
            I=I, V=0, f=0, c=1, L=1, dt=dt, C=C, T=T,
            user_action=action)
        return retrieve


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
    T = 5
    T = 0.5
    # Choose dt the same as the stability limit for Nx=50
    dt = L/50./1

    def I(x):
        return x/x0 if x < x0 else (L-x)/(L-x0)

    retrieve = solver_scaled(I, dt, C, T)
    if not animate:
        return retrieve

    from scitools.std import plot
    x = retrieve('x')
    t = retrieve('t')
    for n in range(len(t)):
        u = retrieve('u%d' %n)
        plot(x, u, 'r-', label='t=%.2f' % t[n],
             axis=[x[0], x[-1], -1.2, 1.2])
    return retrieve

def guitar(C, I_max, c, L):
    """Triangular wave (pulled guitar string). Unscaled version."""
    retrieve = guitar_scaled(C, animate=False)
    x = retrieve('x')
    t = retrieve('t')
    x = unscale_x(x, I_max, c, L)
    t = unscale_t(t, I_max, c, L)

    from scitools.std import plot
    for n in range(len(t)):
        u = retrieve('u%d' %n)
        u = unscale_u(u, I_max, c, L)
        plot(x, u, 'r-', label='t=%.2f' % t[n],
             axis=[x[0], x[-1], -I_max*1.2, 1.2*I_max])

if __name__ == '__main__':
    guitar_scaled(0.6, animate=False)
    guitar(C=0.6, I_max=.005, c=0.5, L=0.5)
    guitar(C=0.6, I_max=.005, c=0.5, L=2)
    guitar(C=0.8, I_max=.005, c=0.5, L=0.2)
