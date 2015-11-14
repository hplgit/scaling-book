"""
Solve the scaled wave equation using a solver with dimensions
(wave1D_u0.py).
Detect previously run cases and reload from file if possible.
"""

"""
Method:

The package `joblib` can be used for making a function that detects if
a case has already been run and in that case the previous solution can
be returned from a database.  However, it turns out that `joblib`
cannot handle functions with function arguments, which we have a lot
of in the `solver` functions for 1D wave equations.

A manual strategy taken from wave1D_dn_vc.py and explained in the book
Finite difference computing with PDEs, by Langtangen and Linge, 2015,
is to convert all input data to the `solver` function to a string,
which is thereafter converted to an SHA1 hash string (via
`hashlib.sha1`) and used to recognize the input.  A SHA1 string is
also suitable as part of a file or directory name where computed
solutions can be stored.

We can, in the wave equation solver retrieve the solution, rather than
computing it, if the hash string is the same (because then the
computations have already been done). This can save a lot of
computations if a scaled solution can be reused in a number of cases
with dimensions. We will sketch the code that implements the idea.

A solver for the scaled problem is first developed. We limit the
focus to the simple constant-coefficient wave equation with u_t(x,0)=0.
The solver for the unscaled problem is taken from the previously
mentioned wave1D_u0.py file.

def solver_scaled(I, dt, C, T):
    """
    Solve 1D wave equation in dimensionless form.
    """
    # Make a hash of the arguments
    import inspect, hashlib
    data = inspect.getsource(I) + '_' + str(dt) + '_' + \
           str(C) + '_' + str(T)
    # Not fool proof: if x0 changes value, the source code of I
    # is still the same, and no recomputation takes place...
    hashed_input = hashlib.sha1(data).hexdigest()

    # Use joblib-based tool (class Storage) to store already
    # computed solutions in files
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

This function stores arrays on disk with use of joblib (class Storage)
and recognizing previous input through a hash string. If the input is
the same, the hash is the same and we can test on the existence of a
directory whose name contains the hash. If that directory exists, the
solution for this set of input data is already computed, and we can
just return the `storage` object from which one can retrieve the space
and time mesh as well as all the solutions u0, u1, and so on.

Although the partial differential equation model has no physical
parameters (assuming, for a guitar string, that x_0 fixed), the
corresponding numerical model depends on the Courant number C and the
duration T of the simulations.

A specific application of this simple solver is the vibrations of a guitar
string, see function guitar_scaled. The scaled version depends only on C (if we say T is fixed
and Nx is fixed through dt).
Any unscaled case can be solved by the guitar function below.

If guitar_scaled figures out that the scaled problem is already
solved, it just returns the storage object, otherwise it performs
calculations.  Anyway, we retrieve the space and time mesh as well as
all the solutions.  (The plot function from SciTools
(https://github.com/hplgit/scitools) is used for compact code for
animation, but Matplotlib can equally well be used - with a bit more
coding.)

Suppose we run three calls to guitar with three different values of
I_max. The output will be

  Computing the numerical solution
  No need to compute the numerical solution
  No need to compute the numerical solution

This indicates that we rely on the scaled solution for the two other
cases with different I_max parameter. Running such a program again
will avoid all computations and show movies solely based on
precomputed file data.

"""

import sys, os
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
    # Not fool proof: if x0 changes value, the source code of I
    # is still the same, and no recomputation takes place...
    hashed_input = hashlib.sha1(data).hexdigest()

    # Use joblib-based tool (class Storage) to store already
    # computed solutions in files
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
