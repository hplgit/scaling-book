"""
Solver function that stores the simulation on file and repeats
the simulation only when necessary, otherwise the simulation
is grabbed from the file.

Computational algorithm:

 o A computed solution u(t) is stored in a file with name `u_scaled.dat`.
 o The first line in the file contains T, dt, and theta
   used to compute the stored u(t).
 o The T, dt, and theta parameters are read from the first line
   in the file and compared with those required by the user.
 o If one of the three parameters changes, the solution in the file
   must be recomputed.
"""
from decay import solver as solver_unscaled
import numpy as np
import sys, os

def solver_scaled(T, dt, theta):
    """
    Solve u'=-u, u(0)=1 for (0,T] with step dt and theta method.
    """
    # Is the scaled problem already solved and dimensionless
    # curve available from file?
    # See if u_scaled.dat has the right parameters.
    already_computed = False
    datafile = 'u_scaled.dat'
    if os.path.isfile(datafile):      # does u_scaled.dat exist?
        infile = open(datafile, 'r')
        infoline = infile.readline()  # read the first line
        words = infoline.split()      # split line into words
        T_, dt_, theta_ = [float(w) for w in words]
        if T_ == T and dt_ == dt and theta_ == theta:
            # The file was computed with the desired data, load
            # the solution into arrays
            data = np.loadtxt(infile)
            u_scaled = data[1,:]
            t_scaled = data[0,:]
            print 'Read scaled solution from file'
            already_computed = True
        infile.close()
    if not already_computed:
        # T, dt or theta is different from u_scaled.dat
        u_scaled, t_scaled = \
           solver_unscaled(I=1, a=1, T=T, dt=dt, theta=theta)
        outfile = open(datafile, 'w')
        outfile.write('%f %f %.1f\n' % (T, dt, theta))
        # np.savetxt saves a two-dim array (table) to file
        np.savetxt(outfile, np.array([t_scaled, u_scaled]))
        outfile.close()
        print 'Computed scaled solution'
    return u_scaled, t_scaled

def unscale(u_scaled, t_scaled, I, a):
    return I*u_scaled, a*t_scaled

from decay import read_command_line_argparse

def main():
    # Read parameters, solve and plot
    I, a, T, theta, dt_values = read_command_line_argparse()
    dt = dt_values[0]  # use only the first dt value
    u_scaled, t_scaled = solver_scaled(T, dt, theta)
    u, t = unscale(u_scaled, t_scaled, I, a)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(t_scaled, u_scaled)
    plt.xlabel('scaled time'); plt.ylabel('scaled velocity')
    plt.title('Universial solution of scaled problem')
    plt.savefig('tmp1.png');  plt.savefig('tmp1.pdf')

    plt.figure()
    plt.plot(t, u)
    plt.xlabel('t'); plt.ylabel('u')
    plt.title('I=%g, a=%g, theta=%g' % (I, a, theta))
    plt.savefig('tmp.png')
    plt.show()

if __name__ == '__main__':
    main()
