import sys, os
# Enable loading modules in ../../wave/src-wave
sys.path.insert(0, os.path.join(os.pardir, os.pardir,
                                'wave', 'src-wave', 'wave1D'))
from wave1D_dn_vc import solver as solver_unscaled, \
     PlotAndStoreSolution

def solver_scaled(I, V, f, c, U_0, U_L, L, dt, C, T,
                  version='vectorized',
                  stability_safety_factor=1.0):
    """Solve 1D wave equation in dimensionless form."""
    action = PlotAndStoreSolution(
        '', -1.3, 1.3,  # assume properly scaled u
        every_frame=1,
        filename='tmpdata')
    print 'Computing the numerical solution'
    cpu, hashed_input = solver_unscaled(
        I=I, V=V, f=f, c=c, U_0=U_0, U_L=U_L,
        L=1, dt=dt, C=C, T=T,
        user_action=action, version=version,
        stability_safety_factor=stability_safety_factor)
    if cpu > 0:
        print 'Computing the numerical solution'
        action.close_file(hashed_input)
    return filedata

import joblib
disk_memory = joblib.Memory(cachedir='temp')
solver_scaled = disk_memory.cache(solver_scaled)

def main():
    # Read parameters, solve and plot
    I, a, T, theta, dt_values = read_command_line_argparse()
    dt = dt_values[0]  # use only the first dt value
    u_scaled, t_scaled = solver_scaled(T, dt, theta)
    u, t = unscale(u_scaled, t_scaled, I, a)

    plt.figure()
    plt.plot(t_scaled, u_scaled)
    plt.xlabel('scaled time'); plt.ylabel('scaled velocity')
    plt.title('Universial solution of scaled problem')
    plt.savefig('tmp1.png');  plt.savefig('tmp1.pdf')

    plt.figure()
    plt.plot(t, u)
    plt.xlabel('t'); plt.ylabel('u')
    plt.title('I=%g, a=%g, theta=%g' % (I, a, theta))
    plt.savefig('tmp2.png'); plt.savefig('tmp2.pdf')
    plt.show()

if __name__ == '__main__':
    main()
