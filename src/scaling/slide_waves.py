from wave1D_dn_vc import solver, PlotAndStoreSolution, np

def B(x, a, L, B0):
    """
    B is linear between - and -B0 on [0, a*L] andx<=a*L) and
    constant -B0 on [a*L, L].
    """
    if isinstance(x, (float,int)):
        return -B0*x/(a*L) if x <= a*L else -B0
    elif isinstance(x, np.ndarray):
        return np.where(x <= a*L, -B0*x/(a*L), -B0)

def S(x, A, x0, sigma):
    """
    Gaussian function centered at x0, with
    'standard devition' sigma and amplitude A.
    """
    return A*exp(-0.5*(x-x0)**2/sigma**2)

class PlotSurfaceAndBottom(PlotAndStoreSolution):
    def __init__(self, B, S, a, L, B0, A, x0, sigma, **kwargs):
        self.B, self.S = B, S  # functions for bottom and slide
        self.a, self.L, self.B0 = a, L, B0
        self.A, self.x0, self.sigma = A, x0, sigma
        PlotAndStoreSolution.__init__(self, **kwargs)

    def __call__(self, u, x, t, n):
        # Save solution u to a file using numpy.savez
        if self.filename is not None:
            name = 'u%04d' % n  # array name
            kwargs = {name: u}
            fname = '.' + self.filename + '_' + name + '.dat'
            savez(fname, **kwargs)
            self.t.append(t[n])  # store corresponding time value
            if n == 0:           # save x once
                savez('.' + self.filename + '_x.dat', x=x)

        # Animate
        if n % self.skip_frame != 0:
            return
        title = 'Nx=%d' % (x.size-1)
        if self.title:
            title = self.title + ' ' + title
        bottom = self.B(x, a, L, B0) + self.S(x, A, x0(t), sigma)
        if self.backend is None:
            # native matplotlib animation
            if n == 0:
                self.plt.ion()
                self.lines = self.plt.plot(
                    x, u, 'r-',
                    x, bottom, 'b-')
                self.plt.axis([x[0], x[-1],
                               self.yaxis[0], self.yaxis[1]])
                self.plt.xlabel('x')
                self.plt.ylabel('u')
                self.plt.title(title)
                self.plt.text(0.75, 1.0, 'C=0.25')
                self.plt.text(0.32, 1.0, 'C=1')
                self.plt.legend(['t=%.3f' % t[n], 'bottom'])
            else:
                # Update new solution
                self.lines[0].set_ydata(u)
                self.lines[1].set_ydata(bottom)
                self.plt.legend(['t=%.3f' % t[n]])
                self.plt.draw()
        else:
            # scitools.easyviz animation
            self.plt.plot(x, u, 'r-', x, bottom, 'b-',
                          xlabel='x', ylabel='u',
                          axis=[x[0], x[-1],
                                self.yaxis[0], self.yaxis[1]],
                          title=title,
                          show=self.screen_movie)
        # pause
        if t[n] == 0:
            time.sleep(2)  # let initial condition stay 2 s
        else:
            if self.pause is None:
                pause = 0.2 if u.size < 100 else 0
            time.sleep(pause)

        self.plt.savefig('frame_%04d.png' % (n))

def slides_waves():
    L = 10
    a = 3
    A = 0.3
    t0 = 3
    v = 1
    x0 = lambda t: v*t if t < t0 else v*t0
    sigma = 1.0

    def bottom(x, t):
        return 0 - ???
