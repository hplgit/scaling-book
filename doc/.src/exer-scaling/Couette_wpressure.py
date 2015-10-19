import sympy as sym
mu, beta, z, H = sym.symbols('mu beta z H',
                             real=True, positive=True)
U0, C1, C2 = sym.symbols('U0 C1 C2', real=True)

# Integrate u''(z) = -beta/mu twice and add integration constants
u = sym.integrate(sym.integrate(-beta/mu, z) + C1, z) + C2

# Use the boundary conditions
eq = [sym.Eq(u.subs(z, 0), 0),
      sym.Eq(u.subs(z, H), U0)]
s = sym.solve(eq, [C1, C2])
print s
u = u.subs(C1, s[C1]).subs(C2, s[C2])
u = sym.simplify(sym.expand(u))
print u
print sym.latex(u)

# Find max u
dudz = sym.diff(u, z)
s = sym.solve(dudz, z)
print s
umax = u.subs(z, s[0])
umax = sym.simplify(sym.expand(umax))
print umax
print sym.latex(umax)
