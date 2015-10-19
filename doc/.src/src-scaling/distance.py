"""Compute with a formula involving units using parampool for input."""

def define_input():
    pool = [
        'Main', [
            dict(name='initial velocity', default=1.0, unit='m/s'),
            dict(name='acceleration', default=1.0, unit='m/s**2'),
            dict(name='time', default=10.0, unit='s')
            ]
        ]

    from parampool.pool.UI import listtree2Pool
    pool = listtree2Pool(pool)  # convert list to Pool object
    return pool

def distance(pool):
    v_0 = pool.get_value('initial velocity')
    a = pool.get_value('acceleration')
    t = pool.get_value('time')
    s = v_0*t + 0.5*a*t**2
    return s

def distance_unit(pool):
    """Compute distance $s = v_0t + \frac{1}{2}at^2$. (DocOnce)"""
    # Compute with units
    from parampool.PhysicalQuantities import PhysicalQuantity as PQ
    v_0 = pool.get_value_unit('initial velocity')
    a = pool.get_value_unit('acceleration')
    t = pool.get_value_unit('time')
    s = v_0*t + 0.5*a*t**2
    return s.getValue(), s.getUnitName()

def distance_unit2(pool):
    # Wrap result from distance_unit in HTML
    s, s_unit = distance_unit(pool)
    return '<b>Distance:</b> %.2f %s' % (s, s_unit)

def distance_table(pool):
    """Grab multiple values of parameters from the pool."""
    table = []
    for v_0 in pool.get_values('initial velocity'):
        for a in pool.get_values('acceleration'):
            for t in pool.get_values('time'):
                s = v_0*t + 0.5*a*t**2
                table.append((v_0, a, t, s))
    return table

def main():
    pool = define_input()
    # Can define other default values in a file: --poolfile name
    from parampool.pool.UI import set_defaults_from_file
    pool = set_defaults_from_file(pool)
    # Can override default values on the command line
    from parampool.pool.UI import set_values_from_command_line
    pool = set_values_from_command_line(pool)

    s, s_unit = distance_unit(pool)
    print 's=%g' % s, s_unit

def main_table():
    """Make a table of s values based on multiple input of v_0, t, a."""
    pool = define_input()
    # Can define other default values in a file: --poolfile name
    from parampool.pool.UI import set_defaults_from_file
    pool = set_defaults_from_file(pool)
    # Can override default values on the command line
    from parampool.pool.UI import set_values_from_command_line
    pool = set_values_from_command_line(pool)

    table = distance_table(pool)
    print '|-----------------------------------------------------|'
    print '|      v_0   |      a     |      t     |      s       |'
    print '|-----------------------------------------------------|'
    for v_0, a, t, s in table:
        print '|%11.3f | %10.3f | %10.3f | %12.3f |' % (v_0, a, t, s)
    print '|-----------------------------------------------------|'

if __name__ == '__main__':
    #main()
    main_table()
