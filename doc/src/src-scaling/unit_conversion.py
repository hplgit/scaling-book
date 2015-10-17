from PhysicalQuantities import PhysicalQuantity as PQ
v = PQ('120 yd/min')   # velocity
t = PQ('1 h')          # time
s = v*t                # distance
print s                # s is string
s.convertToUnit('m')
print s
print s.getValue()     # florat
print s.getUnitName()  # string
v.convertToUnit('km/h')
print v
v.convertToUnit('m/s')
print v

c = PQ('1 cal/g/K')
c.convertToUnit('J/(g*K)')
print 'Specific heat capacity of water (at const. pressure):', c

d = PQ('1000 kg/m**3')
d.convertToUnit('oz/floz')
print d
d = PQ('1.05 oz/floz')
d.convertToUnit('kg/m**3')
print d
