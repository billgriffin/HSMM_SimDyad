#!/usr/bin/env python
from mayavi import mlab
import numpy as np
import sys
from scipy.special import sph_harm
#n,m=int(sys.argv[1]),int(sys.argv[2])
m = 10
n = 15
r = 0.3
phi, theta = np.mgrid[0:np.pi:101j, 0:2*np.pi:101j]
x = r*np.sin(phi)*np.cos(theta)
y = r*np.sin(phi)*np.sin(theta)
z = r*np.cos(phi)
s = sph_harm(m, n, theta, phi).real
mlab.mesh(x, y, z, scalars=s, colormap='jet')
s[s<0]*=0.97
s /= s.max()
#mlab.mesh(s*x+1, s*y, s*z, scalars=s, colormap='Spectral')
mlab.show()
