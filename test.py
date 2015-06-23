#!/packages/python-2.7/bin/python

import scipy
import numpy as np
import matplotlib
import random,time


time.sleep(10)
fname = str(random.random())
print fname
o = open(fname+'.txt','w')
o.write(fname)
o.close()
