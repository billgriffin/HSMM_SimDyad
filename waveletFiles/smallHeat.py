from random import gauss
from pylab import *
import numpy as np

from pylab import * # or just launch "IPython -pylab" from the command line

a = np.array([[37.319,35.685,35.971,37.376],
              [36.711,34.183,34.817,37.514],
              [34.222,36.005,34.691,36.175]])


# We create a custom colormap:
myblue = cm.colors.LinearSegmentedColormap("myblue", {
    'red':   [(0, 1, 1), (1, 0, 0)], 
    'green': [(0, 1, 1), (1, 0, 0)],
    'blue':  [(0, 1, 1), (1, 1, 1)]})

#for i in xrange(0,6):
    #for j in xrange(0,8):
        #text(j,i, "{0:5.2f}".format(a[i][j]),
             #horizontalalignment="center",
             #verticalalignment="center")   

# Plotting the graph:
xticks(arange(4),('10','20','30','40'))
yticks(arange(3),('High','Med','Low'))

imshow(a, cmap='jet',interpolation='nearest') #cmap=myblue)
cbar = colorbar()
cbar.ax.set_ylabel('Distance Value')
show()