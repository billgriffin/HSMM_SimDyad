import fileinput
import numpy as np
from numpy.random import randn, shuffle, choice
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import cPickle as pickle


simStates = pickle.load( open( "simValues.pck", "rb" ) )

realStates = pickle.load( open( "realValues.pck", "rb" ) )

sns.set(palette="Set2")

print len(simStates)
print len(realStates)


shrt = []

### if x > y: do 1, else do 2
for i in range(706):
    shrt.append(realStates[i])

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
c1, c2 = sns.color_palette("Set1", 2)
#?c1, c2 = sns.cubehelix_palette(2)

g = sns.tsplot(simStates, color=c1, ax=ax1)
f =sns.tsplot(shrt, color=c2, ax=ax2)
g.set_ylim(0,10)
f.set_ylim(0,10)
### set the same y scale
g.set_title('Simulated States Trace From A Cluster 3 Couple')
g.set_ylabel("States")
f.set_title('Original States Trace From A Cluster 3 Couple')
f.set_xlabel("Time (seconds)")
f.set_ylabel("States");


plt.show()

