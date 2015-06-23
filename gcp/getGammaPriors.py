import os
import shutil
import math
import numpy as np
from gaussian_prior import *
import scipy.stats
import fileinput

cluster1 = []
all_data = []
for line in fileinput.input('Cluster1.csv'):
   line = line.strip()
   line = line.split(',')
   cluster1.append(int(line[0]))
fileinput.close()

#for f in file_group:
            #fname = '../matAll/c%s.txt' % (f)
            #data = numpy.transpose(numpy.loadtxt(fname, usecols=(0,1),unpack=True))

#gp = GaussianPrior(2)
#gp.addPrior(numpy.array([0.0,0.0]),numpy.array([[100.0,0.0],[0.0,100.0]]))  
state_dur_dict = {} # state1: {dur:1, dur2:3,...}

for i in range(1,10):
   state_dur_dict[float(i)] = {}

   prev_stat = None
   stat_count = 0

   for sample in cluster1:
      #gp.addSample(sample)
      stat = sample 
   
      if prev_stat and prev_stat != stat:
         if stat_count not in state_dur_dict[prev_stat]:
            state_dur_dict[prev_stat][stat_count] = 0
         state_dur_dict[prev_stat][stat_count] += 1
         prev_stat = stat
         stat_count = 0
   
         if not prev_stat:
            prev_stat = stat
   
         stat_count += 1
   
      all_data.append(cluster1)
      
      poisson_lmda_list = []
      for stat,v in state_dur_dict.iteritems():
         dur_counts = v.values()
         if len(dur_counts) > 0:
            poisson_lmda = sum(dur_counts) / float(len(dur_counts))
            poisson_lmda_list.append(poisson_lmda)
         else:
            poisson_lmda_list.append(0)


k, loc, theta = scipy.stats.gamma.fit(poisson_lmda_list)
print k, theta

k, loc, theta = scipy.stats.gamma.fit(cluster1)
# python scale returns beta, which = 1/theta
print k, theta