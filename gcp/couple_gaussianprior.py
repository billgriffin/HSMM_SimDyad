import os
import shutil
import math
import numpy
from gaussian_prior import *
import scipy.stats

file_dict = dict()
file_dict["Cluster"] = {"c1": [203,212,217,225],
                     "c2": [204, 205, 208, 209, 210, 211, 214, 215, 219, 224],
                     "c3": [206, 207, 218, 220, 222, 228, 230],
                     "c4": [200,202, 213],
                     "c5": [201, 221, 223, 226, 227, 229]}
"""
file_dict["High"] = {"All": [200,203,206,208,209,211,212,213,217,225],
                     "Cluster1": [200,203,206,208,211,212,213],
                     "Cluster2": [209,217,225]}
file_dict["Med"]  = {"All": [202,205,207,210,214,215,218,219,220,228],
                     "Cluster1": [202,214,215,218,220,228],
                     "Cluster2": [205,207,210,219]}
file_dict["Low"]  = {"All": [201,204,221,222,223,224,226,227,229,230],
                     "Cluster1": [201,204,222,223,226,227],
                     "Cluster2": [221,224,229,230]}
"""
o = open('couple_gaussian_prior.csv','w')
all_data = []

for cat in file_dict:
    # cat: Hight, Med, Low
    file_groups_dict = file_dict[cat]
    for group in file_groups_dict:
        print group
        # group: All, Cluster1, Cluster2

        # Setup the inital prior...
        gp = GaussianPrior(2)
        gp.addPrior(numpy.array([0.0,0.0]),numpy.array([[100.0,0.0],[0.0,100.0]]))

        file_group = file_groups_dict[group]
        state_dict = {} # state: length
        state_cnt_dict = {} # state: duration_count
        for i in range(1,10):
            for j in range(1,10):
                state_dict[(i,j)] = 0
                state_cnt_dict[(i,j)] = 0            
        prev_stat = None

        for f in file_group:
            fname = '../matAll/c%s.txt' % (f)
            data = numpy.transpose(numpy.loadtxt(fname, usecols=(0,1),unpack=True))
            for sample in data:
                gp.addSample(sample)
                stat = (int(sample[0]), int(sample[1]))
                if prev_stat != stat:
                    state_cnt_dict[stat] += 1
                    prev_stat = stat
                state_dict[stat] += 1

            all_data.append(data)

        avg_stat_dur = []
        for stat in state_dict:
            sum_dur = state_dict[stat]
            dur_occur_times = state_cnt_dict[stat]
            if dur_occur_times == 0:
                #avg_dur_of_stat = 0
                pass
            else:
                avg_dur_of_stat = sum_dur / float(dur_occur_times)
                avg_stat_dur.append(avg_dur_of_stat)

        print avg_stat_dur
        k, loc, theta = scipy.stats.gamma.fit(avg_stat_dur)
        print k, theta
        # results
        mu = gp.getMu()
        lbd = gp.getLambda()
        n = gp.getN() -2 

        o.write('%s,%s,%s,%s,%s,%s,%s\n' %(cat, group, mu, lbd, n, k, theta))

gp = GaussianPrior(2)
gp.addPrior(numpy.array([0.0,0.0]),numpy.array([[100.0,0.0],[0.0,100.0]]))

for data in all_data:
    for sample in data:
        gp.addSample(sample)
    #gp.addSamples(data)
mu = gp.getMu()
lbd = gp.getLambda()
n = gp.getN() - 2

k, loc, theta = scipy.stats.gamma.fit(avg_stat_dur)

print avg_stat_dur

o.write('All,All,%s,%s,%s,%s,%s\n' %(mu, lbd, n, loc, theta))

o.close()
