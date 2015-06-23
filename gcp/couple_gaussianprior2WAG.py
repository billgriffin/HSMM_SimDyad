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
o = open('couple_gaussian_priorWAGMod.csv','w')
all_data = []

for cat in file_dict:
    # cat: Hight, Med, Low
    file_groups_dict = file_dict[cat]
    for group in file_groups_dict:
        # group: All, Cluster1, Cluster2

        # Setup the inital prior...
        gp = GaussianPrior(2)
        gp.addPrior(numpy.array([0.0,0.0]),numpy.array([[100.0,0.0],[0.0,100.0]]))

        file_group = file_groups_dict[group]
        
        # for state 1, count how many times it's duration is 1,2,3,...,max_duration
        # Get the Possion distribution of durations of state 1
        # for state 2, ... (same)
        
        state_dur_dict = {} # state1: {dur:1, dur2:3,...}
        for i in range(1,10):
            state_dur_dict[float(i)] = {}
            
        prev_stat = None
        stat_count = 0
        
        for f in file_group:
            fname = '../matAll/c%s.txt' % (f)
            data = numpy.transpose(numpy.loadtxt(fname, usecols=(0,1),unpack=True))
            for sample in data:
                gp.addSample(sample)
                stat = sample[0] + sample[1] -1
                
                if prev_stat and prev_stat != stat:
                    if stat_count not in state_dur_dict[prev_stat]:
                        state_dur_dict[prev_stat][stat_count] = 0
                    state_dur_dict[prev_stat][stat_count] += 1
                    prev_stat = stat
                    stat_count = 0
                    
                if not prev_stat:
                    prev_stat = stat
                    
                stat_count += 1

            all_data.append(data)

        poisson_lmda_list = []
        for stat,v in state_dur_dict.iteritems():
            dur_counts = v.values()
            if len(dur_counts) > 0:
                poisson_lmda = sum(dur_counts) / float(len(dur_counts))
                poisson_lmda_list.append(poisson_lmda)
            else:
                pass
                #poisson_lmda_list.append(0)
            
            
        k, loc, theta = scipy.stats.gamma.fit(poisson_lmda_list)
        print k, theta
        # results
        mu = gp.getMu()
        lbd = gp.getLambda()
        n = gp.getN() -2

        o.write('%s,%s,%s,%s,%s,%s,%s\n' %(cat, group, mu, lbd, n, k, theta))

gp = GaussianPrior(2)
gp.addPrior(numpy.array([0.0,0.0]),numpy.array([[100.0,0.0],[0.0,100.0]]))

state_dur_dict = {} # state1: {dur:1, dur2:3,...}
for i in range(1,10):
    state_dur_dict[float(i)] = {}
    
prev_stat = None
stat_count = 0

for data in all_data:
    for sample in data:
        gp.addSample(sample)
        stat = sample[0] + sample[1] -1
        
        if prev_stat and prev_stat != stat:
            if stat_count not in state_dur_dict[prev_stat]:
                state_dur_dict[prev_stat][stat_count] = 0
            state_dur_dict[prev_stat][stat_count] += 1
            prev_stat = stat
            stat_count = 0
            
        if not prev_stat:
            prev_stat = stat
            
        stat_count += 1
        
        
    
poisson_lmda_list = []

for stat,v in state_dur_dict.iteritems():
    dur_counts = v.values()
    if len(dur_counts) > 0:
        poisson_lmda = sum(dur_counts) / float(len(dur_counts))
        poisson_lmda_list.append(poisson_lmda)
    #else:
        #poisson_lmda_list.append(0)
        #pass
    grab_freq = []
    grab_values=[]
    for k, v in state_dur_dict.iteritems(): 
        if k == 9:        
            grab_values.append(v.keys())
            grab_freq.append(v.items())        
    dog =  sum(grab_values, [])
    cat =  sum(grab_freq, [])
    dogcat_values = []
    for i in range(len(dog)):
        dogcat_values.append( cat[i][0] *cat[i][1])   
    dogcat_sum = (sum(dogcat_values) )/len(dogcat_values)
    print dogcat_sum   

        
#print grab_freq   
mu = gp.getMu()
lbd = gp.getLambda()
n = gp.getN() -2

k, loc, theta = scipy.stats.gamma.fit(poisson_lmda_list)

o.write('All,All,%s,%s,%s,%s,%s\n' %(mu, lbd, n, k, theta))

o.close()
print
