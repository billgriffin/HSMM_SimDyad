import os, sys, re
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from pymattutil.stats import flattendata
from pymattutil.plot import project_data, plot_gaussian_projection, pca
import scipy.special as special
import scipy.stats as stats
from nestLists import isplit, printplus
from pprint import pprint
import itertools

if len(sys.argv) < 4:
    #print "usage: python GenHSMMGraph.py configure.txt result.txt 1"
    exit
    '''python GenHSMMGraph.py conf/paramterX.conf results/correspond_result.txt couple_index
   
   Ex: generate the result graph of first couple:
   python ExtractCoupleStatesHSMM.py conf/parameter0.conf results/All_All_10000_4.0_4.0_30_50.txt 
   or
   conf/parameter16_allHigh50.conf results/High_All_10000_8.0_8.0_40_50.txt   '''   
 
conf_path = sys.argv[1]
result_path = sys.argv[2]

# read configuration file
f = open(conf_path)
f.readline()
f.readline()
couples_ids = eval(f.readline().strip())
num_couples = len(couples_ids)
f.close()

f = open(result_path)
line = f.readline()
startRead = False
startSystem = False

states_used = []
couple_binary_states = {}

while len(line) > 0:        
    if not startRead and line.strip().startswith("Individual States"):
        startRead = True
    if startRead and re.match(r'^[1-9]', line.strip()):
        items = line.strip().split()
        state = int(items[1])  #state
        states_used.append(state)
    if startRead and line.strip().startswith("Estimated"):
        startRead = False
        
    line = f.readline()    
f.close()

#print states_used

list_break=[]
maxvalue = len(states_used)

for i in range(len(states_used)):
        list_break.append(states_used[i])
        x = states_used[i]
        if i < maxvalue -2:            
            y = states_used[i+1]
        else:     
            list_break.append(x)
        if y < x: 
            list_break.append(None)

list_break.pop() #remove last element
list_break.pop() #remove last None

c_states = isplit(list_break,(None,))

binary_matrix = np.zeros([30,30]) #fit to data

for i in range(len(c_states)):
    for j in range(len(c_states[i])):
        loc = c_states[i][j]
        binary_matrix[i][loc] = 1
        
print couples_ids
print
print binary_matrix
for i in range(len(couples_ids)):
    couple_binary_states[couples_ids[i]]= ( binary_matrix[i] )
    
for k, v in sorted(couple_binary_states.items()): 
    print k, v
print
    
printplus(couple_binary_states)    
#print 

#pprint(couple_binary_states)