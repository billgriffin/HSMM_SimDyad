""" generate time series values from base sequence  """
import scipy 
from scipy import stats
import numpy as np
from numpy import savetxt  #add
import numpy as np
from numpy import *
from numpy.random import *
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pywt
import glob
import os

file_dict = {}
file_dict["All"] = {"All": [200,203,206,208,209,211,212,213,217,225, 202,205,207,210,214,215,218,219,220,228, 201,204,221,222,223,224,226,227,229,230]}
file_dict["High"] = {"All": [200,203,206,208,209,211,212,213,217,225],
                     "Cluster1": [200,203,206,208,211,212,213],
                     "Cluster2": [209,217,225]}
file_dict["Med"]  = {"All": [202,205,207,210,214,215,218,219,220,228],
                     "Cluster1": [202,214,215,218,220,228],
                     "Cluster2": [205,207,210,219]}
file_dict["Low"]  = {"All": [201,204,221,222,223,224,226,227,229,230],
                     "Cluster1": [201,204,222,223,226,227],
                     "Cluster2": [221,224,229,230]}

wavelet_dict = {}

# Get all wavelet values from original couple data
def compute_wavlets():
    real_wavelet_dict = {}
    os.chdir("./data")
    for file in glob.glob("*.txt"):
        file_id = (int)(file[1:-4])
        dyad_data = np.loadtxt( file)
        male = []
        female = []
        dyadicstate = []
        for i in range(len(dyad_data)):
                male.append(int(dyad_data[i][0])) 
                female.append(int(dyad_data[i][1])) 
                dyadicstate.append( (int(dyad_data[i][0]) + int(dyad_data[i][1]) )/2.)        
        
        male = np.array(male)
        female = np.array(female)
        dyadicstate = np.array(dyadicstate)
        ### normalize the data
        dyadicstate_scaled = preprocessing.scale(dyadicstate) 
        
        (cA, cD) = pywt.dwt(dyadicstate_scaled , 'haar', mode='zpd')
        cA_real = cA
        cD_real = cD
        real_wavelet_dict[file_id] = (cA_real, cD_real)
    os.chdir("../")
    return real_wavelet_dict

real_wavelet_dict = compute_wavlets()

result = open("wavelets_result_detail.csv","w")
def extract_simulation(fname):
    keys = fname.split("_")
    cat_key = keys[0]
    group_key = keys[1]
    couple_ids = file_dict[cat_key][group_key]
    
    o = open(fname)
    line = o.readline()
    start_process = False
    c_idx = 0
    write = False
    avg_A = []
    avg_D = []    
    
    while len(line) > 0:
        line = line.strip()
        if line.startswith("Individual States"):
            line = o.readline()
            line = o.readline()
            line = o.readline()
            line = o.readline()
            line = o.readline()
            line = o.readline()  # move to detail
            start_process = True
            
        if start_process:
            line = o.readline()
            sim_dict = {}
            while not line.startswith("--"):
                line = line.strip()
                row = line.split()
                items = []
                for item in row:
                    if item != " ":
                            items.append(item)
                # create dict
                sim_dict[ items[1] ] = (float(items[3]), float(items[4]))
                line = o.readline()
            # simulation
            male = []
            female = []
            line = o.readline()
            line = o.readline()
            line = o.readline()
            line = o.readline()
            line = o.readline()
            line = o.readline()
            
            while not line.startswith("--"):
                line = line.strip()
                row = line.split()
                items = []
                for item in row:
                    if item != " ":
                            items.append(item)
                state = items[0]
                x,y = sim_dict[state]
                dur = int(float(items[1]))
                for i in range(dur):
                    male.append( x )
                    female.append( y )
                line = o.readline()
            # start using wavlet on male[] and female[]
            dyadicstate = []
            n = len(male)
            for i in range(n):
                dyadicstate.append( (male[i] + female[i])/2.0 )
            dyadicstate = np.array(dyadicstate)
            # normalize the data
            dyadicstate_scaled = preprocessing.scale(dyadicstate) 
            (cA, cD) = pywt.dwt(dyadicstate_scaled , 'haar', mode='zpd')
            cA_sim = cA
            cD_sim = cD
            # for i in cA: print i
            
            # compare
            couple_id = couple_ids[c_idx]
            cA_real, cD_real = real_wavelet_dict[couple_id] 
            
            distA = np.linalg.norm(cA_sim[:len(cA_real)] - cA_real)
            distD = np.linalg.norm(cD_sim[:len(cA_real)] - cD_real)
            #result.write("{0},{1},{2},{3}\n".format(fname, c_idx,  distA, distD)  )
            avg_A.append(distA)
            avg_D.append(distD)
                         
            c_idx += 1
            start_process = False
            
        line = o.readline()
    if len(avg_A) > 0:
        result.write("{0},,{1},{2},{3},{4}\n".format(fname, np.mean(avg_A), np.mean(avg_D), np.std(avg_A), np.std(avg_D)))  

        
for file in glob.glob("*.txt"):
    extract_simulation(file)
result.close()