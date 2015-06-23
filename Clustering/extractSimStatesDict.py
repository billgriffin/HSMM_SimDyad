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

file_dict["Ids"] = {"Ids":[200, 201, 202, 203, 204, 205, 206, 207, 208,\
      209, 210, 211, 212, 213, 214, 215, 217, 218, 219,\
      220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230]}

Ids =[200, 201, 202, 203, 204, 205, 206, 207, 208,\
    209, 210, 211, 212, 213, 214, 215, 217, 218, 219,\
    220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230]

'''These are ordered by MAT Category level -- not raw score'''
file_dict["All"] = {"All": [200,203,206,208,209,211,212,213,217,225, \
                            202,205,207,210,214,215,218,219,220,228, \
                            201,204,221,222,223,224,226,227,229,230]}

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

# values from original couple data
def compute_realData():
    real_couple_dict = {}
    os.chdir("./matAll")
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
        couple_pair = zip(male,female)
        #dyadicstate = np.array(dyadicstate)
        #### normalize the data
        #dyadicstate_scaled = preprocessing.scale(dyadicstate)         
        real_couple_dict[file_id] = (couple_pair)
    os.chdir("../")
    #if real_couple_dict.keys == 200:
    #for k, v in real_couple_dict.iteritems():        
        #for i in range(len(v)):
            #print k,v[i][0],v[i][1] 
    
    for j in range(30):
        x = 1
        print 'couple: %s'%(Ids[j])
        for i in real_couple_dict[Ids[j]]: 
            #print x,i
            print (str(i[0]) + str(i[1]))
            #print x ,(str(i[0]) + str(i[1]))
            x += 1                
    #print real_couple_dict
    return real_couple_dict

def extract_simulation(fpath):    
    fname = fpath.split("/")[-1]
    keys = fname.split("_")
    cat_key = keys[0]
    group_key = keys[1]
    couple_ids = file_dict[cat_key][group_key]
    #result = open("wavelets_result_%sstats_%strunk.csv"%(keys[-2],keys[-1]),"w")
    
    o = open(fpath)
    line = o.readline()
    start_process = False
    c_idx = 0
    write = False
    sim_couple_dict = {}
    #summary of individual couple
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
            sim_dict_round ={}
            while not line.startswith("--"):
                line = line.strip()
                row = line.split()
                items = []
                for item in row:
                    if item != " ":
                            items.append(item)
                # create dict
                sim_dict[ items[1] ] = (float(items[3]), float(items[4]))
                sim_dict_round[ items[1] ] = (round(float(items[3])), round(float(items[4])))
                line = o.readline()
            ## simulation data
            male = []
            female = []
            line = o.readline()
            # skip the lines until male/female sequence data
            while not line.startswith("--"):
                line = o.readline()
            line = o.readline()            
            while not line.startswith("--"):
                line = line.strip()
                row = line.split()
                items = []
                for item in row:
                    if item != " ":
                            items.append(item)
                state = items[0] #the first item is state, pick up key 
                x,y = sim_dict[state]
                dur = int(float(items[1])) #pick up duration
                for i in range(dur):
                    male.append( int(round(x)) )
                    female.append( int(round (y)) )
                    #print c_idx, male, female
                line = o.readline()
            ## start using wavlet on male[] and female[]
            #dyadicstate = []
            couple_pair = zip(male,female)
            #sim_couple_dict[c_idx] = (male,female)
            sim_couple_dict[c_idx] = (couple_pair)
            
            
            #for i in range(n):
                #dyadicstate.append( (male[i] + female[i])/2.0 )
            #dyadicstate = np.array(dyadicstate)
            ## normalize the data
            #dyadicstate_scaled = preprocessing.scale(dyadicstate) 
                        
            ## compare
            #couple_id = couple_ids[c_idx]
            #cA_real, cD_real = real_wavelet_dict[couple_id] 
            #distA = .0
            #distD = .0
            #try:
                #distA = np.linalg.norm(cA_sim[:len(cA_real)] - cA_real)
                #distD = np.linalg.norm(cD_sim[:len(cA_real)] - cD_real)
            #except:
                #distA = np.linalg.norm(cA_sim - cA_real[:len(cA_sim)])
                #distD = np.linalg.norm(cD_sim - cD_real[:len(cD_sim)])
                
            #result.write("{0},{1},{2},{3}\n".format(fname, couple_id,  distA, distD)  )
            #avg_A.append(distA)
            #avg_D.append(distD)
                         
            c_idx += 1
            start_process = False
            
        line = o.readline()
    
    #for k, v in sim_dict_round.iteritems():
        #print k,v
    #print
    
    coupleSimRatings = []
    #if sim_couple_dict.keys == 0:
    #for k, v in sim_couple_dict.iteritems():
        #for i in range(len(v)):
            #print k,v[i][0],v[i][1] 
            #coupleSimRatings.append((k,v[i][0],v[i][1]))
            
    print  
    
    for j in range(30):
        x = 1
        print 'couple: %s'%(j)
        for i in sim_couple_dict[j]: 
            #print x, i
            print (str(i[0]) + str(i[1]))
            #print x ,(str(i[0]) + str(i[1]))
            x += 1
                
    #print coupleSimRatings
    #for k, v in sim_couple_dict.iteritems():
                #print k,v 
    
    #if len(avg_A) > 0:
        #result.write("{0},{1},{2},{3},{4}\n".format("Average:", np.mean(avg_A), np.mean(avg_D), np.std(avg_A), np.std(avg_D)))  
    #result.close()
    return sim_couple_dict


#compute_realData()    
#extract_simulation("results/All_All_10000_4.0_4.0_10_50.txt")
#extract_simulation("results/All_All_10000_4.0_4.0_20_50.txt")
#extract_simulation("results/All_All_10000_4.0_4.0_30_50.txt")
extract_simulation("results/All_All_10000_4.0_4.0_40_50.txt")

