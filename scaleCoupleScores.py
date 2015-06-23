import scipy
from scipy import stats
import numpy as np
from numpy import savetxt  #add
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
import cPickle as pickle
import seaborn as sns



np.set_printoptions(precision=4)

""" These are ordered from 200 to 230; also see means_stds.py for 
different formating """
ids = [200,201,202,203,204,205,206,207,
208,209,210,211,212,213,214,215,
217,218,219,220,221,222,223,224,
225,226,227,228,229,230]

cluster =[4,5,4,1,2,2,3,3,2,2,2,2,1,4,2,2,
1,3,2,3,5,3,5,2,1,5,5,3,5,3]

EntropyRate = [1.0138,1.03682,0.84431,1.32977,0.90481,1.36473,\
               1.15746,1.38914,1.3514,1.95135,1.3015,1.90252,1.27632,\
               0.85904,1.14763,1.15041,1.677,0.79842,1.58242,1.24907,\
               1.5168,1.15361,1.29914,1.53865,1.94022,1.44717,1.07476,\
               1.10038,1.67092,1.87015]

mat= [135,56,106,118,88,114,127,104,138,127,95,133,136,120,105,\
     100,137,102,95,97,67,88,64,82,121,77,75,117,44,75]

entropy=[1.833,2.302,1.324,3.041,2.189,2.477,2.683,2.694,\
         2.559,2.55,2.631,2.112,2.767,1.833,2.21,2.455,\
         2.887,2.249,2.697,2.54,1.706,2.329,2.085,2.454,\
         2.985,2.271,1.66,2.698,2.408,2.828]

mean=[2.001,4.978,2.303,5.353,5.106,4.299,6.747,6.192,5.355,\
      4.976,4.035,4.538,3.403,2.154,5.829,5.992,4.579,6.785,\
      5.258,6.958,6.88,7.467,6.836,6.189,5.17,7.3,6.299,6.095,5.251,5.748]

std=[0.93,1.553,0.654,2.282,1.116,1.518,1.997,1.784,1.517,1.452,\
     1.601,1.034,2.206,0.874,1.149,1.34,1.899,2.108,1.676,1.755,\
     0.801,1.588,1.159,1.403,2.083,1.252,0.769,1.718,1.334,1.813]

cDTW=[24,204,99,729,465,259,168,520,204,189,367,193,289,172,125,\
      282,421,163,283,344,331,108,172,348,244,275,991,278,222,280]

SimDist=[21.8,9.2,15.25,153.8,19.4,54.4,62.6,123,95.6,125.2,\
         91,93.6,91.4,21.6,25.4,82.4,150.4,13.6,77.6,88.2,\
         4.4,66.2,7.2,79.8,135.2,46.4,10.8,83.4,9,122.4]

x = EntropyRate
for i in range(len(EntropyRate)):
   EntropyRate_minmax = [(x_i - float(min(x))) / (max(x) - min(x)) for x_i in x]
   
x = mat
for i in range(len(mat)):
   mat_minmax = [(x_i - float(min(x))) / (max(x) - min(x)) for x_i in x]

x = entropy
for i in range(len(entropy)):
   entropy_minmax = [(x_i - float(min(x))) / (max(x) - min(x)) for x_i in x]

x = mean
for i in range(len(mean)):
   mean_minmax = [(x_i - float(min(x))) / (max(x) - min(x)) for x_i in x]
   
x = std
for i in range(len(std)):
   std_minmax = [(x_i - float(min(x))) / (max(x) - min(x)) for x_i in x]
   
x = cDTW
for i in range(len(cDTW)):
   cDTW_minmax = [(x_i - float(min(x))) / (max(x) - min(x)) for x_i in x]

x = SimDist
for i in range(len(SimDist)):
   simdist_minmax = [(x_i - float(min(x))) / (max(x) - min(x)) for x_i in x]

#for i in EntropyRate_minmax: print i
print EntropyRate_minmax 
print mat_minmax
print entropy_minmax
print mean_minmax
print std_minmax
print cDTW_minmax
print simdist_minmax



## Min-Max scaling


#np_minmax = (x_np - x_np.min()) / (x_np.max() - x_np.min())

#print np_minmax
print 
corrRaw = np.column_stack((EntropyRate,mat,entropy,mean,std,cDTW,SimDist))
print np.corrcoef(corrRaw, rowvar=0)

print 
corrScale = np.column_stack((EntropyRate_minmax, mat_minmax, entropy_minmax, \
                             mean_minmax, std_minmax, cDTW_minmax, simdist_minmax))
#print np.corrcoef(corrScale, rowvar=0)

print 
scaledMatrix = np.array(corrScale)

np.set_printoptions(precision=4)
print
''' Order of columns:
Id	Cluster	EntropyRate	mat	entropy	mean	std	cDTW	SImDist'''
complete_set_normed = np.column_stack((ids,cluster,scaledMatrix))

np.savetxt('complete_set_normedCouples.txt',complete_set_normed, delimiter=',')

pk_couples = open('complete_set_normedCouples.pk','wb')

pickle.dump(complete_set_normed, pk_couples)
pk_couples.close()

wanted = [1] 
'''wanted is the column location for selection'''
cluster1_normed = complete_set[np.logical_or.reduce([complete_set[:,1] == x for x in wanted])]
cluster1_normed_averages =  cluster1_normed.mean(axis=0)
cluster1_normed_std = stats.sem(cluster1_normed[:,2:])

wanted = [2]
cluster2_normed = complete_set[np.logical_or.reduce([complete_set[:,1] == x for x in wanted])]
cluster2_normed_averages =  cluster2_normed.mean(axis=0)
cluster2_normed_std = stats.sem(cluster2_normed[:,2:])

wanted = [3]
cluster3_normed = complete_set[np.logical_or.reduce([complete_set[:,1] == x for x in wanted])]
cluster3_normed_averages =  cluster3_normed.mean(axis=0)
cluster3_normed_std = stats.sem(cluster3_normed[:,2:])

wanted = [4]
cluster4_normed = complete_set[np.logical_or.reduce([complete_set[:,1] == x for x in wanted])]
cluster4_normed_averages =  cluster4_normed.mean(axis=0)
cluster4_normed_std = stats.sem(cluster4_normed[:,2:])

wanted = [5]
cluster5_normed = complete_set[np.logical_or.reduce([complete_set[:,1] == x for x in wanted])]
cluster5_normed_averages =  cluster5_normed.mean(axis=0)
cluster5_normed_std = stats.sem(cluster5_normed[:,2:])

print cluster1_normed_averages[2:]
print cluster1_normed_std
entropy_rate1 = cluster1_normed_averages[2]
entropy_rate1_std = cluster1_normed_std[0]
mat1 = cluster1_normed_averages[3]
mat1_std = cluster1_normed_std[1]

print cluster2_normed_averages[2:]
print cluster2_normed_std

print cluster3_normed_averages[2:]
print cluster3_normed_std

print cluster4_normed_averages[2:]
print cluster4_normed_std

print cluster5_normed_averages[2:]
print cluster5_normed_std

print 
plotData = np.vstack((cluster1_normed_averages[2:],
                           cluster2_normed_averages[2:],
                           cluster3_normed_averages[2:],
                           cluster4_normed_averages[2:],
                           cluster5_normed_averages[2:]))
                           
print plotData


import matplotlib.pyplot as plt


''' Order of columns:
Id	Cluster	EntropyRate	mat	entropy	mean	std	cDTW	SImDist'''
#entropyRate = ax.bar(xpoints,plotData[:,0],color = 'black')
#mat = ax.bar(xpoints+width,plotData[:,1], color = 'red')
#entropy = ax.bar(xpoints+width*2,plotData[:,2],color = 'yellow')
#mean = ax.bar(xpoints+width*3,plotData[:,3])
#std = ax.bar(xpoints+width*4,plotData[:,4])
#cDTW  = ax.bar(xpoints+width*5,plotData[:,5])
#SimDist = ax.bar(xpoints+width*6,plotData[:,6])

N = 6
ind = np.arange(N)
width = 0.15
fig, ax = plt.subplots()
#pylab.rcParams['legend.loc'] = 'best'

#entropybar = ax.bar(ind, bar_entropy,width, color = 'r',yerr = bar_entropy_error, ecolor='k' )
entropybar = ax.bar(ind, plotData[:,0].T,width, color = 'r')#,yerr = bar_entropy_error, ecolor='k' )

#matbar = ax.bar(ind, bar_mats,width, color = 'b',yerr = bar_mats_error, ecolor='k')
matbar = ax.bar(ind+width, plotData[:,1].T,width, color = 'b')#,yerr = bar_mats_error, ecolor='k')

#meanbar = ax.bar(ind+width*2, bar_mean,width, color = 'y',yerr = bar_mean_error, ecolor='k' )
#stdbar = ax.bar(ind+width*3, bar_std,width, color = 'g',yerr = bar_std_error, ecolor='k' )
#DTWbar = ax.bar(ind+width*4, bar_DTW,width, color = 'm',yerr = bar_DTW_error, ecolor='k' )
# add some labels2
ax.set_ylabel('Magnitude')
ax.set_title('Clusters by Feature (with Standard Error)')
ax.set_xticks(ind+.35)
ax.set_xticklabels( ('Cluster1', 'Cluster2', 'Cluster3', 'Cluster4', 'Cluster5') )

ax.legend( (matbar[0], entropybar[0],meanbar[0],stdbar[0], DTWbar[0]),
           ('MAT', 'Entropy', 'Affect Mean','Mean Sd','DTW') )
plt.axis('tight')
plt.ylim(0,1.0)

plt.show()

plt.show()
