import numpy as np
ids = [200, 201, 202, 203, 204, 205, 206, 207, 208,\
      209, 210, 211, 212, 213, 214, 215, 217, 218, 219,\
      220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230]

mat_hi_loCAT = [2081, 2171, 2121, 2001, 2111, 2061, 2091, 2251,2131, 2031,\
             2282, 2052, 2022, 2142, 2072, 2182, 2152, 2202, 2102, 2192,\
             2043, 2223, 2243, 2263, 2273, 2303, 2213, 2233, 2013, 2293]

def ReadStatesDict(o):
    state_dict_book = []
    start_process = False
    line = o.readline()
    while len(line) > 0:
        line = line.strip()
        if line.startswith("System states"):
            while not line.startswith("--"):
                line = o.readline()  # move to detail
            line = o.readline()
            start_process = True
            
        if start_process:
            while not line.startswith("--"):
                line = line.strip()
                row = line.split()
                state = int(row[0])
                state_dict_book.append(state)
                line = o.readline()
            start_process = False
            return state_dict_book
        line = o.readline()

def ReadStatesMatrix(o):
    ind_state_dict = {}
    start_process = False
    line = o.readline()
    while len(line) > 0:
        line = line.strip()
        if line.startswith("State Duration Cumulative"):
            line = o.readline()  # move to detail
            line = o.readline()  # move to detail
            start_process = True
            
        if start_process:
            while not line.startswith("--"):
                line = line.strip()
                row = line.split()
                state = int(row[0])
                dur = int(float(row[1]))
                if state in ind_state_dict:
                    ind_state_dict[state] += dur 
                else:
                    ind_state_dict[state] = dur
                line = o.readline()
            start_process = False
            return ind_state_dict
        line = o.readline()
    
num_couples = 30
n_states = 30    
result = open("results/All_All_10000_4.0_4.0_30_50.txt")

data = np.zeros([num_couples,n_states])
state_book = ReadStatesDict(result)
couple_dict_list = []
for i in range(num_couples):
    couple_dict = ReadStatesMatrix(result)
    for s,c in enumerate(couple_dict):
        data[i][s] = c
        #print i, s
    couple_dict_list.append(couple_dict) 
result.close()

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(data)



from pylab import *
figure(1)
title("data")
imshow(data, cmap="PuRd",interpolation='nearest')
cbar = colorbar()
cbar.solids.set_edgecolor("face")
yticks( arange(30))
ylabel("couple")
xticks( arange(n_states))
xlabel("state")

figure(2)
title("Tf-idf")
imshow(tfidf.toarray(), cmap="PuRd",interpolation='nearest')
cbar = colorbar()
cbar.solids.set_edgecolor("face")
yticks( arange(30))
ylabel("couple")
xticks( arange(n_states))
xlabel("state")
### k means ### to cluster
from sklearn.cluster import KMeans, MiniBatchKMeans
k = 3
km = KMeans(k)
km.fit(tfidf)
clusters = km.labels_

couple_id = []
sorted_data = []
sorted_tfidf = []
for i in range(k):
    for idx in np.where(clusters==i)[0]:
        sorted_data.append(list(data[idx]))
        sorted_tfidf.append(list(tfidf.toarray()[idx]))
        couple_id.append(idx)

figLabels = []       
#for i in range(len(couple_id)): 
    #figLabels.append(ids[couple_id[i]])    
for i in range(len(couple_id)): 
    figLabels.append(mat_hi_loCAT[couple_id[i]])  
    
for i in figLabels: print i

figure(3)
title("sorted data")
imshow(sorted_data, cmap="PuRd", interpolation='nearest')
cbar = colorbar()
cbar.solids.set_edgecolor("face")
yticks( range(30),couple_id)
ylabel("couple")
xticks( arange(n_states))
xlabel("state")

figure(4)
title("sorted Tf-idf")
imshow(sorted_tfidf,cmap="PuRd", interpolation='nearest')
cbar = colorbar()
cbar.solids.set_edgecolor("face")
yticks( range(30),figLabels)
#yticks( range(30),couple_id)
ylabel("couple")
xticks( arange(n_states))
xlabel("state")

show()