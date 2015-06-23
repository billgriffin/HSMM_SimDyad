import os, sys, re
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from pymattutil.stats import flattendata
from pymattutil.plot import project_data, plot_gaussian_projection, pca
import scipy.special as special
import scipy.stats as stats
import cPickle as pickle

if len(sys.argv) < 4:
    print "usage: python GenHSMMGraphStates.py configure.txt result.txt 1"
    exit
    '''python GenHSMMGraphStatesDataGet.py conf/paramterX.conf results/correspond_result.txt couple_index

   Ex: generate the result graph of first couple:
   python GenHSMMGraphsStatesDataGet.py conf/parameter11.conf results5/Summary/All_Cluster1_11_10000_4.0_8.0_20_60.txt 2
   '''

conf_path = sys.argv[1]
result_path = sys.argv[2]
select_couple = int(sys.argv[3]) - 1  #internal index begins with 0

# read configuration file
# get number of couples and raw data files
f = open(conf_path)
f.readline()
f.readline()
data_files = eval(f.readline().strip())
num_couples = len(data_files)
f.close()

# sigma not used
sigma = np.array([[4.71, 1.75], [1.75, 3.93]])
vecs = np.array([[1., 0.],[0.,1.]])
number_of_states = 0

# read states_colors and used_states
lmbda_dict = {}
state_colors = {}
f = open(result_path)
line = f.readline()
startRead = False
startSystem = False
while len(line) > 0:
    if not startSystem and line.strip().startswith("Group Level"):
        startSystem = True
    if startSystem and re.match(r'^[0-9]', line.strip()):  # fix: 0-9
        items = line.strip().split()
        lmbda_dict[ int(items[0]) ] = float(items[1])
        number_of_states += 1 # fixes error, wag
        state = int(items[0])  #state
        D_1_Avg = float(items[2]) #duration given -- not state;should this be 3
        D_2_Avg = float(items[3]) #should this be 4
        state_colors[state] = (D_1_Avg + D_2_Avg )/9 #need more discrimination
    if startSystem and line.strip().startswith("***"):
        startSystem = False

    line = f.readline()

f.close()
used_states = set(state_colors.keys())

T = 0
num_subfig_cols = 1
num_subfig_rows = 4

########
realdata_set = []
for data_file in data_files:
    data_fname = "matAll/c" + str(data_file) + ".txt"
    realdata = np.transpose(np.loadtxt(data_fname, usecols=(0,1), unpack=True))
    realdata_set.append(realdata)

stateseq_array = []
stat_durations_array = []
stateseq_norep_array = []

mu_dict = {}
mu_sig_dict = {}
f = open(result_path)
line = f.readline()
startRead = False
startMu = False
while len(line) > 0:
    if not startMu and line.strip().startswith("Simulated States for couple indexed as %d" % (select_couple)):
        startMu= True
        line = f.readline()

    if startMu and re.match(r'^[1-9]', line.strip()):
        items = line.strip().split()
        mu_dict[ int(items[1]) ] = (float(items[3]), float(items[4]) )
        mu_sig_dict[int(items[1]) ] = (float(items[5]), float(items[7]),\
                                       float(items[7]), float(items[6]) )
    if startMu and line.strip().startswith("Estimated"):
        startMu = False

    if line.startswith("Estimated"):
        startRead = True
        stateseq = []#np.array([14,0,1,7,7])
        stat_durations = []#np.array([1,1,1,2,1])
        stateseq_norep = []#np.array([7, 19,5,19,10,1,12])
    if startRead and re.match(r'^[1-9]', line.strip()):
        items = line.strip().split()
        stateseq_norep.append( int(items[0]) )
        dur = int(float(items[1]))
        stat_durations.append( float(items[1]) )
        for i in range(dur):
            stateseq.append( int(items[0]) )
        T = int(items[2])
    if startRead and line.strip().startswith("trans_distn"):
        startRead = False
        stateseq_array.append( np.array(stateseq) )
        stateseq_norep_array.append( np.array(stateseq_norep) )
        stat_durations_array.append( np.array(stat_durations) )

    line = f.readline()
f.close()

final_states = list(used_states)
get_states =  []
get_means0 =  []
get_means0_round =  []
get_means1 =  []
get_means1_round =  []
np.set_printoptions(precision=2, suppress = True)
for i in range(number_of_states):
    for j in range(len(final_states)):
        if i == final_states[j] and i in mu_dict:
            get_states.append(final_states[j])
            get_means0.append(float(str(mu_dict[i][0])[:4]))
            get_means0_round.append(np.round(float(str(mu_dict[i][0])[:4]))) #need for dict
            get_means1.append(float(str(mu_dict[i][1])[:4]))
            get_means1_round.append(np.round(float(str(mu_dict[i][1])[:4])))
state_means = zip(get_states, get_means0, get_means1)

for idx in range(num_couples):
    if idx != select_couple:
        continue

    stateseq = stateseq_array[idx]
    stateseq_norep = stateseq_norep_array[idx]
    stat_durations = stat_durations_array[idx]


    state_means_dict = zip(get_means0_round, get_means1_round)
    subject_dict = {}
    subject_dict = dict(zip(get_states,state_means_dict ))

    sub_state_seq = []
    sub_state_seq_X =  []
    sub_state_seq_Y =  []
    States = []

    for i in range(len(stateseq)):
        sub_state_seq.append(subject_dict[stateseq[i]])
    for i in range(len(sub_state_seq)):
        sub_state_seq_X.append(sub_state_seq[i][0])
        sub_state_seq_Y.append(sub_state_seq[i][1])
        States.append((sub_state_seq[i][0] + sub_state_seq[i][1])-1)
    States = np.array(States)
    #print States
    ### dump simulated States values into pickle ###
    sim_values = open("simValues.pck", "wb") # write mode
    pickle.dump(States, sim_values)
    sim_values.close()


    ### create and dumpy real valued States
    realdata = realdata_set[idx]
    realStates = ((realdata[:,0] + realdata[:,1])-1)
    #print realStates
    real_values = open("realValues.pck", "wb") # write mode
    pickle.dump(realStates, real_values)
    real_values.close()
