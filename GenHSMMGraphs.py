import os, sys, re
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from pymattutil.stats import flattendata
from pymattutil.plot import project_data, plot_gaussian_projection, pca
import scipy.special as special
import scipy.stats as stats

if len(sys.argv) < 4:
    print "usage: python GenHSMMGraph.py configure.txt result.txt 1"
    exit
    '''python GenHSMMGraph.py conf/paramterX.conf results/correspond_result.txt couple_index

   Ex: generate the result graph of first couple:
   python GenHSMMGraph.py conf/parameter0.conf results/All_All_10000_4.0_4.0_30_50.txt 1
   or
   conf/parameter16_allHigh50.conf results/High_All_10000_8.0_8.0_40_50.txt 1'''

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
        D_1_Avg = float(items[2]) #duration given -- not state;should this be 3 xun: see line 55 this is state
        D_2_Avg = float(items[3]) #should this be 4 xun this is state not duration
        state_colors[state] = (D_1_Avg + D_2_Avg - 2)/8 #need more discrimination
    if startSystem and line.strip().startswith("***"):
        startSystem = False

    """
    if not startRead and line.strip().startswith("%s Individual states" % num_couples):
        startRead = True
        line = f.readline()
    if startRead and re.match(r'^[1-9]', line.strip()):
        items = line.strip().split()
        state = int(items[1])  #state
        D_1_Avg = float(items[3]) #duration given -- not state;should this be 3
        D_2_Avg = float(items[4]) #should this be 4
        state_colors[state] = (D_1_Avg + D_2_Avg )/16 #need more discrimination
    if startRead and line.strip().startswith("Estimated"):
        startRead = False
    """
    line = f.readline()

f.close()
used_states = set(state_colors.keys())

T = 0
num_subfig_cols = 1
num_subfig_rows = 4

########
current_dir = os.path.dirname(os.path.realpath(__file__))
realdata_set = []
for data_file in data_files:
    #data_fname = "matAll/c" + str(data_file) + ".txt"
    data_fname = os.path.join(current_dir, 'matAll', "c%s.txt"%str(data_file))
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

########
## plot every couple
#########
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

fig = plt.figure(figsize=(10, 12))
fig.subplots_adjust(hspace=0.5)#, wspace=0.45)
#fig = plt.gcf()
cmap = cm.get_cmap('RdYlBu_r')
def label(xy, text):
    y = xy[1] - 0.15 # shift y-value for label so that it's below the artist
    plt.text(xy[0], y, text, ha="center", family='sans-serif', size=8)

for idx in range(num_couples):
    if idx != select_couple:
        continue

    # legend: 1x1 1x2 1x3 1x4 1x5 25
    ax0 = plt.subplot2grid((9,2),(0,0),colspan=2)

    """
    # create 15x2 grid to plot the artists
    grid = np.mgrid[0.1:0.9:15j, 0.2:0.6:2j].reshape(2, -1).T
    w = 0.03
    h = 0.1
    offset = [0.015, 0.05]

    # first row
    lbls = [(5,5),(5,4),(5,3),(5,2),(5,1),
            (4,5),(4,4),(4,3),(4,2),(4,1),
            (3,5),(3,4),(3,3),(3,2),(3,1)]
    for i in range(15):
        ii = i*2 + 1
        color = cmap(sum(lbls[i])/10.0)
        rect = mpatches.Rectangle(grid[ii]-offset, w, h, ec="none",color=color)
        label(grid[ii], str(lbls[i]))
        ax0.add_patch(rect)

    # second row
    lbls = [(2,5),(2,4),(2,3),(2,2),(2,1),
            (1,5),(1,4),(1,3),(1,2),(1,1)]
    for i in range(10):
        ii = i*2
        color = cmap(sum(lbls[i])/10.0)
        rect = mpatches.Rectangle(grid[ii]-offset, w, h, ec="none", color=color)
        label(grid[ii], str(lbls[i]))
        ax0.add_patch(rect)

    #collection = PatchCollection(patches, alpha=0.3)

    plt.axis('off')
    plt.xlim ((0.05, 0.95))
    plt.ylim ((0, 0.7))
    """
    z = np.array([[6,2+5,3+5,4+5,5+5],
                  [5,2+4,3+4,4+4,5+4],
                  [4,5,6,7,8],
                  [3,4,5,6,7],
                  [2,3,4,5,6],
                  ])
    plt.imshow(z, cmap="RdYlBu_r", extent=[0,4,0,4])
    x = [0,1,2,3,4]
    plt.xticks(x,['1','2','3','4','5'])
    y = [0,1,2,3,4]
    plt.yticks(y,['1','2','3','4','5'])
    #plt.axis('off')
    ax0.set_ylabel('Male')
    ax0.set_xlabel('Female')
    plt.title('Legend')

    #--------------------------------------------------
    # State scatter
    ax1 = plt.subplot2grid((9,2),(1,0),rowspan=2)

    stateseq = stateseq_array[idx]
    stateseq_norep = stateseq_norep_array[idx]
    stat_durations = stat_durations_array[idx]

    for state in range(number_of_states):
        if state in stateseq_norep:
            obs_color = cmap(state_colors[state])
            obs_data = realdata_set[idx][ stateseq[:len(realdata_set[idx])] == state ]
            obs_data = flattendata(obs_data)
            projected_data = project_data(obs_data,vecs)
            colors = [cmap((sum(i) - 2)/8) for i in projected_data]
            plt.scatter(projected_data[:,0],projected_data[:,1], s=20, c=colors,edgecolors='none')
            sigmaState = np.array([mu_sig_dict[state][:2],mu_sig_dict[state][2:]])
            
            gaussian_clr = cmap((sum(mu_dict[state])-2)/8.0)
            plot_gaussian_projection(mu_dict[state],sigmaState,vecs,color=gaussian_clr,label='')
            #plot_gaussian_projection(mu_dict[state],sigma,vecs,color=obs_color,label='')

    #frame  = ax1.get_frame()
    #frame.set_facecolor('0.90')
    ax1.set_ylabel('Male')
    ax1.set_xlabel('Female')
    x = [0,1,2,3,4,5,6]
    plt.xlim ((0, 6))
    plt.xticks(x,['','1','2','3','4','5',''])
    plt.ylim ((0, 6))
    plt.yticks(x,['','1','2','3','4','5',''])
    plt.title('Observation Distributions')

    #-------------------------------------------------
    # Duration Distribution
    ax2 = plt.subplot2grid((9,2),(1,1),rowspan=2)
    for state in range(number_of_states):
        if state in stateseq_norep:
            #color = cmap(state_colors[state])
            color = cmap((sum(mu_dict[state])-2)/8.0)
            dur_data = stat_durations[ stateseq_norep == state ]
            dur_data = flattendata(dur_data)
            lmbda = lmbda_dict[state]

            x = np.arange(1,1000)
            try:
                tmax = np.where(np.exp(stats.poisson.logsf(x-1, lmbda)) < 1e-3)[0][0]
            except:
                #tmax = self.rvs(1000).max()
                print "error"
            tmax = max(tmax, dur_data.max()) if dur_data is not None else tmax
            t = np.arange(1,tmax)

            x = t-1
            x = np.array(x,ndmin=1)
            raw = np.empty(x.shape)
            raw[x>=0] = -lmbda + x[x>=0]*np.log(lmbda) - special.gammaln(x[x>=0]+1)
            raw[x<0] = -np.inf

            if isinstance(x, np.ndarray):
                x = raw
            else:
                x = raw[0]
            x = np.exp(x)
            plt.ylim(ymax = .5, ymin = .0)
            plt.plot(t,x,color=color)
            #plt.plot(t,np.exp(self.log_likelihood(t-1)),color=color)
            #try:
                #plt.ylim(ymax = 1.0, ymin = .0)
                #'''Xun - this should be working but it is not; some stacks are hidden'''
                #plt.hist(dur_data,bins=t-0.5, histtype='bar', stacked=True,fill=True,color=color,normed=True)
            #except:
                #pass
            #if state >20:
                #break


    llines = ax2.get_lines()
    plt.setp(llines, linewidth=2.5)
    #frame  = ax2.get_frame()
    #frame.set_facecolor('0.90')
    plt.xlim((0,stat_durations.max()*1.1))
    plt.title('Duration Distributions')
    #ax2.set_ylabel('Percentage')
    ax2.set_xlabel('Duration')

    #-------------------------------------------------
    # State Sequence
    ax3 = plt.subplot2grid((9,2),(3,0),colspan=2,rowspan=2)
    #ax3 = plt.subplot(5,1,3)
    #s.plot(colors_dict=state_colors,cmap=cmap)
    X,Y = np.meshgrid(np.hstack((0,stat_durations.cumsum())),(0,1))
    C = np.array([[state_colors[state] for state in stateseq_norep]])
    plt.pcolor(X,Y,C,vmin=0,vmax=1, cmap=cmap)
    plt.ylim((0,1))
    plt.xlim((0,700))
    plt.yticks([])
    plt.title('State Sequence')
    ax3.set_ylabel('State')
    ax3.set_xlabel('Time; Width Reflects Duration')

    #-------------------------------------------------
    # sudo trace
    colors = ["#348ABD", "#A60628", "#7A68A6", "#467821", "#CF4457"]

    ax4 = plt.subplot2grid((9,2),(5,0),colspan=2,rowspan=2)
    #ax4 = plt.subplot(5,1,4)

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
    #plt.plot(sub_state_seq_X, label ='D1', color='b' )
    #plt.plot(sub_state_seq_Y, label = 'D2', color='r')
    plt.plot(sub_state_seq_X,  color="#348ABD", lw=1.8)#, label='Male', )
    plt.plot(sub_state_seq_Y,  color="#A60628", lw=1.8)#, label='Female',)
    plt.title('Simulated Male-Female Affect Trace')
    ax4.set_ylabel('Affect (0 = +);\nMale = Blue')#,color = 'b')
    ax4.set_xlabel('Time')
    plt.legend()
    plt.xlim((0, 700))
    plt.ylim((0, 6))
    y = [0,1,2,3,4,5,6]
    plt.yticks(y,['','1','2','3','4','5',''])

    #-------------------------------------------------
    # original data
    ax5 = plt.subplot2grid((9,2),(7,0),colspan=2,rowspan=2)
    #ax5 = plt.subplot(5,1,5)
    realdata = realdata_set[idx]
    plt.plot(realdata[:,0],  color="#348ABD", lw=1.8)#label ='Male',
    plt.plot(realdata[:,1],  color="#A60628", lw=1.8)#label = 'Female',
    #plt.plot(realdata[:,0],  color='b')#, label='Male',)
    #plt.plot(realdata[:,1],  color='r')#label='Female',)
    plt.title('Original Male-Female Affect Trace')
    ax5.set_ylabel('Affect (0 = +);\nMale = Blue')
    ax5.set_xlabel('Time')
    plt.legend()
    plt.xlim((0, 700))
    plt.ylim((0, 6))
    y = [0,1,2,3,4,5,6]
    plt.yticks(y,['','1','2','3','4','5',''])
    #plt.tight_layout(pad=0.1, w_pad=0.5, h_pad=0.5)
    plt.tight_layout()
    
    
    plt.show()
    #plt.savefig("couple%d.tiff"%idx)
