from __future__ import division
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
from matplotlib import pyplot as plt
import copy

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'HSMM_SimDyad_New'))

import pyhsmm
#pyhsmm.internals.states.use_eigen()
from pyhsmm.util.text import progprint_xrange
#from pyhsmm.util.text import progprint_xrange

SAVE_FIGURES = False

print \
'''
This demo shows the HDP-HSMM in action. Its iterations are slower than those for
the (Sticky-)HDP-HMM, but explicit duration modeling can be a big advantage for
conditioning the prior or for discovering structure in data.
'''
#####################
#  read real data #
#####################
import sys
path = './couplesRawtxt/'
sys.path.append(path)
from get_couple_data import load_high,  load_med,  load_low
matHigh = [200,203,206,208,209,211,212,213,217,225]
matMed =  [202,205,207,210,214,215,218,219,220,228]
matLow =  [201,204,221,222,223,224,226,227,229,230]
h =  load_high()
m =  load_med()
l =  load_low()

datasets =  []
for i in range(len(m)):
    datasets.append(np.transpose(np.loadtxt(path +m[i],  usecols=(0, 1),  unpack = True)))    
    
### which couple within the set ###   
data0 = datasets[0]
data1 = datasets[1]
data2 = datasets[2]
#data3 = datasets[3]
#data4 = datasets[4]
#data5 = datasets[5]
#data6 = datasets[6]
#data7 = datasets[7]
#data8 = datasets[8]
#data9 = datasets[9]    

obs_dim = 2

obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'sigma_0':np.eye(obs_dim),
                'kappa_0':0.3,
                'nu_0':obs_dim+5}

dur_hypparams = {'alpha_0':2*30, # xun: used to be k:8  and theta:5
                 'beta_0':2}

#########################
#  posterior inference  #
#########################

# Set the weak limit truncation level
Nmax = 25
T = 500

obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

posteriormodel = pyhsmm.models.HSMM(
        alpha=6.,gamma=6., # these can matter; better to sample over them (concentration-resampling.py)
        init_state_concentration=6., # pretty inconsequential
        obs_distns=obs_distns,
        dur_distns=dur_distns,
        trunc=90) # duration truncation speeds things up when it's possible 
        # xun: trunc was 60 in demo
        
posteriormodel.add_data(data0)
posteriormodel.add_data(data1)
posteriormodel.add_data(data2)

models = []
for idx in progprint_xrange(11): # xun: we use 31 instead of 150
    posteriormodel.resample_model()
    if (idx+1) % 10 == 0:
        models.append(copy.deepcopy(posteriormodel))

fig = plt.figure()
for idx, model in enumerate(models):
    plt.clf()
    model.plot()
    plt.gcf().suptitle('HDP-HSMM sampled after %d iterations' % (10*(idx+1)))
    if SAVE_FIGURES:
        plt.savefig('iter_%.3d.png' % (10*(idx+1)))

plt.show()
