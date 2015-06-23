import numpy as np
from matplotlib import pyplot as plt
import  sys, math
sys.path.append('/Users/atwag/Dropbox/Machine_Learning/HMM/') #linux
sys.path.append('/Users/Griffin_vaio/Dropbox/Machine_Learning/HMM/') #laptop
sys.path.append('/Users/wmgriffin/Dropbox/Machine_Learning/HMM/') #mac
import pyhsmm
from pyhsmm.util.plot import pca_project_data, plot_simulation
import time
#sys.path.append('./couplesRawtxt/')
from get_couple_data import load_high, load_high_1, load_high_2, load_med,  load_low
path =  './couplesRawtxt/'
start = time.time()
print "it took", time.time() - start, "seconds."
#import  nicetext

# suppress  warnings
np.seterr(divide='ignore')

#matHighG1 = [200*,203*,206*,208*,209,211*,212*,213*,217,225] # * indicates grp 1
#matHighG2 = [200,203,206,208,209*,211,212,213,217*,225*] # * indicates grp 2

#matMed =  [202,205,207,210,214,215,218,219,220,228]
#matLow =  [201,204,221,222,223,224,226,227,229,230]

'''sorted; high to low within group'''
matHighGp1 = [208, 212, 200, 211, 206, 213, 203] 
matHighGp2 = [217, 209, 225] 


#matMed =  [228, 205, 202, 214, 207, 218,215, 220, 210, 219]
#matLow =  [204, 222, 224, 226,227, 230, 221, 223, 201, 229]

#mat_hi_lo = [208, 217, 212, 200, 211, 206, 209, 225,
            #213, 203, 228, 205, 202, 214, 207, 218,
            #215, 220, 210, 219, 204, 222, 224, 226,
            #227, 230, 221, 223, 201, 229]

#h =  load_high()
#h_1 = load_high_1()
h_2 = load_high_2()

#m =  load_med()
#l =  load_low()

datasets =  []

### high, med, low selection ###
#for i in range(len(h)):
    #datasets.append(np.transpose(np.loadtxt(path +h[i],  usecols=(0, 1),  unpack = True)))    
#for i in range(len(h_1)):
    #datasets.append(np.transpose(np.loadtxt(path +h_1[i],  usecols=(0, 1),  unpack = True)))   
for i in range(len(h_2)):
    datasets.append(np.transpose(np.loadtxt(h_2[i],  usecols=(0, 1),  unpack = True)))    
    
#for i in range(len(m)):
    #datasets.append(np.transpose(np.loadtxt(path +m[i],  usecols=(0, 1),  unpack = True)))    
#for i in range(len(l)):
    #datasets.append(np.transpose(np.loadtxt(path +l[i],  usecols=(0, 1),  unpack = True)))
    
### which couple within the set ###   sorted
data0 = datasets[0] 
data1 = datasets[1]
data2 = datasets[2] 

sub_state_seq_X =  []
sub_state_seq_Y =  []

print 
#for i in range(len(s.stateseq)):
    #sub_state_seq.append(subject_dict[s.stateseq[i]])
for i in range(len(data2)):    
    sub_state_seq_X.append(data2[i][0])
    sub_state_seq_Y.append(data2[i][1])           

plt.plot(sub_state_seq_X, label ='D1', color='b' )
plt.plot(sub_state_seq_Y, label = 'D2', color='r')
plt.title('Sequence of Affect States: Couple 225')
plt.legend()
plt.xlim((0, 720))
plt.ylim((0, 8))    
plt.show()