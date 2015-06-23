import scipy
import numpy as np
import fileinput
import  sys
sys.path.append('/home/atwag/Dropbox/Machine_Learning/HMM/pyhsmm/examples/couplesRawtxt/')
import pyhsmm
sys.path.append('./couplesRawtxt/')
path =  './couplesRawtxt/'

id_data, male_data, female_data, seq, state  = [],[],[],[],[]
id_state = []
coupleGrpData = []   

for line in fileinput.input('couples.txt'):
   if not fileinput.isfirstline():        
      line = line.strip()        
      line = line.split(',') 
      id_data.append(int(line[0]))
      seq.append(int(line[1]))
      male_data.append(int(line[2]))
      female_data.append(int(line[3]))
      #id_state.append([int(line[0]),int(line[12])])
fileinput.close()
matHigh = [200,203,206,208,209,211,212,213,217,225]
matMed =  [202,205,207,210,214,215,218,219,220,228]
matLow =  [201,204,221,222,223,224,226,227,229,230]

   
male = []; female = []
for i in range(len(id_data)):
   Cid = 222
   if id_data[i] == Cid:
      male.append( male_data[i] )
      female.append(female_data[i] )
      
f = open('c'+ str(Cid) +'.txt','a')
for i in range(len(male)):
   f.write('%s %s\n' % (male[i], female[i]))
f.close()