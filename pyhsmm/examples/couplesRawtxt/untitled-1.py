
import random,sys
import numpy as np
from random import shuffle
import scipy
from scipy import stats
import fileinput
import matplotlib
import pylab
from mpl_toolkits.mplot3d.axes3d import Axes3D

####################################

def gibbsMove(L):
   L = [t[0] for t in L]
   S = sum(L)
   cumulative = 0
   # weight by score
   r = random.random()
   for i,s in enumerate(L):
      cumulative += s
      f = cumulative*1.0/S
      if f > r:  return i
   raise ValueError('f < r')

###################################

##########################################################
if __name__=="__main__":   

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

   def get_mat():
      mat_data = \
         '''
           200,128,142,135,2,200,H
           201,35,78,56,1,201,L
           202,111,101,106,2,202,M
           203,117,120,118,2,203,H
           204,86,91,88,1,204,L
           205,114,115,114,2,205,M
           206,121,133,127,2,206,H
           207,107,101,104,2,207,M
           208,144,132,138,2,208,H
           209,125,129,127,2,209,H
           210,111,79,95,1,210,M
           211,142,124,133,2,211,H
           212,135,137,136,2,212,H
           213,110,130,120,2,213,H
           214,99,111,105,2,214,M
           215,68,132,100,2,215,M
           217,134,140,137,2,217,H
           218,113,92,102,2,218,M
           219,93,98,95,1,219,M
           220,102,93,97,1,220,M
           221,63,72,67,1,221,L
           222,79,98,88,1,222,L
           223,75,54,64,1,223,L
           224,97,68,82,1,224,L
           225,134,108,121,2,225,H
           226,68,86,77,1,226,L
           227,90,61,75,1,227,L
           228,119,116,117,2,228,M
           229,45,43,44,1,229,L
           230,67,83,75,1,230,L'''

      lines = mat_data.split('\n')
      for line in lines:
         if len(line.strip()) == 0:
            continue
         items = line.split(',')
         c_id               = int(items[0])
         m_mat              = float(items[1])
         f_mat              = float(items[2])
         c_mat_ratio        = float(m_mat/f_mat) #make ln
         c_mat_level        = str(items[6])

         if not c_id in couple_mat: 
            couple_mat[c_id] = {} 
         couple_mat[c_id] = (m_mat,f_mat,c_mat_ratio,c_mat_level)
      return couple_mat

   male = []; female = []

   coupleID = 200; sex = female

   for i in range(len(id_data)):
      Cid = coupleID
      if id_data[i] == Cid:
         male.append( male_data[i] )
         female.append(female_data[i] )



   data = np.array(sex)
   ###########################################################
   male = np.array(male)
   female = np.array(female)
##---------------------------------
## a discrete distribution
#R = range(10)
#distr = [
      #[1,1,2,2,1,1,1,1,1,1],
      #[1,3,4,5,2,1,1,1,1,1],
      #[1,4,7,6,2,1,1,1,1,1],
      #[1,3,4,3,2,1,1,1,1,1],
      #[1,1,1,1,1,1,1,1,1,1],
      #[1,1,1,1,1,1,1,1,1,1],
      #[1,1,1,1,1,2,2,3,2,1],
      #[1,1,1,1,1,2,3,4,2,1],
      #[1,1,1,1,1,2,2,5,3,1],
      #[1,1,1,1,1,1,1,1,2,1]]

#def f(x,y):
   #return distr[x][y]
##---------------------------------
## save distribution for R plot
#FH = open('results.txt','w')
#for x in R:
   #for y in R:
      ## convert to 1-based index
      #t = (x+1,y+1,f(x,y))
      #L = [str(e) for e in t]
      #FH.write('  '.join(L) + '\n')
#FH.close()
######################################
#The R code to do the plots:

#Rcode
#setwd('Desktop')
#library(scatterplot3d)
#m = read.table('results.txt',as.is=T)
#par(las=1)
#scatterplot3d(m,
   #type='h',pch=16,
   #cex.symbol=3,
   #cex.axis=2,
   #highlight.3d = T,
   #xlab='x',ylab='y',
   #zlab='z')

#s = m[,3]
#dim(s) = c(10,10)
#s = t(s)
#image(s,
   #col=topo.colors(10))



   x = female
   y = male
   L = [[f(x,y),x,y]]
   t = ('x','y')
   
   T = int(1E5)
   for i in range(T):
      L2 = list()
      if random.choice(t) == 'x':
         y = L[-1][2]
         for x in R:  L2.append((f(x,y),x,y))
      else:
         x = L[-1][1]
         for y in R:  L2.append((f(x,y),x,y))
   
      j = gibbsMove(L2)
      L.append(L2[j])
   #---------------------------------
   # show results
   D = dict()
   for t in L:
      t = tuple(t)
      if D.has_key(t):  D[t] += 1
      else:  D[t] = 1
   
   kL = sorted(D.keys(),key=lambda t:  t[0])
   kL.reverse()
   
   def show(t):
      counts = D[t]
      score = t[0]
      print str(counts).rjust(4),
      ratio = 1.0*counts/score
      print str(round(ratio,2)).ljust(7),
      print '(' + str(round(score,1)).rjust(4),
      print ',',t[1],',',t[2],')'
   
   print 'c = count, s = score'
   print '  c   c/s     ( s,  x,  y)'
   for t in kL[:20]:  show(t)
   print '-' * 10
   for t in kL[-20:]: show(t)