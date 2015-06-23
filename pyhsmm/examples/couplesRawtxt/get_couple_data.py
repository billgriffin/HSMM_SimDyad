from scipy import *
import numpy as np
import fileinput
import math
from sets import Set

#matHigh = [200,203,206,208,209,211,212,213,217,225]
#matMed =  [202,205,207,210,214,215,218,219,220,228]
#matLow =  [201,204,221,222,223,224,226,227,229,230]

'''sorted; high to low within group'''
matHigh  =   [208, 217, 212, 200, 211, 206, 209, 225,213, 203]
matHighGp1 = [208, 212, 200, 211, 206, 213, 203] 
matHighGp2 = [217, 209, 225] 



matMed  =   [228, 205, 202, 214, 207, 218,215, 220, 210, 219]
matMedGp1 = [228, 202, 214, 218, 215, 220]
matMedGp2 = [205, 207, 210, 219]



matLow =   [204, 222, 224, 226,227, 230 , 221, 223, 201, 229 ] #sorted
matLowGp1 = [204, 222, 226, 227, 223, 201]
matLowGp2 = [224, 230, 221, 229]


matAll =  matHigh + matMed + matLow

mat_hi_lo = [208, 217, 212, 200, 211, 206, 209, 225,
         213, 203, 228, 205, 202, 214, 207, 218,
         215, 220, 210, 219, 204, 222, 224, 226,
         227, 230, 221, 223, 201, 229]

#### where is 216 -not avaible; this is correct
dataset_high = []
dataset_high_1 = []
dataset_high_2 = []

dataset_med =  []
dataset_med_1 =  []
dataset_med_2 =  []

dataset_low =  []
dataset_low_1 =  []
dataset_low_2 =  []



dataset_all = []
dataset_all_sort = []

def load_high():
   x =  'c'; y =  '.txt'
   for i in range(len(matHigh)):
      dataset_high.append(x + str(matHigh[i]) + y)
   return dataset_high

def load_high_1():
   x =  'c'; y =  '.txt'
   for i in range(len(matHighGp1)):
      dataset_high_1.append(x + str(matHighGp1[i]) + y)
   return dataset_high_1

def load_high_2():
   x =  'c'; y =  '.txt'
   for i in range(len(matHighGp2)):
      dataset_high_2.append(x + str(matHighGp2[i]) + y)
   return dataset_high_2
#########################################################

def load_med():
   x =  'c'; y =  '.txt'
   for i in range(len(matMed)):
      dataset_med.append(x + str(matMed[i]) + y)
   return dataset_med

def load_med_1():
   x =  'c'; y =  '.txt'
   for i in range(len(matMedGp1)):
      dataset_med_1.append(x + str(matMedGp1[i]) + y)
   return dataset_med_1

def load_med_2():
   x =  'c'; y =  '.txt'
   for i in range(len(matMedGp2)):
      dataset_med_2.append(x + str(matMedGp2[i]) + y)
   return dataset_med_2  
#########################################################
def load_low():
   x =  'c'; y =  '.txt'
   for i in range(len(matLow)):
      dataset_low.append(x + str(matLow[i]) + y)
   return dataset_low

def load_low_1():
   x =  'c'; y =  '.txt'
   for i in range(len(matLowGp1)):
      dataset_low_1.append(x + str(matLowGp1[i]) + y)
   return dataset_low_1

def load_low_2():
   x =  'c'; y =  '.txt'
   for i in range(len(matLowGp2)):
      dataset_low_2.append(x + str(matLowGp2[i]) + y)
   return dataset_low_2

#########################################################
def load_all():
   x =  'c'; y =  '.txt'
   for i in range(30):
      dataset_all.append(x + str(matAll[i]) + y)
   return dataset_all

# sorted from hi => lo MAT score
def load_all_sort():
   x =  'c'; y =  '.txt'
   for i in range(30):
      dataset_all_sort.append(x + str(mat_hi_lo[i]) + y)
   return dataset_all_sort
      
