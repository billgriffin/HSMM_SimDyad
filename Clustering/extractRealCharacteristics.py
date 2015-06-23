""" generate time series values from base sequence  """
import scipy
from scipy import stats
import numpy as np
from numpy import savetxt  #add
import numpy as np
from numpy import *
from numpy.random import *
from sklearn import preprocessing, metrics
import matplotlib.pyplot as plt
#import pywt
import glob
import os
import nicetext
import sys
sys.path.append('/home/wmgriffin/anaconda/lib/python2.7/site-packages/thoth')

file_dict = {}

file_dict["Ids"] = {"Ids":[200, 201, 202, 203, 204, 205, 206, 207, 208,\
                           209, 210, 211, 212, 213, 214, 215, 217, 218, 219,\
                           220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230]}

Ids =[200, 201, 202, 203, 204, 205, 206, 207, 208,\
      209, 210, 211, 212, 213, 214, 215, 217, 218, 219,\
      220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230]



'''These are ordered by MAT Category level -- not raw mat score'''
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

allHigh =   [200,203,206,208,209,211,212,213,217,225]
allMedium = [202,205,207,210,214,215,218,219,220,228]
allLow =    [201,204,221,222,223,224,226,227,229,230]

mat_scores = [(128,142,135),
              (35,78,56.5),
              (111,101,106),
              (117,120,118.5),
              (86,91,88.5),
              (114,115,114.5),
              (121,133,127),
              (107,101,104),
              (144,132,138),
              (125,129,127),
              (111,79,95),
              (142,124,133),
              (135,137,136),
              (110,130,120),
              (99,111,105),
              (68,132,100),
              (134,140,137),
              (113,92,102.5),
              (93,98,95.5),
              (102,93,97.5),
              (63,72,67.5),
              (79,98,88.5),
              (75,54,64.5),
              (97,68,82.5),
              (134,108,121),
              (68,86,77),
              (90,61,75.5),
              (119,116,117.5),
              (45,43,44),
              (67,83,75)]

mat_dict = dict(zip(Ids,mat_scores))

#for k,v in mat_dict.iteritems():
   #print k,v

print

for i in allLow:
   print '%g,%g,%g,%1.1f' % (i,mat_dict[i][0],mat_dict[i][1],mat_dict[i][2])
print

entropy_values =[]
for i in allHigh: print
print
for i in allMedium: print i
print
for i in allLow: print i

def entropy1d_get(a): #similar to entropy1
   #scores = stats._support.unique(a)
   scores = np.unique(a)
   sumforprob = len(a)
   freq = np.zeros(len(scores))
   probi = np.zeros(len(scores))
   surprise = np.zeros(len(scores))
   entropyestimate = np.zeros(len(scores))
   if a.ndim == 1:
      scores = np.sort(scores)
      for i in range(len(scores)):
         freq[i] = np.sum(a==scores[i])
         probi[i] = np.array(freq[i]/sumforprob)
         surprise[i] = np.log2(1/probi[i])
         entropyestimate[i] = probi[i] * surprise[i]
      entropy = sum( entropyestimate )
      maxentropy = -np.log2(1./len(scores))
      entropies = [maxentropy,entropy, maxentropy - entropy, entropy/maxentropy]
      '''entropy_values.append(entropies)'''
      print 'Simple dyadic states probability values'
      print 'Max Entropy: %1.4f; Entropy: %1.4f; Divergence: %1.4f; Ratio: %1.4f' %\
            ( maxentropy,entropy, maxentropy - entropy, entropy/maxentropy)
   elif a.ndim == 2: #2 event transitions
      scores = scores[np.lexsort(np.fliplr(scores).T)]
      for i in range(len(scores)):
         freq[i] = np.sum(np.all(a==scores[i],1))
         probi[i] = np.array(freq[i]/sumforprob)
         surprise[i] = np.log2(1/probi[i])
         entropyestimate[i] = probi[i] * surprise[i]
      entropy = sum( entropyestimate )
      maxentropy = -np.log2(1./len(scores))
      #print 'item', scores[i], freq[i], probi[i], surprise[i]
   else:
      raise ValueError, "Input must be <= 2-d."
   #return np.array( stats._support.abut(scores, freq, probi, surprise) )
   return np.array( zip(scores, freq, probi, surprise) )

# values from original couple data
def extract_realCharacteristics():
   couple_summary =[]
   real_couple_dict = {}
   os.chdir("./matAll")
   for file in glob.glob("*.txt"):
      file_id = (int(file[1:-4]))
      dyad_data = np.loadtxt(file)
      male = []; male1 = []; male2 = []
      female = []; female1 = []; female2 = []
      dyadicstate = []; dyadicstate1 = []; dyadicstate2 = []
      ### make length even number
      if len(dyad_data)%2 != 0:
         use_range = (len(dyad_data))-1
      else:
         use_range = len(dyad_data)

      '''First Half'''
      if file_id in allHigh:
         #print 1
         matLevel = 1
      elif file_id in allMedium:
         #print 2
         matLevel = 2
      else:
         #print 3  # low MAT
         matLevel = 3

      print
      print file_id, matLevel, '\n\nFirst half\n'

      male_mat   = mat_dict[file_id][0]
      female_mat = mat_dict[file_id][1]
      dyad_mat   = mat_dict[file_id][2]

      for i in range(use_range):
         if i <= (use_range/2):
            male1.append(int(dyad_data[i][0]))
            female1.append(int(dyad_data[i][1]))
            dyadicstate1.append( (int(dyad_data[i][0]) + int(dyad_data[i][1]) )/2.)

      addAffects = [3,4,5]    #avoid divide by zero errors with ratios
      for i in range(3):
         male1.append(addAffects[i])
         female1.append(addAffects[i])
         dyadicstate1.append(float(addAffects[i]))

      male1 = np.array(male1)
      female1 = np.array(female1)
      dyadicstate1 = np.array(dyadicstate1)

      """ Get mutual information """
      print 'skilearn MI: %1.3f' % metrics.mutual_info_score(male1, female1, contingency=None)
      print

      #import thoth.thoth as thoth

      #print("=========ENTROPY _male===========")
      #a_prob = thoth.prob_from_array(male1)
      #thoth.norm_prob(a_prob)
      ##thoth.entropy_nsb(a_prob)
      #results = thoth.calc_entropy(male1, 100000)
      #print(results)
      #print("=========ENTROPY_female===========")
      #a_prob = thoth.prob_from_array(female1)
      #thoth.norm_prob(a_prob)
      ##thoth.entropy_nsb(a_prob)
      #results = thoth.calc_entropy(female1, 100000)
      #print(results)

      #print("=========JSD===========")
      #results = thoth.calc_jsd(male1, female1, 0.5, 100000)
      #print(results)

      #print("=========MI===========")
      #arrays = np.array([male1,female1])
      #results = thoth.calc_mi(arrays, 100000)
      #print(results)
      #print



      #from it_tool_information import InformationTheoryTool
      ## Define data array
      #dataSex = np.vstack((male1,female1))
      #it_tools = InformationTheoryTool(dataSex)

      #print
      #print 'it_entropy male: %1.3f '  % it_tools.single_entropy(dataSex[0], 2, True)
      #print 'it_entropy female: %1.3f '% it_tools.single_entropy(1, 2)
      #print np.mean(male1)/np.mean(female1)
      #mf_eratio_it =  it_tools.single_entropy(0, 2)/it_tools.single_entropy(1, 2)
      #print ('it_entropy:male/female: %1.3f' % mf_eratio_it)
      #print 'it_entropy between: %1.3f ' % it_tools.entropy(0, 1, 2)
      #print 'it_mutual Information: %1.3f '%mutual_information(dataSex[0], dataSex[1], 2)
      #print

      import gistfile1  #base2
      gist_m_entropy = gistfile1.entropy(male1)
      gist_f_entropy = gistfile1.entropy(female1)
      mf_eratio_gist = gist_m_entropy/gist_f_entropy
      gist_mi = gistfile1.mutual_information(female1,male1)
      print 'gist_m_entropy: %1.3f' % gist_m_entropy
      print 'gist_f_entropy: %1.3f' % gist_f_entropy
      print 'gist_mf_ratio: %1.3f' % mf_eratio_gist
      print 'gist_mi: %1.3f' % gist_mi
      print

      import entropy_estimators  #this is npeet; base 2
      print 'npeet_entropy male: %1.3f ' % entropy_estimators.entropyd(male1, base=2)
      print 'npeet_entropy female: %1.3f'  % entropy_estimators.entropyd(female1, base=2)
      mf_eratio_npeet = entropy_estimators.entropyd(male1, base=2)/entropy_estimators.entropyd(female1, base=2)
      print 'npeet_entropy:male/female: %1.3f' % mf_eratio_npeet
      print 'npeet_mutual Information: %1.3f '% entropy_estimators.midd(male1,female1)
      #npeet_cmidd = entropy_estimators.cmidd(male1,female1,?)
      print

      import kl
      print 'K-l (relative entropy): %1.3f:' % kl.kl(male1,female1)
      print


      """ Male 1 """
      male1_mean = np.mean(male1[:-3])         #remove added elements
      male1_sd = np.std(male1[:-3])
      male1_sum = float(np.sum(male1[:-3]))    #higher more negative

      ### Negative
      male1_neg = []
      for i in range(len(male1)):
         if male1[i] > 4:                      #excludes positive; neutral
            male1_neg.append( male1[i] )
      male1_neg_count = float(len(male1_neg))
      ### Positive
      male1_pos = []
      for i in range(len(male1)):
         if male1[i] < 4:                      #excludes negative; neutral
            male1_pos.append( male1[i] )
      male1_pos_count = float(len(male1_pos))
      ### Neutral
      male1_neu = []
      for i in range(len(male1)):
         if male1[i] == 4:
            male1_neu.append( male1[i] )
      male1_neu_count = float(len(male1_neu))

      male1_pos_neg_ratio = male1_pos_count/male1_neg_count  #ratio of neg to pos
      male1_neg_neu_ratio = male1_neg_count/male1_neu_count  #ratio of neg to neu
      male1_pos_neu_ratio = male1_pos_count/male1_neu_count  #ratio of pos to neu
      male1_lnodds_ratio = np.log(male1_pos_neu_ratio/male1_neg_neu_ratio)

      male1_neg_mean = float(np.mean(male1_neg))
      male1_neg_sd = float(np.std(male1_neg))

      male1_pos_mean = float(np.mean(male1_pos))
      male1_pos_sd = float(np.std(male1_pos))

      """ Female 1 """

      female1_mean = np.mean(female1[:-3]) #remove added elements
      female1_sd = np.std(female1[:-3])
      female1_sum = float(np.sum(female1[:-3]))       #higher more negative

      ### Negative
      female1_neg = []
      for i in range(len(female1)):
         if female1[i] > 4:                    #excludes positive; neutral
            female1_neg.append( female1[i] )
      female1_neg_count = float(len(female1_neg))
      ### Positive
      female1_pos = []
      for i in range(len(female1)):
         if female1[i] < 4:                    #excludes negative; neutral
            female1_pos.append( female1[i] )
      female1_pos_count = float(len(female1_pos))
      ### Neutral
      female1_neu = []
      for i in range(len(female1)):
         if female1[i] == 4:
            female1_neu.append( female1[i] )
      female1_neu_count = float(len(female1_neu))

      female1_pos_neg_ratio = female1_pos_count/female1_neg_count  #ratio of neg to pos
      female1_neg_neu_ratio = female1_neg_count/female1_neu_count  #ratio of neg to neu
      female1_pos_neu_ratio = female1_pos_count/female1_neu_count  #ratio of pos to neu
      female1_lnodds_ratio = np.log(female1_pos_neu_ratio/female1_neg_neu_ratio)

      female1_neg_mean = float(np.mean(female1_neg))
      female1_neg_sd = float(np.std(female1_neg))

      female1_pos_mean = float(np.mean(female1_pos))
      female1_pos_sd = float(np.std(female1_pos))

      ###########################################
      dyadicstate1_mean = np.mean(dyadicstate1)
      dyadicstate1_sd = np.std(dyadicstate1)
      dyadicstate1_sum = float(np.sum(dyadicstate1))       #higher more negative
      ### Negative
      dyadicstate1_neg = []
      for i in range(len(dyadicstate1)):
         if dyadicstate1[i] > 4:                    #excludes positive; neutral
            dyadicstate1_neg.append( dyadicstate1[i] )
      dyadicstate1_neg_sum = float(np.sum(dyadicstate1_neg))
      dyadicstate1_neg_count = float(len(dyadicstate1_neg))
      ### Positive
      dyadicstate1_pos = []
      for i in range(len(dyadicstate1)):
         if dyadicstate1[i] < 4:                    #excludes neutral,  negative
            dyadicstate1_pos.append( dyadicstate1[i] )
      dyadicstate1_pos_sum = float(np.sum(dyadicstate1_pos))
      dyadicstate1_pos_count = float(len(dyadicstate1_pos))

      ### Neutral
      dyadicstate1_neu = []
      for i in range(len(dyadicstate1)):
         if dyadicstate1[i] == 4:                    #excludes positive; negative
            dyadicstate1_neu.append( dyadicstate1[i] )
      dyadicstate1_neu_sum = float(np.sum(dyadicstate1_neu))
      dyadicstate1_neu_count = float(len(dyadicstate1_neu))

      dyadicstate1_pos_neg_ratio = dyadicstate1_pos_count/dyadicstate1_neg_count  #ratio of neg to pos

      dyadicstate1_neg_neu_ratio = dyadicstate1_neg_count/dyadicstate1_neu_count  #ratio of neg to neu
      dyadicstate1_pos_neu_ratio = dyadicstate1_pos_count/dyadicstate1_neu_count  #ratio of pos to neu
      dyadicstate1_lnodds_ratio = np.log(dyadicstate1_pos_neu_ratio/dyadicstate1_neg_neu_ratio)


      #####################################
      window = 1
      male1_female1_compare1 = [file_id,matLevel,
                                window,
                                male_mat,
                                male1_mean,
                                male1_sd,
                                male1_sum,
                                male1_neg_count,
                                male1_pos_count,
                                male1_neu_count,
                                male1_pos_neg_ratio,
                                male1_neg_neu_ratio,
                                male1_pos_neu_ratio,
                                male1_lnodds_ratio,
                                male1_neg_mean,
                                male1_neg_sd,
                                male1_pos_mean,
                                male1_pos_sd,
                                female_mat,
                                female1_mean,
                                female1_sd,
                                female1_sum,
                                female1_neg_count,
                                female1_pos_count,
                                female1_neu_count,
                                female1_pos_neg_ratio,
                                female1_neg_neu_ratio,
                                female1_pos_neu_ratio,
                                female1_lnodds_ratio,
                                female1_neg_mean,
                                female1_neg_sd,
                                female1_pos_mean,
                                female1_pos_sd,
                                dyad_mat,
                                dyadicstate1_mean,
                                dyadicstate1_sd,
                                dyadicstate1_sum,
                                dyadicstate1_neg_sum,
                                dyadicstate1_neg_count,
                                dyadicstate1_pos_sum,
                                dyadicstate1_pos_count,
                                dyadicstate1_neu_sum,
                                dyadicstate1_neu_count,
                                dyadicstate1_pos_neg_ratio,
                                dyadicstate1_neg_neu_ratio,
                                dyadicstate1_pos_neu_ratio,
                                dyadicstate1_lnodds_ratio]

      couple_summary.append(male1_female1_compare1)



      print 'Male 1st'
      prob_base = entropy1d_get(male1)
      headers = 'State','Freq','Prob','Surprise'
      print nicetext.SimpleTable(prob_base,
                                 headers,
                                 title = 'Distribution of Single States' ,
                                 fmt={'data_fmt':["%s","%g","%.4f","%2.4f"]}).as_text()
      print 'Female 1st'
      prob_base = entropy1d_get(female1)
      print nicetext.SimpleTable(prob_base,
                                 headers,
                                 title = 'Distribution of Single States' ,
                                 fmt={'data_fmt':["%s","%g","%.4f","%2.4f"]}).as_text()
      ####################################################################################
      '''Second Half'''
      print file_id, matLevel, '\n\nSecond half\n'

      for i in range(use_range):
         if i > (use_range/2):
            male2.append(int(dyad_data[i][0]))
            female2.append(int(dyad_data[i][1]))
            dyadicstate2.append( (int(dyad_data[i][0]) + int(dyad_data[i][1]) )/2.)

      addAffects = [3,4,5]    #avoid divide by zero errors with ratios
      for i in range(3):
         male2.append(addAffects[i])
         female2.append(addAffects[i])
         dyadicstate2.append(float(addAffects[i]))

      male2 = np.array(male2)
      female2 = np.array(female2)
      dyadicstate2 = np.array(dyadicstate2)

      """ Get mutual information """
      mu_info2 = metrics.mutual_info_score(male2, female2, contingency=None)
      print mu_info2

      """ Male 2 """
      male2_mean = np.mean(male2[:-3])         #remove added elements
      male2_sd = np.std(male2[:-3])
      male2_sum = float(np.sum(male2[:-3]))    #higher more negative

      ### Negative
      male2_neg = []
      for i in range(len(male2)):
         if male2[i] > 4:                      #excludes positive; neutral
            male2_neg.append( male2[i] )
      male2_neg_count = float(len(male2_neg))
      ### Positive
      male2_pos = []
      for i in range(len(male2)):
         if male2[i] < 4:                      #excludes negative; neutral
            male2_pos.append( male2[i] )
      male2_pos_count = float(len(male2_pos))
      ### Neutral
      male2_neu = []
      for i in range(len(male2)):
         if male2[i] == 4:
            male2_neu.append( male2[i] )
      male2_neu_count = float(len(male2_neu))

      male2_pos_neg_ratio = male2_pos_count/male2_neg_count  #ratio of neg to pos
      male2_neg_neu_ratio = male2_neg_count/male2_neu_count  #ratio of neg to neu
      male2_pos_neu_ratio = male2_pos_count/male2_neu_count  #ratio of pos to neu
      male2_lnodds_ratio = np.log(male2_pos_neu_ratio/male2_neg_neu_ratio)

      male2_neg_mean = float(np.mean(male2_neg))
      male2_neg_sd = float(np.std(male2_neg))

      male2_pos_mean = float(np.mean(male2_pos))
      male2_pos_sd = float(np.std(male2_pos))

      """ Female 2 """

      female2_mean = np.mean(female2[:-3]) #remove added elements
      female2_sd = np.std(female2[:-3])
      female2_sum = float(np.sum(female2[:-3]))       #higher more negative

      ### Negative
      female2_neg = []
      for i in range(len(female2)):
         if female2[i] > 4:                    #excludes positive; neutral
            female2_neg.append( female2[i] )
      female2_neg_count = float(len(female2_neg))
      ### Positive
      female2_pos = []
      for i in range(len(female2)):
         if female2[i] < 4:                    #excludes negative; neutral
            female2_pos.append( female2[i] )
      female2_pos_count = float(len(female2_pos))
      ### Neutral
      female2_neu = []
      for i in range(len(female2)):
         if female2[i] == 4:
            female2_neu.append( female2[i] )
      female2_neu_count = float(len(female2_neu))

      female2_pos_neg_ratio = female2_pos_count/female2_neg_count  #ratio of neg to pos
      female2_neg_neu_ratio = female2_neg_count/female2_neu_count  #ratio of neg to neu
      female2_pos_neu_ratio = female2_pos_count/female2_neu_count  #ratio of pos to neu
      female2_lnodds_ratio = np.log(female2_pos_neu_ratio/female2_neg_neu_ratio)

      female2_neg_mean = float(np.mean(female2_neg))
      female2_neg_sd = float(np.std(female2_neg))

      female2_pos_mean = float(np.mean(female2_pos))
      female2_pos_sd = float(np.std(female2_pos))

      ###########################################
      dyadicstate2_mean = np.mean(dyadicstate2)
      dyadicstate2_sd = np.std(dyadicstate2)
      dyadicstate2_sum = float(np.sum(dyadicstate2))       #higher more negative
      ### Negative
      dyadicstate2_neg = []
      for i in range(len(dyadicstate2)):
         if dyadicstate2[i] > 4:                    #excludes positive; neutral
            dyadicstate2_neg.append( dyadicstate2[i] )
      dyadicstate2_neg_sum = float(np.sum(dyadicstate2_neg))
      dyadicstate2_neg_count = float(len(dyadicstate2_neg))
      ### Positive
      dyadicstate2_pos = []
      for i in range(len(dyadicstate2)):
         if dyadicstate2[i] < 4:                    #excludes neutral,  negative
            dyadicstate2_pos.append( dyadicstate2[i] )
      dyadicstate2_pos_sum = float(np.sum(dyadicstate2_pos))
      dyadicstate2_pos_count = float(len(dyadicstate2_pos))

      ### Neutral
      dyadicstate2_neu = []
      for i in range(len(dyadicstate2)):
         if dyadicstate2[i] == 4:                    #excludes positive; negative
            dyadicstate2_neu.append( dyadicstate2[i] )
      dyadicstate2_neu_sum = float(np.sum(dyadicstate2_neu))
      dyadicstate2_neu_count = float(len(dyadicstate2_neu))

      dyadicstate2_pos_neg_ratio = dyadicstate2_pos_count/dyadicstate2_neg_count  #ratio of neg to pos

      dyadicstate2_neg_neu_ratio = dyadicstate2_neg_count/dyadicstate2_neu_count  #ratio of neg to neu
      dyadicstate2_pos_neu_ratio = dyadicstate2_pos_count/dyadicstate2_neu_count  #ratio of pos to neu
      dyadicstate2_lnodds_ratio = np.log(dyadicstate2_pos_neu_ratio/dyadicstate2_neg_neu_ratio)


      #####################################
      window = 2
      male2_female2_compare2 = [file_id,matLevel,window,
                                male_mat,
                                male2_mean,
                                male2_sd,
                                male2_sum,
                                male2_neg_count,
                                male2_pos_count,
                                male2_neu_count,
                                male2_pos_neg_ratio,
                                male2_neg_neu_ratio,
                                male2_pos_neu_ratio,
                                male2_lnodds_ratio,
                                male2_neg_mean,
                                male2_neg_sd,
                                male2_pos_mean,
                                male2_pos_sd,
                                female_mat,
                                female2_mean,
                                female2_sd,
                                female2_sum,
                                female2_neg_count,
                                female2_pos_count,
                                female2_neu_count,
                                female2_pos_neg_ratio,
                                female2_neg_neu_ratio,
                                female2_pos_neu_ratio,
                                female2_lnodds_ratio,
                                female2_neg_mean,
                                female2_neg_sd,
                                female2_pos_mean,
                                female2_pos_sd,
                                dyad_mat,
                                dyadicstate2_mean,
                                dyadicstate2_sd,
                                dyadicstate2_sum,
                                dyadicstate2_neg_sum,
                                dyadicstate2_neg_count,
                                dyadicstate2_pos_sum,
                                dyadicstate2_pos_count,
                                dyadicstate2_neu_sum,
                                dyadicstate2_neu_count,
                                dyadicstate2_pos_neg_ratio,
                                dyadicstate2_neg_neu_ratio,
                                dyadicstate2_pos_neu_ratio,
                                dyadicstate2_lnodds_ratio]

      couple_summary.append(male2_female2_compare2)

      print 'Male 2nd'
      prob_base = entropy1d_get(male2)
      headers = 'State','Freq','Prob','Surprise'
      print nicetext.SimpleTable(prob_base,
                                 headers,
                                 title = 'Distribution of Single States' ,
                                 fmt={'data_fmt':["%s","%g","%.4f","%2.4f"]}).as_text()
      print 'Female 2nd'
      prob_base = entropy1d_get(female2)
      print nicetext.SimpleTable(prob_base,
                                 headers,
                                 title = 'Distribution of Single States' ,
                                 fmt={'data_fmt':["%s","%g","%.4f","%2.4f"]}).as_text()

   ####################################################################################
      '''Complete Sequence'''
      print file_id, matLevel, 'Complete Sequence'

      for i in range(use_range):
         male.append(int(dyad_data[i][0]))
         female.append(int(dyad_data[i][1]))
         dyadicstate.append( (int(dyad_data[i][0]) + int(dyad_data[i][1]) )/2.)

      addAffects = [3,4,5]    #avoid divide by zero errors with ratios
      for i in range(3):
         male.append(addAffects[i])
         female.append(addAffects[i])
         dyadicstate.append(float(addAffects[i]))

      male = np.array(male)
      female = np.array(female)
      dyadicstate = np.array(dyadicstate)
      """ Get mutual information """
      mu_info = metrics.mutual_info_score(male, female, contingency=None)
      print mu_info

      """ Male  """
      male_mean = np.mean(male[:-3])         #remove added elements
      male_sd = np.std(male[:-3])
      male_sum = float(np.sum(male[:-3]))    #higher more negative

      ### Negative
      male_neg = []
      for i in range(len(male)):
         if male[i] > 4:                      #excludes positive; neutral
            male_neg.append( male[i] )
      male_neg_count = float(len(male_neg))
      ### Positive
      male_pos = []
      for i in range(len(male)):
         if male[i] < 4:                      #excludes negative; neutral
            male_pos.append( male[i] )
      male_pos_count = float(len(male_pos))
      ### Neutral
      male_neu = []
      for i in range(len(male)):
         if male[i] == 4:
            male_neu.append( male[i] )
      male_neu_count = float(len(male_neu))

      male_pos_neg_ratio = male_pos_count/male_neg_count  #ratio of neg to pos
      male_neg_neu_ratio = male_neg_count/male_neu_count  #ratio of neg to neu
      male_pos_neu_ratio = male_pos_count/male_neu_count  #ratio of pos to neu
      male_lnodds_ratio = np.log(male_pos_neu_ratio/male_neg_neu_ratio)

      male_neg_mean = float(np.mean(male_neg))
      male_neg_sd = float(np.std(male_neg))

      male_pos_mean = float(np.mean(male_pos))
      male_pos_sd = float(np.std(male_pos))

      """ Female  """

      female_mean = np.mean(female[:-3]) #remove added elements
      female_sd = np.std(female[:-3])
      female_sum = float(np.sum(female[:-3]))       #higher more negative

      ### Negative
      female_neg = []
      for i in range(len(female)):
         if female[i] > 4:                    #excludes positive; neutral
            female_neg.append( female[i] )
      female_neg_count = float(len(female_neg))
      ### Positive
      female_pos = []
      for i in range(len(female)):
         if female[i] < 4:                    #excludes negative; neutral
            female_pos.append( female[i] )
      female_pos_count = float(len(female_pos))
      ### Neutral
      female_neu = []
      for i in range(len(female)):
         if female[i] == 4:
            female_neu.append( female[i] )
      female_neu_count = float(len(female_neu))

      female_pos_neg_ratio = female_pos_count/female_neg_count  #ratio of neg to pos
      female_neg_neu_ratio = female_neg_count/female_neu_count  #ratio of neg to neu
      female_pos_neu_ratio = female_pos_count/female_neu_count  #ratio of pos to neu
      female_lnodds_ratio = np.log(female_pos_neu_ratio/female_neg_neu_ratio)

      female_neg_mean = float(np.mean(female_neg))
      female_neg_sd = float(np.std(female_neg))

      female_pos_mean = float(np.mean(female_pos))
      female_pos_sd = float(np.std(female_pos))

      ###########################################
      dyadicstate_mean = np.mean(dyadicstate)
      dyadicstate_sd = np.std(dyadicstate)
      dyadicstate_sum = float(np.sum(dyadicstate))       #higher more negative
      ### Negative
      dyadicstate_neg = []
      for i in range(len(dyadicstate)):
         if dyadicstate[i] > 4:                    #excludes positive; neutral
            dyadicstate_neg.append( dyadicstate[i] )
      dyadicstate_neg_sum = float(np.sum(dyadicstate_neg))
      dyadicstate_neg_count = float(len(dyadicstate_neg))
      ### Positive
      dyadicstate_pos = []
      for i in range(len(dyadicstate)):
         if dyadicstate[i] < 4:                    #excludes neutral,  negative
            dyadicstate_pos.append( dyadicstate[i] )
      dyadicstate_pos_sum = float(np.sum(dyadicstate_pos))
      dyadicstate_pos_count = float(len(dyadicstate_pos))

      ### Neutral
      dyadicstate_neu = []
      for i in range(len(dyadicstate)):
         if dyadicstate[i] == 4:                    #excludes positive; negative
            dyadicstate_neu.append( dyadicstate[i] )
      dyadicstate_neu_sum = float(np.sum(dyadicstate_neu))
      dyadicstate_neu_count = float(len(dyadicstate_neu))

      dyadicstate_pos_neg_ratio = dyadicstate_pos_count/dyadicstate_neg_count  #ratio of neg to pos

      dyadicstate_neg_neu_ratio = dyadicstate_neg_count/dyadicstate_neu_count  #ratio of neg to neu
      dyadicstate_pos_neu_ratio = dyadicstate_pos_count/dyadicstate_neu_count  #ratio of pos to neu
      dyadicstate_lnodds_ratio = np.log(dyadicstate_pos_neu_ratio/dyadicstate_neg_neu_ratio)


      #####################################
      window = 3
      male_female_compare = [file_id,matLevel,window,
                             male_mat,
                             male_mean,
                             male_sd,
                             male_sum,
                             male_neg_count,
                             male_pos_count,
                             male_neu_count,
                             male_pos_neg_ratio,
                             male_neg_neu_ratio,
                             male_pos_neu_ratio,
                             male_lnodds_ratio,
                             male_neg_mean,
                             male_neg_sd,
                             male_pos_mean,
                             male_pos_sd,
                             female_mat,
                             female_mean,
                             female_sd,
                             female_sum,
                             female_neg_count,
                             female_pos_count,
                             female_neu_count,
                             female_pos_neg_ratio,
                             female_neg_neu_ratio,
                             female_pos_neu_ratio,
                             female_lnodds_ratio,
                             female_neg_mean,
                             female_neg_sd,
                             female_pos_mean,
                             female_pos_sd,
                             dyad_mat,
                             dyadicstate_mean,
                             dyadicstate_sd,
                             dyadicstate_sum,
                             dyadicstate_neg_sum,
                             dyadicstate_neg_count,
                             dyadicstate_pos_sum,
                             dyadicstate_pos_count,
                             dyadicstate_neu_sum,
                             dyadicstate_neu_count,
                             dyadicstate_pos_neg_ratio,
                             dyadicstate_neg_neu_ratio,
                             dyadicstate_pos_neu_ratio,
                             dyadicstate_lnodds_ratio]
      couple_summary.append(male_female_compare)

      from operator import itemgetter
      couple_summary.sort(key=itemgetter(0))  #id
      couple_summary.sort(key=itemgetter(2))  # 1 = mat level; 2 = window (3 is complete sequence)



      couple_headers = 'file_id','matLevel','window','male mat','male_mean',\
      'male_sd','male_sum','male_neg_count','male_pos_count',\
      'male_neu_count','male_pos_neg_ratio','male_neg_neu_ratio',\
      'male_pos_neu_ratio','male_lnodds_ratio','male_neg_mean',\
      'male_neg_sd','male_pos_mean','male_pos_sd','female mat','female_mean',\
      'female_sd','female_sum','female_neg_count',\
      'female_pos_count','female_neu_count','female_pos_neg_ratio',\
      'female_neg_neu_ratio','female_pos_neu_ratio',\
      'female_lnodds_ratio','female_neg_mean','female_neg_sd',\
      'female_pos_mean','female_pos_sd','dyad mat','dyadicstate_mean',\
      'dyadicstate_sd','dyadicstate_sum','dyadicstate_neg_sum',\
      'dyadicstate_neg_count','dyadicstate_pos_sum',\
      'dyadicstate_pos_count','dyadicstate_neu_sum',\
      'dyadicstate_neu_count','dyadicstate_pos_neg_ratio',\
      'dyadicstate_neg_neu_ratio','dyadicstate_pos_neu_ratio',\
      'dyadicstate_lnodds_ratio'
   print nicetext.SimpleTable(couple_summary,
                              couple_headers,
                              title = 'Couple characteristcs' ,
                              fmt={'data_fmt':["%g","%g","%g","%g","%1.3f",\
                                               "%1.3f","%g","%g","%g",\
                                               "%1.3f","%1.3f","%1.3f",\
                                               "%1.3f","%1.3f","%1.3f",\
                                               "%1.3f","%1.3f", "%1.3f","%g","%1.3f",\
                                               "%1.3f","%1.3f","%g",\
                                               "%g","%g","%1.3f",\
                                               "%1.3f","%1.3f",\
                                               "%1.3f","%1.3f","%1.3f",\
                                               "%1.3f","%1.3f","%1.3f","%1.3f",\
                                               "%1.3f","%g","%g",\
                                               "%g","%g",\
                                               "%g","%g",\
                                               "%g","%1.3f",\
                                               "%1.3f","%1.3f",\
                                               "%1.3f"\
                                               ]}).as_text()


   print nicetext.SimpleTable(couple_summary,
                              couple_headers,
                              title = 'Couple characteristcs' ,
                              fmt={'data_fmt':["%g","%g","%g","%g","%1.3f",\
                                               "%1.3f","%g","%g","%g",\
                                               "%1.3f","%1.3f","%1.3f",\
                                               "%1.3f","%1.3f","%1.3f",\
                                               "%1.3f","%1.3f", "%1.3f","%g","%1.3f",\
                                               "%1.3f","%1.3f","%g",\
                                               "%g","%g","%1.3f",\
                                               "%1.3f","%1.3f",\
                                               "%1.3f","%1.3f","%1.3f",\
                                               "%1.3f","%1.3f","%1.3f","%1.3f",\
                                               "%1.3f","%g","%g",\
                                               "%g","%g",\
                                               "%g","%g",\
                                               "%g","%1.3f",\
                                               "%1.3f","%1.3f",\
                                               "%1.3f"\
                                               ]}).as_csv()


   modified_couples = np.array(couple_summary)
   shrt_modified_couples = modified_couples[:,3:]
   dyads_metrics_scaled = preprocessing.scale(shrt_modified_couples)

   """ This is the scaled data -- begining with male mat;
   time window 1,2,3 -- 3 is the complete sequence"""
   couple_headers_scaled = 'male mat','male_mean',\
   'male_sd','male_sum','male_neg_count','male_pos_count',\
   'male_neu_count','male_pos_neg_ratio','male_neg_neu_ratio',\
   'male_pos_neu_ratio','male_lnodds_ratio','male_neg_mean',\
   'male_neg_sd','male_pos_mean','male_pos_sd','female mat','female_mean',\
   'female_sd','female_sum','female_neg_count',\
   'female_pos_count','female_neu_count','female_pos_neg_ratio',\
   'female_neg_neu_ratio','female_pos_neu_ratio',\
   'female_lnodds_ratio','female_neg_mean','female_neg_sd',\
   'female_pos_mean','female_pos_sd','dyad mat','dyadicstate_mean',\
   'dyadicstate_sd','dyadicstate_sum','dyadicstate_neg_sum',\
   'dyadicstate_neg_count','dyadicstate_pos_sum',\
   'dyadicstate_pos_count','dyadicstate_neu_sum',\
   'dyadicstate_neu_count','dyadicstate_pos_neg_ratio',\
   'dyadicstate_neg_neu_ratio','dyadicstate_pos_neu_ratio',\
   'dyadicstate_lnodds_ratio'

   print nicetext.SimpleTable(dyads_metrics_scaled,couple_headers_scaled,
                              fmt={'data_fmt':["%g","%1.3f",\
                                               "%1.3f","%g","%g","%g",\
                                               "%1.3f","%1.3f","%1.3f",\
                                               "%1.3f","%1.3f","%1.3f",\
                                               "%1.3f","%1.3f", "%1.3f","%g","%1.3f",\
                                               "%1.3f","%1.3f","%g",\
                                               "%g","%g","%1.3f",\
                                               "%1.3f","%1.3f",\
                                               "%1.3f","%1.3f","%1.3f",\
                                               "%1.3f","%1.3f","%1.3f","%1.3f",\
                                               "%1.3f","%g","%g",\
                                               "%g","%g",\
                                               "%g","%g",\
                                               "%g","%1.3f",\
                                               "%1.3f","%1.3f",\
                                               "%1.3f"\
                                               ]}).as_csv()

   #dyads_metrics_normalized = preprocessing.normalize(shrt_modified_couples)
   print ('finished')

extract_realCharacteristics()