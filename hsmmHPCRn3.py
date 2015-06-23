import os,sys
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

import pyhsmm
from pyhsmm import nicetext
import warnings

import brewer2mpl
import seaborn as sns
import statsmodels.api as sm
import pandas as pd

import shelve
import csv

sns.set(style='white')
#pyhsmm.internals.states.use_eigen()

warnings.filterwarnings("ignore")
# suppress  warnings
np.seterr(divide='ignore')

class HSMM():
   def __init__(self, pnn, couple_type,n_resampling,alpha,gamma,mu, lmda, kappa,nu,k,theta,n_states,n_trunc):
      self.data = None
      self.couple_type = couple_type
      self.n_resampling = '100' if not n_resampling else n_resampling  #Gibbs
      self.alpha = '10.0' if not alpha else alpha                                      # Transitions  concentration parameter
      self.gamma = '10.0' if not gamma else gamma                            #  Transitions  GEM(gamma)
      self.mu = 'np.zeros(2)' if not mu else mu                                      #  Observation; mean
      self.lmbda = 'np.eye(2)' #if not lmda else lmda                              #  Observation: covariance, variance
      self.kappa = '0.1' if not kappa else kappa                                     #  Observation: Prior observations
      self.nu = '4' if not nu else nu                                                         #  Observation: degrees of freedom; dimensions + 2
      self.k = '5' if not k else k                                                                # Duration: shape for gamma distribution; prior on Poisson lamda
      self.theta = '3' if not theta else theta                                             # Duration: scale for gamma distribution; prior on Poisson lamda
      self.n_states = '20' if not n_states else n_states                            # Maximum number of States
      self.n_trunc = '60' if not n_trunc else n_trunc                                #  Maximum length of duration
      self.model = None


      self.file_prefix = 'results3/%s_%s_%s_%s_%s_%s_%s' %\
         (self.couple_type,pnn,self.n_resampling,self.alpha,self.gamma,self.n_states,self.n_trunc)

   def GetLabel(self):
      return "%s HSMM models for %s couples." %(len(self.data),self.couple_type )

   def setData(self, data):
      self.data = data

   def equals(self, hsmm):
      self.n_resampling = hsmm.n_resampling
      self.alpha = hsmm.alpha
      self.gamma = hsmm.gamma
      self.mu = hsmm.mu
      self.lmbda = hsmm.lmbda
      self.kappa = hsmm.kappa
      self.nu = hsmm.nu
      self.k = hsmm.k
      self.theta = hsmm.theta
      self.n_states = hsmm.n_states
      self.n_trunc = hsmm.n_trunc

   def Run(self):
      Nmax = int(self.n_states)
      T = self.data[0].shape[0]
      n_resample = int(self.n_resampling)
      obs_hypparams = {
         'mu_0': eval(self.mu),
         'sigma_0':eval(self.lmbda),
         'kappa_0':float(self.kappa),
         'nu_0':float(self.nu)
      }
      dur_hypparams = {
         'alpha_0': float(self.k),     # shape 
         'beta_0':float(self.theta)  # scale (1/beta)
      }

      obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in xrange(Nmax)] #xun: class changed
      dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in xrange(Nmax)]
      alpha      = float(self.alpha)
      gamma   = float(self.gamma)
      n_trunc    = int(self.n_trunc)      

      model = pyhsmm.models.HSMM( # xun: class changed
                                  alpha=alpha,
                                  gamma=gamma,
                                  init_state_concentration=6.0, # xun: new add
                                  obs_distns=obs_distns,
                                  dur_distns=dur_distns,
                                  trunc=n_trunc) # alpha, gamma was 6.,6.; trunc = max duration
      
      for data in self.data:
         model.add_data(data)
      for i in xrange(n_resample):
         model.resample_model()

      # need this #model.print_results(s, x, used_states, dur_distns, trans_distn)
      # get state distn information
      expect_dur =    []
      grp_means_D1 =  []
      grp_means_D2 =  []
      grp_var_D1   =  []
      grp_var_D2   =  []
      grp_covar =     []
      state_idx =     []
      lmbda_values =  []
      for i in range(len(model.obs_distns)):  #see models compare
         state_idx.append(i)
         lmbda_values.append(dur_distns[i].lmbda)
         if model.obs_distns[i].mu[0] < 0.:
            model.obs_distns[i].mu[0] = 0.
         grp_means_D1.append(np.round(model.obs_distns[i].mu[0]))
         if model.obs_distns[i].mu[1] < 0.:
            model.obs_distns[i].mu[1] = 0.
         grp_means_D2.append(np.round(model.obs_distns[i].mu[1]))
         grp_var_D1.append(model.obs_distns[i].sigma[0][0])
         grp_var_D2.append(model.obs_distns[i].sigma[1][1])
         grp_covar.append(model.obs_distns[i].sigma[0][1])
      state_values = zip(grp_means_D1, grp_means_D2)
      sys_dict =  dict(zip(state_idx, state_values))

      self.T = T
      self.model = model
      self.states_list = model.states_list
      self.sys_dict = sys_dict
      self.grp_info = zip(state_idx,lmbda_values,grp_means_D1,grp_means_D2,grp_var_D1,grp_var_D2, grp_covar)
      self.states_dict_list,self.results_list = self.model.prerun()

   def GetSystemStates(self):
      content   = '\n********************************************************************\n'
      content += 'System Level State Structure for couples: %s' % params[0]
      content += '\n********************************************************************\n'
      headers =  'State', 'Duration', 'Male', 'Female', '  M:Var', '  F:Var', '     Cov'
      content += '\n'
      content += nicetext.SimpleTable(
         self.grp_info,
         headers,
         title = 'Group Level: States, Expected Duration, Means, Variance, Covariance' ,
         fmt={'data_fmt':['%g','%1.3f', '%1.1f', '%1.1f', '%1.3f', '%1.3f', '%1.3f']}).as_text()

      """
        results, labels =  self.model.generate(self.T, keep=False)
        content += '\nStates Module: Generated Sequence; General Representation\n'
        labelout =  labels.tolist()
        for i in labelout:
            content += '%s\n'% labelout[i]
        """

      content += '\n***********************************************************\n'
      content += '%s Individual states:' % len(self.results_list)
      content += '\n***********************************************************\n'
      for results in self.results_list:
         content += results
      return content

   def PlotResults(self, selections):
      title="HSMM model for %s couple (%s files)" %(self.couple_type,len(self.data))
      plt.figure(figsize=(15,15))
      plt.title("HSMM Models") #title)
      self.model.selected_plot(selections,title)
      #plt.show()
      #plt.savefig(self.file_prefix+"_plot.pdf")

   def simulate_individual(self,index):
      SumStatesConstruction = {2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8,10:9}  #sum as key; state as value
      #StatesConstruction = {1:2,2:3,3:4,4:5,5:6,6:7,7:8,8:9,9:10}  #States as key, sum as value
      mat_States = []
      s = self.states_list[index]
      subject_dict = self.states_dict_list[index]
      #s.generate()
      sub_state_seq = []
      sub_state_seq_X =  []
      sub_state_seq_Y =  []
      for i in range(len(s.stateseq)):
         sub_state_seq.append(subject_dict[s.stateseq[i]])
      for i in range(len(sub_state_seq)):
         sub_state_seq_X.append(sub_state_seq[i][0])
         sub_state_seq_Y.append(sub_state_seq[i][1])
      plt.figure(figsize=(15, 10))
      plt.plot(sub_state_seq_X, label = 'M', linewidth=2, color= '#377eb8' )
      plt.plot(sub_state_seq_Y, label = 'F', linewidth=2, color= '#e41a1c' )
      ''' Dump the simulated affect plus mat_States for each couple '''
      with open(self.file_prefix+'_simCoupleAffect.csv', 'a') as sim_affect:
         sim_writer = csv.writer(sim_affect)
         sim_writer.writerow(())
         coupleRow = 'Couple:'
         joiner = 'of'
         sim_writer.writerow((coupleRow,index,joiner,params[0]))
         sim_writer.writerow(())
         for i in range(len(sub_state_seq_X)):
            generateStates =  int(sub_state_seq_X[i])+int(sub_state_seq_Y[i])
            mat_States = SumStatesConstruction[generateStates]
            sim_writer.writerow((index,i,int(sub_state_seq_X[i]),int(sub_state_seq_Y[i]),mat_States))
      sim_affect.close()

      plt.title('Plausible Sequence of States for individual subject %s in Group: %s'%(index,self.couple_type))
      plt.legend()
      plt.xlim((0, self.T))
      plt.ylim((1,5))
      #plt.show()
      #plt.savefig(self.file_prefix+"_sim_%d.pdf"%index)

      sub_state_seq_X = np.asarray(sub_state_seq_X)  #convert to array
      sub_state_seq_Y = np.asarray(sub_state_seq_Y)

      c_data = {'Male': sub_state_seq_X, 'Female': sub_state_seq_Y}
      c_kdes = pd.DataFrame(c_data, columns=['Male', 'Female'])
      sns.jointplot(sub_state_seq_X, sub_state_seq_Y,kind="kde",stat_func=None,
                    xlim= ((1,5)),ylim= ((1,5))); #color=("RdPu_r",8)

      #plt.savefig(self.file_prefix+"kde_sim_%d.pdf"%index)


   def simulate_generic(self):
      T = self.T
      sys_dict = self.sys_dict
      results, labels =  self.model.generate(T, keep=False)
      state_seq = []
      state_seq_X =  []
      state_seq_Y =  []
      for i in range(T):
         state_seq.append(sys_dict[labels[i]])
      for i in range(len(state_seq)):
         state_seq_X.append(state_seq[i][0])
         state_seq_Y.append(state_seq[i][1])
      plt.figure(figsize=(15, 10))
      plt.plot(state_seq_X, label = 'M', linewidth=2, color = '#377eb8' )
      plt.plot(state_seq_Y, label = 'F', linewidth=2, color = '#e41a1c' )
      plt.title('Plausible Sequence of States for Satisfaction Group: %s'%self.couple_type)
      plt.legend()
      plt.xlim((0, self.T))
      plt.ylim((1,5))
      #plt.show()
      #plt.savefig(self.file_prefix+"_sim_generic.pdf")


class HSMMSimDyadApp():
   def __init__(self, params):
      self.hsmm_model = HSMM(params[14], params[1] + "_" + params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9],params[10],params[11],params[12],params[13])
      file_list = eval(params[0])
      files = ['mat%s/c%i.txt'%(params[1],i) for i in file_list]
      dataset = self.select_read_files(files)
      self.hsmm_model.setData(dataset)


   def select_read_files(self,paths):
      dataset = []
      for f in paths:
         data = np.transpose(np.loadtxt(f, usecols=(0,1),unpack=True))
         dataset.append(data)
      return dataset

   def saveResults(self):
      o = open(self.hsmm_model.file_prefix + '.txt','w')
      content = self.hsmm_model.GetSystemStates()
      o.write("%s \n" % self.start_time)
      o.write("%s\n" % self.hsmm_model.GetLabel())
      o.write(content)
      o.write("%s" % self.end_time)
      o.write("%s" % (self.end_time-self.start_time))
      o.close()

   def plotResults(self):
      selections = [0,1,2,3,4]
      lst = ["Observation Distrubtions","State Sequences","Durations","State Traces","Original Data"]
      self.hsmm_model.PlotResults(selections)

   def run(self):
      self.start_time = datetime.now()
      self.hsmm_model.Run()      
      self.end_time = datetime.now()

   def runSimulation(self):
      n_agent = len(self.hsmm_model.data)      
      for i in range(n_agent):
         self.hsmm_model.simulate_individual(i)
      self.hsmm_model.simulate_generic()

def read_parameters(n):
   fname = "./conf/parameter%s.conf" % n
   f = open(fname)
   lines = f.readlines()
   f.close()
   coupletype   = lines[0].strip()
   data_group   = lines[1].strip()
   files        = lines[2].strip()
   n_resampling = lines[3].strip()
   alpha        = lines[4].strip()
   gamma        = lines[5].strip()
   mu           = lines[6].strip()
   lmda         = lines[7].strip()
   kappa        = lines[8].strip()
   nu           = lines[9].strip()
   k            = lines[10].strip()
   theta        = lines[11].strip()
   n_states     = lines[12].strip()
   n_trunc      = lines[13].strip()

   print "Couples included in this analysis:", files

   return files, coupletype,data_group,n_resampling,alpha,gamma,mu,lmda,kappa,nu,k,theta,n_states,n_trunc, n

if __name__ == "__main__":
   from time import time
   t0 = time()
   params = read_parameters(sys.argv[1])
   app = HSMMSimDyadApp(params)
   app.run()
   app.saveResults()
   app.plotResults()
   app.runSimulation()
   t1 = time()
   print 'Run took %3.2f min' %((t1-t0)/60.)