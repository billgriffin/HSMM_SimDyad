from __future__ import division
import numpy as np
import itertools, collections, operator, random
from matplotlib import pyplot as plt
from matplotlib import cm
from warnings import warn
import brewer2mpl

from basic.abstractions import ModelGibbsSampling, ModelEM
from internals import states, initial_state, transitions

#import seaborn as sns
#sns.set(style='white')#, context='talk'))


class HMM(ModelGibbsSampling, ModelEM):
   '''
   The HMM class is a convenient wrapper that provides useful constructors and
   packages all the components.
   '''

   def __init__(self,
                obs_distns,
                trans_distn=None,
                alpha=None,gamma=None,
                alpha_a_0=None,alpha_b_0=None,gamma_a_0=None,gamma_b_0=None,
                init_state_distn=None,
                init_state_concentration=None):

      self.state_dim = len(obs_distns)
      self.obs_distns = obs_distns
      self.states_list = []

      assert (trans_distn is not None) ^ \
             (alpha is not None and gamma is not None) ^ \
             (alpha_a_0 is not None and alpha_b_0 is not None
              and gamma_a_0 is not None and gamma_b_0 is not None)
      if trans_distn is not None:
         self.trans_distn = trans_distn
      elif alpha is not None:
         self.trans_distn = transitions.HDPHMMTransitions(
            state_dim=self.state_dim,
            alpha=alpha,gamma=gamma)
      else:
         self.trans_distn = transitions.HDPHMMTransitionsConcResampling(
            state_dim=self.state_dim,
            alpha_a_0=alpha_a_0,alpha_b_0=alpha_b_0,
            gamma_a_0=gamma_a_0,gamma_b_0=gamma_b_0)

      assert (init_state_distn is not None) ^ \
             (init_state_concentration is not None)

      if init_state_distn is not None:
         self.init_state_distn = init_state_distn
      else:
         # xun: [2] using initial_state():
         # init_state_concentration= rho = alpha_0
         # state_dim = K
         # NOTE: old code only uses "state_dim", because it has a default rho = 2.0
         # NOTE: these 2 (alpha_0, K) are hyperparameters
         # Descrete distribution over labels, where the parameter is weights
         # ***and the prior is a Dirichlet distribution***: Multinomial(weights,alpha_0, K)
         self.init_state_distn = initial_state.InitialState(
            state_dim=self.state_dim,
            rho=init_state_concentration)

   def add_data(self,data,stateseq=None,**kwargs):
      self.states_list.append(states.HMMStates(len(data),self.state_dim,self.obs_distns,self.trans_distn,
                                               self.init_state_distn,data=data,stateseq=stateseq,**kwargs))

   def add_data_parallel(self,data_id):
      from pyhsmm import parallel
      self.add_data(parallel.alldata[data_id])
      self.states_list[-1].data_id = data_id

   def log_likelihood(self,data):
      # TODO avoid this temp states stuff by making messages methods static
      if len(self.states_list) > 0:
         s = self.states_list[0] # any state sequence object will work
      else:
         # we have to create a temporary one just for its methods, though the
         # details of the actual state sequence are never used
         s = states.HMMStates(len(data),self.state_dim,self.obs_distns,self.trans_distn,
                              self.init_state_distn,stateseq=np.zeros(len(data),dtype=np.uint8))

      aBl = s.get_aBl(data)
      betal = s._messages_backwards(aBl)
      return np.logaddexp.reduce(np.log(self.init_state_distn.pi_0) + betal[0] + aBl[0])

   ### generation

   def generate(self,T,keep=True):
      '''
      Generates a forward sample using the current values of all parameters.
      Returns an observation sequence and a state sequence of length T.

      If keep is True, the states object created is appended to the
      states_list. This is mostly useful for generating synthetic data and
      keeping it around in an HSMM object as the latent truth.

      To construct a posterior sample, one must call both the add_data and
      resample methods first. Then, calling generate() will produce a sample
      from the posterior (as long as the Gibbs sampling has converged). In
      these cases, the keep argument should be False.
      '''
      tempstates = states.HMMStates(T,self.state_dim,self.obs_distns,self.trans_distn,
                                    self.init_state_distn)

      return self._generate(tempstates,keep)

   def _generate(self,tempstates,keep):
      obs,labels = tempstates.generate_obs(), tempstates.stateseq

      if keep:
         tempstates.added_with_generate = True
         tempstates.data = obs
         self.states_list.append(tempstates)

      return obs, labels

   ### Gibbs sampling

   def resample_model(self):
      # resample obsparams
      for state, distn in enumerate(self.obs_distns):
         distn.resample([s.data[s.stateseq == state] for s in self.states_list])

      # resample transitions
      self.trans_distn.resample([s.stateseq for s in self.states_list])

      # resample pi_0
      self.init_state_distn.resample([s.stateseq[:1] for s in self.states_list])

      # resample states
      for s in self.states_list:
         s.resample()

   def resample_model_parallel(self,numtoresample='all'):
      from pyhsmm import parallel
      if numtoresample == 'all':
         numtoresample = len(self.states_list)
      elif numtoresample == 'engines':
         numtoresample = len(parallel.dv)

      ### resample parameters locally
      for state, distn in enumerate(self.obs_distns):
         distn.resample([s.data[s.stateseq == state] for s in self.states_list])
      self.trans_distn.resample([s.stateseq for s in self.states_list])
      self.init_state_distn.resample([s.stateseq[:1] for s in self.states_list])

      ### choose which sequences to resample
      states_to_resample = random.sample(self.states_list,numtoresample)

      ### resample states in parallel
      self._push_self_parallel(states_to_resample)
      self._build_states_parallel(states_to_resample)

      ### purge to prevent memory buildup
      parallel.c.purge_results('all')

   def _push_self_parallel(self,states_to_resample):
      from pyhsmm import parallel
      states_to_restore = [s for s in self.states_list if s not in states_to_resample]
      self.states_list = []
      parallel.dv.push({'global_model':self},block=True)
      self.states_list = states_to_restore

   def _build_states_parallel(self,states_to_resample):
      from pyhsmm import parallel
      raw_stateseq_tuples = parallel.build_states.map([s.data_id for s in states_to_resample])
      for data_id, stateseq in raw_stateseq_tuples:
         self.add_data(
            data=parallel.alldata[data_id],
            stateseq=stateseq)
         self.states_list[-1].data_id = data_id
   ### EM

   def EM_step(self):
      assert len(self.states_list) > 0, 'Must have data to run EM'

      ## E step
      for s in self.states_list:
         s.E_step()

      ## M step
      # observation distribution parameters
      for state, distn in enumerate(self.obs_distns):
         distn.max_likelihood([s.data for s in self.states_list],
                              [s.expectations[:,state] for s in self.states_list])

      # initial distribution parameters
      self.init_state_distn.max_likelihood(None,[s.expectations[0] for s in self.states_list])

      # transition parameters (requiring more than just the marginal expectations)
      self.trans_distn.max_likelihood([(s.alphal,s.betal,s.aBl) for s in self.states_list])

      ## for plotting!
      for s in self.states_list:
         s.stateseq = s.expectations.argmax(1)

   def num_parameters(self):
      return sum(o.num_parameters() for o in self.obs_distns) + self.state_dim**2

   def BIC(self):
      # NOTE: in principle this method computes the BIC only after finding the
      # maximum likelihood parameters (or, of course, an EM fixed-point as an
      # approximation!)
      assert len(self.states_list) > 0, 'Must have data to get BIC'
      return -2*sum(self.log_likelihood(s.data).sum() for s in self.states_list) + \
             self.num_parameters() * np.log(sum(s.data.shape[0] for s in self.states_list))

   ### plotting

   def _get_used_states(self,states_objs=None):
      if states_objs is None:
         states_objs = self.states_list
      canonical_ids = collections.defaultdict(itertools.count().next)
      for s in states_objs:
         for state in s.stateseq:
            canonical_ids[state]
      return map(operator.itemgetter(0),sorted(canonical_ids.items(),key=operator.itemgetter(1)))

   def _get_colors(self):
      states = self._get_used_states()
      numstates = len(states)
      return dict(zip(states,np.linspace(0,1,numstates,endpoint=True)))

   def plot_observations(self,colors=None,states_objs=None):
      self.obs_distns[0]._plot_setup(self.obs_distns)
      if colors is None:
         colors = self._get_colors()
      if states_objs is None:
         states_objs = self.states_list

      cmap = cm.get_cmap()
      used_states = self._get_used_states(states_objs)
      for state,o in enumerate(self.obs_distns):
         if state in used_states:
            o.plot(
               color=cmap(colors[state]),
               data=[s.data[s.stateseq == state] if s.data is not None else None
                     for s in states_objs],
               label='%d' % state,
               cmap=cmap)
      plt.title('Observation Distributions')

   def plot(self,color=None,legend=True):
      plt.gcf() #.set_size_inches((10,10))
      colors = self._get_colors()

      num_subfig_cols = len(self.states_list)
      for subfig_idx,s in enumerate(self.states_list):
         plt.subplot(2,num_subfig_cols,1+subfig_idx)
         self.plot_observations(colors=colors,states_objs=[s])

         plt.subplot(2,num_subfig_cols,1+num_subfig_cols+subfig_idx)
         s.plot(colors_dict=colors)


class HSMM(HMM, ModelGibbsSampling):
   '''
   The HSMM class is a wrapper to package all the pieces of an HSMM:
       * HSMM internals, including distribution objects for
           - states
           - transitions
           - initial state
       * the main distributions that define the HSMM:
           - observations
           class :
               - durations
   When an HSMM is instantiated, it is a ``prior'' model object. Observation
   sequences can be added via the add_data(data_seq) method, making it a
   ``posterior'' model object and then the latent components (including all
   state sequences and parameters) can be resampled by calling the resample()
   method.
   '''

   def __init__(self,
                obs_distns,dur_distns,
                trunc=None,
                trans_distn=None,
                alpha=None,gamma=None,
                alpha_a_0=None,alpha_b_0=None,gamma_a_0=None,gamma_b_0=None,
                **kwargs):

      self.state_dim = len(obs_distns)
      self.trunc = trunc
      self.dur_distns = dur_distns

      assert (trans_distn is not None) ^ \
             (alpha is not None and gamma is not None) ^ \
             (alpha_a_0 is not None and alpha_b_0 is not None
              and gamma_a_0 is not None and gamma_b_0 is not None)
      if trans_distn is not None:
         self.trans_distn = trans_distn
      elif alpha is not None:
         # xun: [1] alpha, gamma => trans_distns. This is a HDP, same with old code
         self.trans_distn = transitions.HDPHSMMTransitions(
            state_dim=self.state_dim,
            alpha=alpha,gamma=gamma)
      else:
         # xun: [1.1] if we don't give alpha value, we can use alpha_a_0, alpha_b_0, gamma_a_0, gamma_b_0
         # instead, and Johonson's code can create a dur_distns
         # NOTE: this is new, not in old code
         self.trans_distn = transitions.HDPHSMMTransitionsConcResampling(
            state_dim=self.state_dim,
            alpha_a_0=alpha_a_0,alpha_b_0=alpha_b_0,
            gamma_a_0=gamma_a_0,gamma_b_0=gamma_b_0)

      # xun: [1.2] this wil call HMM to init state_distns using init_state_concentration value
      super(HSMM,self).__init__(obs_distns=obs_distns,trans_distn=self.trans_distn,**kwargs)

   def add_data(self,data,stateseq=None,censoring=True,**kwargs):
      self.states_list.append(states.HSMMStates(len(data),self.state_dim,self.obs_distns,self.dur_distns,
                                                self.trans_distn,self.init_state_distn,trunc=self.trunc,data=data,stateseq=stateseq,
                                                censoring=censoring))

   def resample_model(self):
      # resample durparams
      for state, distn in enumerate(self.dur_distns):
         distn.resample([s.durations[s.stateseq_norep == state] for s in self.states_list])

      # resample everything else an hmm does
      super(HSMM,self).resample_model()

   def generate(self,T,keep=True):
      tempstates = states.HSMMStates(T,self.state_dim,self.obs_distns,self.dur_distns,
                                     self.trans_distn,self.init_state_distn,trunc=self.trunc)
      return self._generate(tempstates,keep)

   ## parallel sampling
   def add_data_parallel(self,data_id):
      from pyhsmm import parallel
      self.add_data(parallel.alldata[data_id])
      self.states_list[-1].data_id = data_id

   def resample_model_parallel(self,numtoresample='all'):
      from pyhsmm import parallel
      if numtoresample == 'all':
         numtoresample = len(self.states_list)
      elif numtoresample == 'engines':
         numtoresample = len(parallel.dv)

      ### resample parameters locally
      self.trans_distn.resample([s.stateseq for s in self.states_list])
      self.init_state_distn.resample([s.stateseq[:1] for s in self.states_list])
      for state, (o,d) in enumerate(zip(self.obs_distns,self.dur_distns)):
         d.resample([s.durations[s.stateseq_norep == state] for s in self.states_list])
         o.resample([s.data[s.stateseq == state] for s in self.states_list])

      ### choose which sequences to resample
      states_to_resample = random.sample(self.states_list,numtoresample)

      ### resample states in parallel
      self._push_self_parallel(states_to_resample)
      self._build_states_parallel(states_to_resample)

      ### purge to prevent memory buildup
      parallel.c.purge_results('all')

   ### plotting

   def plot_durations(self,colors=None,states_objs=None):
      setbr = brewer2mpl.get_map('RdBu', 'diverging', 10, reverse = True )
      if colors is None:
         colors = setbr #self._get_colors()
      if states_objs is None:
         states_objs = self.states_list

      cmap = setbr #cm.get_cmap()
      used_states = self._get_used_states(states_objs)
      for state,d in enumerate(self.dur_distns):
         if state in used_states:
            d.plot(color=cmap(colors[state]),
                   data=[s.durations[s.stateseq_norep == state]
                         for s in states_objs])
      plt.title('Durations')
      plt.grid()

   """
    def plot(self,color=None):
        plt.gcf() #.set_size_inches((10,10))
        colors = self._get_colors()

        num_subfig_cols = len(self.states_list)
        for subfig_idx,s in enumerate(self.states_list):
            plt.subplot(3,num_subfig_cols,1+subfig_idx)
            self.plot_observations(colors=colors,states_objs=[s])

            plt.subplot(3,num_subfig_cols,1+num_subfig_cols+subfig_idx)
            s.plot(colors_dict=colors)

            plt.subplot(3,num_subfig_cols,1+2*num_subfig_cols+subfig_idx)
            self.plot_durations(colors=colors,states_objs=[s])
    """

   def prerun(self,color=None,noplot=False,plotoptions=[0,0,0,0]):  #was 'jet'
      setbr = brewer2mpl.get_map('RdBu', 'diverging', 10, reverse=True)
      assert len(self.obs_distns) != 0
      assert len(set([type(o) for o in self.obs_distns])) == 1, 'plot can only be used when all observation distributions are the same'

      results_list = []
      states_dict_list = []
      used_states = reduce(set.union,[set(s.stateseq_norep) for s in self.states_list])

      for subfig_idx,s in enumerate(self.states_list):
         """ Observation Distributions """
         x = np.array(self.obs_distns)
         final_states = list(used_states)
         get_states =  []
         get_means0 =  []
         get_means0_round =  []
         get_means1 =  []
         get_means1_round =  []
         np.set_printoptions(precision=2, suppress = True)
         for i in range(len(x)):
            for j in range(len(final_states)):
               if i == final_states[j]:
                  get_states.append(final_states[j])
                  get_means0.append(float(str(x[i].mu[0])[:4]))
                  get_means0_round.append(np.round(float(str(x[i].mu[0])[:4]))) #need for dict
                  get_means1.append(float(str(x[i].mu[1])[:4]))
                  get_means1_round.append(np.round(float(str(x[i].mu[1])[:4])))
         '''pseudo-trace of individual subject'''
         state_means_dict = zip(get_means0_round, get_means1_round)
         subject_dict = dict(zip(get_states,state_means_dict ))
         sub_state_seq = []
         sub_state_seq_X =  []
         sub_state_seq_Y =  []
         for i in range(len(s.stateseq)):
            sub_state_seq.append(subject_dict[s.stateseq[i]])
         for i in range(len(sub_state_seq)):
            sub_state_seq_X.append(sub_state_seq[i][0])
            sub_state_seq_Y.append(sub_state_seq[i][1])
         states_dict_list.append(subject_dict)
         results_list.append(self.print_results(subfig_idx,s, x, used_states, self.dur_distns, self.trans_distn))
      return states_dict_list,results_list

   def print_results(self,i, s, x, used_states, dur_distns, trans_distn):
      results =   '\n-----------------------------------------------------\n'
      results += 'Simulated States for couple indexed as %s within this cluster' %  i
      results += '\n-----------------------------------------------------\n\n'

      state_describe_idx = []
      plot_states = []; plot_state_info =  []
      get_means = []; get_means_split = []
      for state,d in enumerate(dur_distns):
         if state in s.stateseq_norep:
            plot_states.append(state)
            state_describe_idx.append(len(plot_states))
            plot_state_info.append(str(d))
      for i in range(len(plot_state_info)):
         get_means_split.append(plot_state_info[i].split('='))
      for i in range(len(get_means_split)):
         get_means.append(float(get_means_split[i][2].rstrip(')')))
      mu_list_dim1 =   []
      mu_list_dim2 =   []
      cov_list_dim1 =  []
      cov_list_dim2 =  []
      cov_list_cov =   []
      np.set_printoptions(precision=3)
      np.set_printoptions(suppress = True)
      final_states = list(used_states)
      SumStatesConstruction = {2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8,10:9}  #sum as key; state as value
      #StatesConstruction = {1:2,2:3,3:4,4:5,5:6,6:7,7:8,8:9,9:10}  #States as key, sum as value

      for i in range(len(x)): #max of n states
         tracker = 0
         for j in range(len(final_states)):  #states used
            if i == final_states[j]:
               x_mu0 = x[i].mu[0]              #dimen 1: Male Affect
               x_mu1 = x[i].mu[1]              #dimen 2: Female Affect
               x_sigma1 = x[i].sigma[0][0]       #dimen 1
               x_sigma2 = x[i].sigma[1][1]       #dimen 2
               x_sigma_cov = x[i].sigma[0][1]  #cov; note location
               mu_list_dim1.append(x_mu0)
               mu_list_dim2.append(x_mu1)
               cov_list_dim1.append(x_sigma1)
               cov_list_dim2.append(x_sigma2)
               cov_list_cov.append(x_sigma_cov)
               #x_mu0States = round(x[i].mu[0])         #dimen 1 Male Affect rounded
               #x_mu1States = round(x[i].mu[1])         #dimen 2 Female Affect rounded
               ########################################################################################

      mu_cov_table = zip(
                                 state_describe_idx,
                                 plot_states,
                                 get_means,
                                 mu_list_dim1,
                                 mu_list_dim2,
                                 cov_list_dim1,
                                 cov_list_dim2,
                                 cov_list_cov)

      import nicetext
      from  collections import  Counter,  OrderedDict

      headers =  'Index', 'State', 'Dur_Lamda', 'Male_Avg',' Female_Avg', 'Male_Var', 'Female_Var', 'Cov'
      results += nicetext.SimpleTable(
         mu_cov_table,
         headers,
         title = 'Couple Ratings By State: Duration, Means, Variance, CoVariance' ,
         fmt={'data_fmt':['%g', '%g', '%1.3f', '%1.3f', '%1.3f', '%1.3f', '%1.3f', '%1.3f']}).as_text()
      results += '\n'
      possible_states = list(used_states)

      ### binary sequence of states used
      couple_states = []
      for i in range(len(possible_states)):
         if possible_states[i] in plot_states:
            couple_states.append(1)
         else:
            couple_states.append(0)

      state_seq = []
      state_dur = []
      for i in range(len(s.durations)):
         state_seq.append(s.stateseq_norep[i])
         state_dur.append(s.durations[i])
         cum_dur = np.cumsum(state_dur)
         cum_count = cum_dur
      state_w_dur =  zip(state_seq, state_dur, cum_dur)
      results += 'Estimated interaction had %d occurrences' % len(state_w_dur)
      headers =  'State', 'Duration', 'Cumulative'
      results += nicetext.SimpleTable(
         state_w_dur, headers,
         title='\nGenerated state sequence with duration',
         fmt={'data_fmt':['%g','%1.1f', '%g']}).as_text()
      results += '\n'

      ##### extract raw data by sex
      #sim_male_affect    = []
      #sim_female_affect = []
      #mat_States = []
      #for i in range(len(s.data)):
         #sim_male_affect.append(s.data[i,0])
         #sim_female_affect.append(s.data[i,1])
         #generateStates =  s.data[i,0] + s.data[i,1]
         #mat_States.append(SumStatesConstruction[generateStates])
      #sim_affect_States = zip(sim_male_affect,sim_female_affect,mat_States)
      #results += 'Simulated Affect with reduced States (low is positive)'
      #results += '\n'
      #for j in sim_affect_States:
         #results += '%g, %g, %g' % (j[0],j[1],j[2])
         #results += '\n'
      #results += '\n'


      ### frequency count os state used
      state_tallies = []
      get_state_freq =  Counter(state_seq)
      for i in range(len(possible_states)):
         if get_state_freq.has_key(possible_states[i]):
            state_tallies.append(get_state_freq[possible_states[i]])
         else:
            state_tallies.append(0)
      ### dump couple unique state indicators and frequency;
      ### possible use in cluster analysis
      #reduce to relevant r,c; relevant with single couple analysis
      results += " trans_distn.A\n"
      x_yA = trans_distn.A
      x_yA_tbl = nicetext.SimpleTable(x_yA,fmt={'data_fmt':["%1.3f"]}).as_text()
      results += x_yA_tbl

      results += " trans_distn.fullA\n"
      x_yfullA = trans_distn.fullA
      x_yfullA_tbl = nicetext.SimpleTable(x_yfullA,fmt={'data_fmt':["%1.3f"]}).as_text()
      results += x_yfullA_tbl

      results += " trans_matrix.A\n"
      trans_matrixA =  x_yA[final_states][:, final_states]  #trans_matrix is the pop level matrix
      trans_matrixA_tbl = nicetext.SimpleTable(trans_matrixA,fmt={'data_fmt':["%1.3f"]}).as_text()
      results += trans_matrixA_tbl

      results += " trans_matrix.fullA\n"
      trans_matrixfullA =  x_yfullA[final_states][:, final_states]  #trans_matrix is the pop level matrix
      trans_matrixfullA_tbl = nicetext.SimpleTable(trans_matrixfullA,fmt={'data_fmt':["%1.3f"]}).as_text()
      results += trans_matrixfullA_tbl

      ''' create unique couple matrix from the population level trans_matrix: A '''
      results += " couple trans_matrix.A\n"
      couple_matrixA = x_yA[plot_states][:, plot_states]
      coupleA_tbl = nicetext.SimpleTable(couple_matrixA,fmt={'data_fmt':["%1.3f"]}).as_text()
      results += coupleA_tbl

      statelen = list(range(10))
      trans_diff =  list(set(statelen) - set(plot_states))  #states used relative to max
      results += 'States (from max possible) not used by this couple: %s' % (trans_diff)
      reduce_couple_matrixA = x_yA #complete_matrix
      for i in range(len(trans_diff)):
         nxt_state =  trans_diff[i]
         reduce_couple_matrixA[nxt_state] = 0.0      #rows
      for i in range(len(trans_diff)):
         nxt_state =  trans_diff[i]
         reduce_couple_matrixA[:,  nxt_state] = 0.0  # columns

      results += " couple trans reduce.A\n"
      reduce_coupleA_tbl = nicetext.SimpleTable(reduce_couple_matrixA,fmt={'data_fmt':["%1.3f"]}).as_text()
      results += reduce_coupleA_tbl

      ################ full A begin #########################
      ''' create unique couple matrix from the population level trans_matrix: full A'''
      results += " couple trans.fullA\n"
      couple_matrixfullA = x_yfullA[plot_states][:, plot_states]
      couplefullA_tbl = nicetext.SimpleTable(couple_matrixfullA,fmt={'data_fmt':["%1.3f"]}).as_text()
      results += couplefullA_tbl

      reduce_couple_matrixfullA = x_yfullA #complete_matrix
      for i in range(len(trans_diff)):
         nxt_state =  trans_diff[i]
         reduce_couple_matrixfullA[nxt_state] = 0.0      #rows
      for i in range(len(trans_diff)):
         nxt_state =  trans_diff[i]
         reduce_couple_matrixfullA[:,  nxt_state] = 0.0  # columns

      results += " couple trans reduce.fullA\n"
      reduce_couplefullA_tbl = nicetext.SimpleTable(reduce_couple_matrixfullA,fmt={'data_fmt':["%1.3f"]}).as_text()
      results += reduce_couplefullA_tbl

      return results

   def selected_plot(self,selections,title,color=None): #('jet')):
      setbr= brewer2mpl.get_map('RdBu', 'diverging', 10, reverse=True)
      cmaps = setbr.get_mpl_colormap(N=20, gamma=2.0)  #n state max
      assert len(self.obs_distns) != 0
      assert len(set([type(o) for o in self.obs_distns])) == 1, \
             'plot can only be used when all observation distributions are the same'

      # set up figure and state-color mapping dict
      fig = plt.gcf()
      fig.set_size_inches((20, 20))
      state_colors = {}
      if color== None:
         cmap = cmaps  #cm.get_cmap(color)
      else:
         cmap = cm.get_cmap()
      used_states = reduce(set.union,[set(s.stateseq_norep) for s in self.states_list])

      for state, o in enumerate(self.obs_distns):  #this determines the color shading
         if state in used_states:
            state_colors[state] = (o.mu[0] + o.mu[1]) /10  # could put in States value here; round

      num_subfig_cols = len(self.states_list)
      num_subfig_rows = len(selections)


      for subfig_idx,s in enumerate(self.states_list):
         """ Observation Distributions """
         selected_fig_idx = 0

         if 0 in selections:
            ax1 = plt.subplot(num_subfig_rows,num_subfig_cols,1+selected_fig_idx*num_subfig_cols+subfig_idx)
            selected_fig_idx += 1
            self.obs_distns[0]._plot_setup(self.obs_distns)


            for state,o in enumerate(self.obs_distns):
               if state in s.stateseq_norep:
                  o.plot(color=cmap(state_colors[state]),
                         data=s.data[s.stateseq == state] if s.data is not None else None,
                         cmap=cmap)
            #frame  = ax1.get_frame()  #matplotlib 1.1.1
            frame  = ax1.patch      #matplotlib 1.2.1
            frame.set_facecolor('0.90')
            plt.xlim ((.5, 5.5))
            plt.ylim ((.5, 5.5))
            plt.grid()
            plt.title('Observation Distributions')

         x = np.array(self.obs_distns)
         final_states = list(used_states)
         get_states =  []
         get_means0 =  []
         get_means0_round =  []
         get_means1 =  []
         get_means1_round =  []
         np.set_printoptions(precision=2, suppress = True)
         for i in range(len(x)):
            for j in range(len(final_states)):
               if i == final_states[j]:
                  get_states.append(final_states[j])
                  get_means0.append(float(str(x[i].mu[0])[:4]))
                  get_means0_round.append(np.round(float(str(x[i].mu[0])[:4]))) #need for dict
                  get_means1.append(float(str(x[i].mu[1])[:4]))
                  get_means1_round.append(np.round(float(str(x[i].mu[1])[:4])))
         state_means = zip(get_states, get_means0, get_means1)

         ''' State Sequence '''
         if 1 in selections:
            ax2 = plt.subplot(num_subfig_rows,num_subfig_cols,1+selected_fig_idx*num_subfig_cols+subfig_idx)
            selected_fig_idx += 1
            s.plot(colors_dict=state_colors,cmap=cmap)
            plt.title('State Sequence')

         ''' Duration Distributions '''
         if 2 in selections:
            ax3 = plt.subplot(num_subfig_rows,num_subfig_cols,1+selected_fig_idx*num_subfig_cols+subfig_idx)
            selected_fig_idx += 1
            for state,d in enumerate(self.dur_distns):
               if state in s.stateseq_norep:
                  d.plot(color=cmap(state_colors[state]),data=s.durations[s.stateseq_norep == state])
            #llines = ax3.get_lines() #matplotlib 1.1.1
            llines = ax3.patch  #matplotlib 1.3.1
            #if matplotlib.__version__ != 1.3.1:
               #lines = ax3.get_lines() #matplotlib 1.1.1
            #else:
               #llines = ax3.patch      #matplotlib 1.3.1
            plt.setp(llines, linewidth=2.5)
            #frame  = ax3.get_frame()#matplotlib 1.1.1
            frame  = ax3.patch  #matplotlib 1.3.1
            frame.set_facecolor('0.90')
            plt.xlim((0,s.durations.max()*1.1))
            plt.ylim(ymax = 1)
            plt.title('Durations')

         plot_states = []
         plot_state_info =  []
         state_describe_idx = []
         for state,d in enumerate(self.dur_distns):
            if state in s.stateseq_norep:
               plot_states.append(state)
               state_describe_idx.append(len(plot_states))
               plot_state_info.append(str(d))
         get_means_split = []
         get_means = []
         for i in range(len(plot_state_info)):
            get_means_split.append(plot_state_info[i].split('='))
         for i in range(len(get_means_split)):
            get_means.append(float(get_means_split[i][2].rstrip(')')))
         state_dur = zip(state_describe_idx, plot_states, get_means, get_means0, get_means1)
         state_describe = []
         for i in range(len(state_dur)):
            state_describe.append(list(state_dur[i]))

         '''plot pseudo-trace of individual subject'''
         if 3 in selections:
            ax4 = plt.subplot(num_subfig_rows,num_subfig_cols,1+selected_fig_idx*num_subfig_cols+subfig_idx)
            selected_fig_idx += 1
            state_means_dict = zip(get_means0_round, get_means1_round)
            subject_dict = {}
            subject_dict = dict(zip(get_states,state_means_dict ))
            sub_state_seq = []
            sub_state_seq_X =  []
            sub_state_seq_Y =  []

            for i in range(len(s.stateseq)):
               sub_state_seq.append(subject_dict[s.stateseq[i]])
            for i in range(len(sub_state_seq)):
               sub_state_seq_X.append(sub_state_seq[i][0])
               sub_state_seq_Y.append(sub_state_seq[i][1])
            plt.plot(sub_state_seq_X, label = 'M', linewidth=2, color= '#377eb8' )
            plt.plot(sub_state_seq_Y, label = 'F', linewidth=2, color= '#e41a1c')
            plt.title('State Trace')
            plt.legend()
            plt.xlim((0, 720))
            plt.ylim((1, 5))

         '''plot original data'''
         if 4 in selections:
            ax4 = plt.subplot(num_subfig_rows,num_subfig_cols,1+selected_fig_idx*num_subfig_cols+subfig_idx)
            selected_fig_idx += 1
            plt.plot(s.data[:,0], label = 'M', linewidth=2, color= '#377eb8' )
            plt.plot(s.data[:,1], label = 'F', linewidth=2, color= '#e41a1c' )
            plt.title('Original Data')
            plt.legend()
            plt.xlim((0, 720))
            plt.ylim((1, 5))

   def plot(self,color= None):  # jet
      setbr = brewer2mpl.get_map('RdBu', 'diverging', 10, reverse=True)
      assert len(self.obs_distns) != 0
      assert len(set([type(o) for o in self.obs_distns])) == 1, \
             'plot can only be used when all observation distributions are the same'

      # set up figure and state-color mapping dict
      fig = plt.gcf()
      fig.set_size_inches((20, 20))
      state_colors = {}
      if color== None:
         cmap = setbr #cm.get_cmap()
      else:
         cmap = cm.get_cmap(color)
      used_states = reduce(set.union,[set(s.stateseq_norep) for s in self.states_list])

      for state, o in enumerate(self.obs_distns):
         if state in used_states:
            state_colors[state] = (o.mu[0] + o.mu[1]) /10
      num_subfig_cols = len(self.states_list)

      for subfig_idx,s in enumerate(self.states_list):
         """ Observation Distributions """
         ### plot the current observation distributions (and obs, if given)
         ### 1 ###
         ax1 = plt.subplot(4,num_subfig_cols,1+subfig_idx)
         self.obs_distns[0]._plot_setup(self.obs_distns)

         for state,o in enumerate(self.obs_distns):
            if state in s.stateseq_norep:
               o.plot(color=cmap(state_colors[state]),
                      data=s.data[s.stateseq == state] if s.data is not None else None)

         x = np.array(self.obs_distns)
         final_states = list(used_states)
         get_states =  []
         get_means0 =  []
         get_means0_round =  []
         get_means1 =  []
         get_means1_round =  []
         np.set_printoptions(precision=2, suppress = True)
         for i in range(len(x)):
            for j in range(len(final_states)):
               if i == final_states[j]:
                  get_states.append(final_states[j])
                  get_means0.append(float(str(x[i].mu[0])[:4]))
                  get_means0_round.append(np.round(float(str(x[i].mu[0])[:4]))) #need for dict
                  get_means1.append(float(str(x[i].mu[1])[:4]))
                  get_means1_round.append(np.round(float(str(x[i].mu[1])[:4])))
         state_means = zip(get_states, get_means0, get_means1)

         #frame  = ax1.get_frame()   #matplotlib 1.1.1
         frame  = ax1.patch          #matplotlib 1.2.1
         frame.set_facecolor('0.90')
         plt.xlim ((1, 5))
         plt.ylim ((1, 5))
         #plt.legend()
         plt.title('Observation Distributions')

         ''' State Sequence '''
         ## plot the state sequence
         ### 2 ###
         ax2 =  plt.subplot(4,num_subfig_cols,1+num_subfig_cols+subfig_idx)
         s.plot(colors_dict=state_colors,cmap=cmap)
         #plt.legend()
         plt.title('State Sequence')

         ''' Duration Distributions '''
         ## plot the current duration distributions
         ### 3 ###
         ax3 = plt.subplot(4,num_subfig_cols,1+2*num_subfig_cols+subfig_idx)
         plot_states = []
         plot_state_info =  []
         state_describe_idx = []
         for state,d in enumerate(self.dur_distns):
            if state in s.stateseq_norep:
               d.plot(color=cmap(state_colors[state]),data=s.durations[s.stateseq_norep == state])
               plot_states.append(state)
               state_describe_idx.append(len(plot_states))
               plot_state_info.append(str(d))
         get_means_split = []
         get_means = []
         for i in range(len(plot_state_info)):
            get_means_split.append(plot_state_info[i].split('='))
         for i in range(len(get_means_split)):
            get_means.append(float(get_means_split[i][2].rstrip(')')))
         state_dur = zip(state_describe_idx, plot_states, get_means, get_means0, get_means1)
         state_describe = []
         for i in range(len(state_dur)):
            state_describe.append(list(state_dur[i]))
         #ax3.legend(state_describe,  bbox_to_anchor=(0., -.345, 1., .102), loc=3,
                  #ncol=6, mode="expand", borderaxespad=0.,
                  #shadow = True,  title="Index:State:Averaged Duration:Rating0:Rating1")
         #ax3.get_legend().get_title().set_color("black")
         #for i in range(len(state_describe)):
            #ax3.get_legend().legendHandles[i].set_linewidth(4.5)
         llines = ax3.get_lines()
         plt.setp(llines, linewidth=2.5)
         #frame  = ax3.get_frame()
         frame  = ax3.patch          #matplotlib 1.3.1
         frame.set_facecolor('0.90')
         plt.xlim((0,s.durations.max()*1.1))
         plt.title('Durations')
         ######################################################################################
         '''plot pseudo-trace of individual subject'''
         ax4 = plt.subplot(4,num_subfig_cols,1+3*num_subfig_cols+subfig_idx)
         state_means_dict = zip(get_means0_round, get_means1_round)
         subject_dict = {}
         subject_dict = dict(zip(get_states,state_means_dict ))
         sub_state_seq = []
         sub_state_seq_X =  []
         sub_state_seq_Y =  []

         ### plot general representative sequence
         for i in range(len(s.stateseq)):
            sub_state_seq.append(subject_dict[s.stateseq[i]])
         for i in range(len(sub_state_seq)):
            sub_state_seq_X.append(sub_state_seq[i][0])
            sub_state_seq_Y.append(sub_state_seq[i][1])

         plt.plot(sub_state_seq_X, label = 'M', linewidth=2, color= '#377eb8' )
         plt.plot(sub_state_seq_Y, label = 'F', linewidth=2, color= '#e41a1c' )
         plt.title('State Trace')
         plt.legend()
         plt.xlim((0, 720))
         plt.ylim((1, 5))
         #plt.savefig('.png')

         ####################################################################################
         print
         print 'Selected states:', used_states                                         #wag
         print ('Number of selected states: %s' % (len(used_states)))
         print
         ####################################################################################

         ### wag - do not erase #######################################################
         x = np.array(self.obs_distns)
         #writer.writerows(state_describe)

         ### do not erase ###
         #headers = ' Index',' State',' lambda',' Dimen_1 Avg',' Dimen_2 Avg'
         #print nicetext.SimpleTable(state_describe,headers,
                              #title = 'Couple State Process Discriptors' ,
                              #fmt={'data_fmt':['%g','%g','%1.3f','%1.3f','%1.3f']}).as_text()

         #print 'Means and Covaranice Matrices for Final States'
         #generate table
         mu_list_dim1 =   []
         mu_list_dim2 =   []
         cov_list_dim1 =  []
         cov_list_dim2 =  []
         cov_list_cov =   []
         np.set_printoptions(precision=3)
         np.set_printoptions(suppress = True)
         final_states = list(used_states)
         for i in range(len(x)):
            for j in range(len(final_states)):
               if i == final_states[j]:
                  x_mu0 = x[i].mu[0]              #dimen 1 Male Affect
                  x_mu1 = x[i].mu[1]              #dimen 2 Female Affect
                  x_sigma1 = x[i].sigma[0][0]       #dimen 1
                  x_sigma2 = x[i].sigma[1][1]       #dimen 2
                  x_sigma_cov = x[i].sigma[0][1]  #cov; note location
                  mu_list_dim1.append(x_mu0)
                  mu_list_dim2.append(x_mu1)
                  cov_list_dim1.append(x_sigma1)
                  cov_list_dim2.append(x_sigma2)
                  cov_list_cov.append(x_sigma_cov)

         mu_cov_table = zip(state_describe_idx,
                            plot_states,
                            get_means,
                            mu_list_dim1,
                            mu_list_dim2,
                            cov_list_dim1,
                            cov_list_dim2,
                            cov_list_cov)
         import nicetext
         headers =  'Index', 'State', 'Dur(Lamda)', 'Male_Avg',' Female_Avg', 'Male_Var', 'Female_Var', 'Cov'
         print nicetext.SimpleTable(
            mu_cov_table,
            headers,
            title = 'Couple Ratings By State: Means, Variance, CoVariance' ,
            fmt={'data_fmt':['%g', '%g', '%1.3f', '%1.3f', '%1.3f', '%1.3f', '%1.3f', '%1.3f']}).as_text()
         print
         possible_states = list(used_states)
         #### binary sequence of states used
         couple_states = []
         for i in range(len(possible_states)):
            if possible_states[i] in plot_states:
               couple_states.append(1)
            else:
               couple_states.append(0)

         state_seq = []
         state_dur = []
         for i in range(len(s.durations)):
            state_seq.append(s.stateseq_norep[i])
            state_dur.append(s.durations[i])
         state_w_dur =  zip(state_seq, state_dur)
         print ('Estimated interaction had %d occurences.' % len(state_w_dur))
         headers =  'State', 'Duration'
         print nicetext.SimpleTable(state_w_dur, headers,
                                    title='Generated state sequence with durations',
                                    fmt={'data_fmt':['%g','%1.1f']}).as_text()

         print
         ### frequency count os state used
         #state_tallies = []
         #get_state_freq =  Counter(state_seq)
         #for i in range(len(possible_states)):
            #if get_state_freq.has_key(possible_states[i]):
               #state_tallies.append(get_state_freq[possible_states[i]])
            #else:
               #state_tallies.append(0)
         ### dump couple unique state indicators and frequency;
         ### possible use in cluster analysis
         #with open('output/coupleStateIndicate.txt', 'a') as ci:
            #ci.write(str(couple_states))
            #ci.write('\n')

         #with open('output/coupleStateFreq.txt', 'a') as ct:
            #ct.write(str(state_tallies))
            #ct.write('\n')

         #reduce to relevant r,c; relevant with single couple analysis
         x_yA = self.trans_distn.A
         x_yA_tbl = nicetext.SimpleTable(x_yA,fmt={'data_fmt':["%1.3f"]})
         with open('output/A.csv','a') as x_yAfile:
            x_yAfile.write( x_yA_tbl.as_csv() )
            x_yAfile.write('\t')

         x_yfullA = self.trans_distn.fullA
         x_yfullA_tbl = nicetext.SimpleTable(x_yfullA,fmt={'data_fmt':["%1.3f"]})
         with open('output/fullA.csv','a') as x_yfullAfile:
            x_yfullAfile.write( x_yfullA_tbl.as_csv() )
            x_yfullAfile.write('\t')

         #________________________________________
         #complete_matrix =  x_yA
         #complete_tbl = nicetext.SimpleTable(complete_matrix,fmt={'data_fmt':["%1.3f"]})
         #with open('output/completeTrans.csv','a') as complm:
            #complm.write( complete_tbl.as_csv() )
            #complm.write('\t')
         #_________________________________________
         trans_matrixA =  x_yA[final_states][:, final_states]  #trans_matrix is the pop level matrix
         trans_matrixA_tbl = nicetext.SimpleTable(trans_matrixA,fmt={'data_fmt':["%1.3f"]})
         with open('output/trans_matrixA.csv','a') as transA:
            transA.write(trans_matrixA_tbl.as_csv() )
            transA.write('\t')

         trans_matrixfullA =  x_yfullA[final_states][:, final_states]  #trans_matrix is the pop level matrix
         trans_matrixfullA_tbl = nicetext.SimpleTable(trans_matrixfullA,fmt={'data_fmt':["%1.3f"]})
         with open('output/trans_matrixfullA.csv','a') as transfullA:
            transfullA.write(trans_matrixfullA_tbl.as_csv() )
            transfullA.write('\t')


         ''' create unique couple matrix from the population level trans_matrix: A '''
         couple_matrixA = x_yA[plot_states][:, plot_states]
         coupleA_tbl = nicetext.SimpleTable(couple_matrixA,fmt={'data_fmt':["%1.3f"]})
         with open('output/coupleTransA.csv','a') as coupmA:
            coupmA.write( coupleA_tbl.as_csv() )
            coupmA.write('\t')

         statelen = list(range(20))
         trans_diff =  list(set(statelen) - set(plot_states))  #states used relative to max
         print ('States (from max) not used by this couple: %s' % (trans_diff))
         reduce_couple_matrixA = x_yA #complete_matrix
         for i in range(len(trans_diff)):
            nxt_state =  trans_diff[i]
            reduce_couple_matrixA[nxt_state] = 0.0      #rows
         for i in range(len(trans_diff)):
            nxt_state =  trans_diff[i]
            reduce_couple_matrixA[:,  nxt_state] = 0.0  # columns

         reduce_coupleA_tbl = nicetext.SimpleTable(reduce_couple_matrixA,fmt={'data_fmt':["%1.3f"]})
         with open('output/coupleTransReduceA.csv','a') as coup_redA:
            coup_redA.write( reduce_coupleA_tbl.as_csv() )
            coup_redA.write('\t')

         ################ full A begin #########################
         ''' create unique couple matrix from the population level trans_matrix: full A'''
         couple_matrixfullA = x_yfullA[plot_states][:, plot_states]
         couplefullA_tbl = nicetext.SimpleTable(couple_matrixfullA,fmt={'data_fmt':["%1.3f"]})
         with open('output/coupleTransfullA.csv','a') as coupmfullA:
            coupmfullA.write( couplefullA_tbl.as_csv() )
            coupmfullA.write('\t')

         #statelen = list(range(20))
         #trans_diff =  list(set(statelen) - set(plot_states))  #states used relative to max
         #print ('States not used by this couple: %s' % (trans_diff))
         reduce_couple_matrixfullA = x_yfullA #complete_matrix
         for i in range(len(trans_diff)):
            nxt_state =  trans_diff[i]
            reduce_couple_matrixfullA[nxt_state] = 0.0      #rows
         for i in range(len(trans_diff)):
            nxt_state =  trans_diff[i]
            reduce_couple_matrixfullA[:,  nxt_state] = 0.0  # columns

         reduce_couplefullA_tbl = nicetext.SimpleTable(reduce_couple_matrixfullA,fmt={'data_fmt':["%1.3f"]})
         with open('output/coupleTransReducefullA.csv','a') as coup_redfullA:
            coup_redfullA.write( reduce_couplefullA_tbl.as_csv() )
            coup_redfullA.write('\t')
         ################ full A end #########################
         #for i in range(len(trans_diff)):
            #delete_row
            #if reduce_couple_matrix[i] is trans_diff[i]:
            #reduce_couple_matrix[i] = 0

         #print
         #tbl_txt = nicetext.SimpleTable(trans_matrix,
                        #title = 'Transition Matrix; no repetition',
                        #fmt={'data_fmt':["%1.3f"]}).as_text()

         #### note these could be saved to a file for ease of publication/analysis
         ##print
         #tbl_csv = nicetext.SimpleTable(trans_matrix,
                        #title = 'Transition Matrix; no repetition',
                        #fmt={'data_fmt':["%1.3f"]}).as_csv()
         ##print
         #tbl_tex = nicetext.SimpleTable(trans_matrix,
                        #title = 'Transition Matrix; no repetition',
                        #fmt={'data_fmt':["%1.3f"]}).as_latex_tabular()

         ''' dump population level transition matrix '''
         #tbl = nicetext.SimpleTable(trans_matrix,
                              #title = 'Transition Matrix',
                              #fmt={'data_fmt':["%1.3f"]})

         ##with open('coupleTrans.tex','w') as fh: fh.write( tbl_tex.as_latex_tabular() )

         #with open('output/populationTrans.csv','w') as fh:
            #fh.write( tbl.as_csv() )

         #with open('coupleTrans.txt','w') as fh: fh.write( tbl_txt.as_text() )

         #print   #not sure what this is; number of occurences are not consistent
         #count_matrix =  self.trans_distn.m[final_states][:, final_states]
         #print nicetext.SimpleTable(count_matrix, title = 'Frequency Count Matrix',
                                       #fmt={'data_fmt':["%d"]}).as_csv()

         print '%' * 80
         print
         ######################################################################################
         #### plot pseudo-trace of individual subject ###
         #state_means_dict = zip(get_means0_round, get_means1_round)
         #subject_dict = {}
         #subject_dict = dict(zip(get_states,state_means_dict ))
         ##print subject_dict
         #print
         #sub_state_seq = []
         #sub_state_seq_X =  []
         #sub_state_seq_Y =  []

         #### plot general representative sequence
         ##sub_dur = np.sum(state_dur) #
         #for i in range(len(s.stateseq)):
            #sub_state_seq.append(subject_dict[s.stateseq[i]])
         #for i in range(len(sub_state_seq)):
            #sub_state_seq_X.append(sub_state_seq[i][0])
            #sub_state_seq_Y.append(sub_state_seq[i][1])

         #plot_seq(sub_state_seq_X,sub_state_seq_Y)
         ##plt.figure(figsize=(15, 10))
         ##plt.plot(sub_state_seq_X, label ='M', color='#7570b3' )
         ##plt.plot(sub_state_seq_Y, label = 'D2', color='#d95f02')
         ##plt.title('Plausible Sequence of States for High Satisfaction Group: subGroup 2')
         ##plt.legend()
         ##plt.xlim((0, 720))
         ##plt.ylim((0, 8))
         ##plt.figure(figsize=(15, 10))
         ## TODO add a figure legend

   def plot_summary(self,color=None):
      # if there are too many state sequences in states_list, make an
      # alternative plot
      raise NotImplementedError

   def log_likelihood(self,data,trunc=None):
      warn('untested')
      T = len(data)
      if trunc is None:
         trunc = T
      # make a temporary states object to make sure no data gets clobbered
      s = states.HSMMStates(T,self.state_dim,self.obs_distns,self.dur_distns,
                            self.trans_distn,self.init_state_distn,trunc=trunc)
      s.obs = data
      possible_durations = np.arange(1,trunc + 1,dtype=np.float64)
      aDl = np.zeros((T,self.state_dim))
      aDsl = np.zeros((T,self.state_dim))
      for idx, dur_distn in enumerate(self.dur_distns):
         aDl[:,idx] = dur_distn.log_pmf(possible_durations)
         aDsl[:,idx] = dur_distn.log_sf(possible_durations)

      s.aBl = s.get_aBl(data)
      betal, betastarl = s._messages_backwards(np.log(self.transition_distn.A),aDl,aDsl,trunc)
      return np.logaddexp.reduce(np.log(self.initial_distn.pi_0) + betastarl[0])

   def EM_step(self):
      raise NotImplementedError # TODO
