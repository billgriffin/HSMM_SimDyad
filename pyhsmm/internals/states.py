import numpy as np
from numpy import newaxis as na
from numpy.random import random
import scipy.weave

from ..util.stats import sample_discrete, sample_discrete_from_log
from ..util import general as util # perhaps a confusing name :P

class HSMMStatesPython(object):
    '''
    HSMM states distribution class. Connects the whole model.

    Parameters include:

    T
    state_dim
    obs_distns
    dur_distns
    transition_distn
    initial_distn
    trunc

    stateseq
    durations
    stateseq_norep
    '''

    # these are convenient
    durations = None
    stateseq_norep = None

    def __init__(self,T,state_dim,obs_distns,dur_distns,transition_distn,initial_distn,
    stateseq=None,trunc=None,data=None,censoring=True,
    initialize_from_prior=True,**kwargs):
        self.T = T
        self.state_dim = state_dim
        self.obs_distns = obs_distns
        self.dur_distns = dur_distns
        self.transition_distn = transition_distn
        self.initial_distn = initial_distn
        self.trunc = T if trunc is None else trunc
        self.data = data
        self.censoring = censoring

        # this arg is for initialization heuristics which may pre-determine the
        # state sequence
        if stateseq is not None:
            self.stateseq = stateseq
            # gather durations and stateseq_norep
            self.stateseq_norep, self.durations = util.rle(stateseq)
        else:
            if data is not None and not initialize_from_prior:
                self.resample()
            else:
                self.generate_states()

    def generate(self):
        self.generate_states()
        return self.generate_obs()

    def generate_obs(self):
        obs = []
        for state,dur in zip(self.stateseq_norep,self.durations):
            obs.append(self.obs_distns[state].rvs(size=int(dur)))
        return np.concatenate(obs)[:self.T] # censoring

    def generate_states(self):
        # TODO TODO make censoring work?
        idx = 0
        nextstate_distr = self.initial_distn.pi_0
        A = self.transition_distn.A

        stateseq = -1*np.ones(self.T,dtype=np.int32)
        stateseq_norep = []
        durations = []

        while idx < self.T:
            # sample a state
            state = sample_discrete(nextstate_distr)
            # sample a duration for that state
            duration = self.dur_distns[state].rvs()
            # save everything
            stateseq_norep.append(state)
            durations.append(duration)
            stateseq[idx:idx+duration] = state # this can run off the end, that's okay
            # set up next state distribution
            nextstate_distr = A[state,]
            # update index
            idx += duration

        self.stateseq_norep = np.array(stateseq_norep,dtype=np.int32)
        self.durations = np.array(durations,dtype=np.int32)
        self.stateseq = stateseq

        # NOTE self.durations.sum() >= self.T since self.T is the censored
        # length

        assert len(self.stateseq_norep) == len(self.durations)
        assert (self.stateseq >= 0).all()

    def resample(self):
        assert self.data is not None
        data= self.data

        # generate duration pmf and sf values
        # generate and cache iid likelihood values, used in cumulative_likelihood functions
        possible_durations = np.arange(1,self.T + 1,dtype=np.float64)
        aDl = np.zeros((self.T,self.state_dim))
        aDsl = np.zeros((self.T,self.state_dim))
        self.aBl = self.get_aBl(data)
        for idx, dur_distn in enumerate(self.dur_distns):
            aDl[:,idx] = dur_distn.log_pmf(possible_durations)
            aDsl[:,idx] = dur_distn.log_sf(possible_durations)
        # run backwards message passing
        betal, betastarl = self._messages_backwards(np.log(self.transition_distn.A),aDl,aDsl,self.trunc)
        # sample forwards
        self.sample_forwards(betal,betastarl)

        return self

    def _messages_backwards(self,Al,aDl,aDsl,trunc):
        T = aDl.shape[0]
        state_dim = Al.shape[0]
        betal = np.zeros((T,state_dim),dtype=np.float64)
        betastarl = np.zeros((T,state_dim),dtype=np.float64)
        assert betal.shape == aDl.shape == aDsl.shape

        for t in xrange(T-1,-1,-1):
            np.logaddexp.reduce(betal[t:t+trunc] + self.cumulative_likelihoods(t,t+trunc) + aDl[:min(trunc,T-t)],axis=0, out=betastarl[t])
            if T-t < trunc and self.censoring:
                np.logaddexp(betastarl[t], self.likelihood_block(t,None) + aDsl[T-t-1], betastarl[t])
            np.logaddexp.reduce(betastarl[t] + Al,axis=1,out=betal[t-1])
        betal[-1] = 0.

        assert not np.isnan(betal).any()
        assert not np.isinf(betal[0]).all()

        return betal, betastarl

    def get_aBl(self,data):
        aBl = np.empty((data.shape[0],self.state_dim))
        for idx in xrange(self.state_dim):
            aBl[:,idx] = self.obs_distns[idx].log_likelihood(data)
        return aBl

    def cumulative_likelihoods(self,start,stop):
        return np.cumsum(self.aBl[start:stop],axis=0)

    def cumulative_likelihood_state(self,start,stop,state):
        return np.cumsum(self.aBl[start:stop,state])

    def likelihood_block(self,start,stop):
        # does not include the stop index
        return np.sum(self.aBl[start:stop],axis=0)

    def likelihood_block_state(self,start,stop,state):
        return np.sum(self.aBl[start:stop,state])

    def sample_forwards(self,betal,betastarl):
        stateseq = self.stateseq = np.zeros(self.T,dtype=np.int32)
        durations = []
        stateseq_norep = []

        idx = 0
        A = self.transition_distn.A
        nextstate_unsmoothed = self.initial_distn.pi_0

        apmf = np.zeros((self.state_dim,self.T))
        arg = np.arange(1,self.T+1)
        for state_idx, dur_distn in enumerate(self.dur_distns):
            apmf[state_idx] = dur_distn.pmf(arg)

        while idx < self.T:
            logdomain = betastarl[idx] - np.amax(betastarl[idx])
            nextstate_distr = np.exp(logdomain) * nextstate_unsmoothed
            if (nextstate_distr == 0.).all():
                # this is a numerical issue; no good answer, so we'll just follow the messages.
                nextstate_distr = np.exp(logdomain)
            state = sample_discrete(nextstate_distr)
            assert len(stateseq_norep) == 0 or state != stateseq_norep[-1]

            durprob = random()
            dur = 0 # always incremented at least once
            prob_so_far = 0.0
            while durprob > 0:
                assert dur < 2*self.T # hacky infinite loop check
                #assert self.dur_distns[state].pmf(dur+1) == apmf[state,dur]
                p_d_marg = apmf[state,dur] if dur < self.T else 1. # note funny indexing: dur variable is 1 less than actual dur we're considering
                assert not np.isnan(p_d_marg)
                assert p_d_marg >= 0
                if p_d_marg == 0:
                    dur += 1
                    continue
                if idx+dur < self.T:
                    mess_term = np.exp(self.likelihood_block_state(idx,idx+dur+1,state) \
                            + betal[idx+dur,state] - betastarl[idx,state])
                    p_d = mess_term * p_d_marg
                    prob_so_far += p_d

                    assert not np.isnan(p_d)
                    durprob -= p_d
                    dur += 1
                else:
                    if self.censoring:
                        # TODO 3*T is a hacky approximation, could use log_sf
                        dur += sample_discrete(dur_distn.pmf(np.arange(dur+1,3*self.T)))
                    else:
                        dur += 1
                    durprob = -1 # just to get us out of loop

            assert dur > 0

            stateseq[idx:idx+dur] = state
            stateseq_norep.append(state)
            assert len(stateseq_norep) < 2 or stateseq_norep[-1] != stateseq_norep[-2]
            durations.append(dur)

            nextstate_unsmoothed = A[state,:]

            idx += dur

        self.durations = np.array(durations,dtype=np.int32)
        self.stateseq_norep = np.array(stateseq_norep,dtype=np.int32)

    def plot(self,colors_dict=None,**kwargs):
        from matplotlib import pyplot as plt
        X,Y = np.meshgrid(np.hstack((0,self.durations.cumsum())),(0,1))

        if colors_dict is not None:
            C = np.array([[colors_dict[state] for state in self.stateseq_norep]])
        else:
            C = self.stateseq_norep[na,:]

        plt.pcolor(X,Y,C,vmin=0,vmax=1,**kwargs)
        plt.ylim((0,1))
        plt.xlim((0,self.T))
        plt.yticks([])
        plt.title('State Sequence')

class HSMMStatesEigen(HSMMStatesPython):
    def __init__(self,T,state_dim,*args,**kwargs):
        self.sample_forwards_codestr = hsmm_sample_forwards_codestr % {'M':state_dim,'T':T}
        super(HSMMStatesEigen,self).__init__(T,state_dim,*args,**kwargs)

    def sample_forwards(self,betal,betastarl):
        aBl = self.aBl
        # stateseq = np.array(self.stateseq,dtype=np.int32)
        stateseq = np.zeros(betal.shape[0],dtype=np.int32)
        A = self.transition_distn.A
        pi0 = self.initial_distn.pi_0

        apmf = np.zeros((self.state_dim,self.T))
        arg = np.arange(1,self.T+1)
        for state_idx, dur_distn in enumerate(self.dur_distns):
            apmf[state_idx] = dur_distn.pmf(arg)

        scipy.weave.inline(self.sample_forwards_codestr,
                ['betal','betastarl','aBl','stateseq','A','pi0','apmf'],
                headers=['<Eigen/Core>'],
                include_dirs=['/usr/include/eigen3'],
                extra_compile_args=['-O3','-DNDEBUG'])#,'-march=native'])


        self.stateseq_norep, self.durations = util.rle(stateseq)
        self.stateseq = stateseq

        if self.censoring:
            dur = self.durations[-1]
            dur_distn = self.dur_distns[self.stateseq_norep[-1]]

            # TODO instead of 3*T, use log_sf
            self.durations[-1] += sample_discrete(dur_distn.pmf(np.arange(dur+1,3*self.T)))


class HMMStatesPython(object):
    def __init__(self,T,state_dim,obs_distns,transition_distn,initial_distn,stateseq=None,data=None,
            initialize_from_prior=True,**kwargs):
        self.state_dim = state_dim
        self.obs_distns = obs_distns
        self.transition_distn = transition_distn
        self.initial_distn = initial_distn

        self.T = T
        self.data = data

        if stateseq is not None:
            self.stateseq = stateseq
        else:
            if data is not None and not initialize_from_prior:
                self.resample()
            else:
                self.generate_states()

    ### generation

    def generate(self):
        self.generate_states()
        return self.generate_obs()

    def generate_states(self):
        T = self.T
        stateseq = np.zeros(T,dtype=np.int32)
        nextstate_distn = self.initial_distn.pi_0
        A = self.transition_distn.A

        for idx in xrange(T):
            stateseq[idx] = sample_discrete(nextstate_distn)
            nextstate_distn = A[stateseq[idx]]

        self.stateseq = stateseq
        return stateseq

    def generate_obs(self):
        obs = []
        for state in self.stateseq:
            obs.append(self.obs_distns[state].rvs(size=1))
        return np.concatenate(obs)

    ### message passing

    def get_aBl(self,data):
        # note: this method never uses self.T
        aBl = np.zeros((data.shape[0],self.state_dim))
        for idx, obs_distn in enumerate(self.obs_distns):
            aBl[:,idx] = obs_distn.log_likelihood(data)
        return aBl

    def messages_forwards(self,aBl):
        # note: this method never uses self.T
        # or self.data
        T = aBl.shape[0]
        alphal = np.zeros((T,self.state_dim))
        Al = np.log(self.transition_distn.A)

        alphal[0] = np.log(self.initial_distn.pi_0) + aBl[0]

        for t in xrange(T-1):
            alphal[t+1] = np.logaddexp.reduce(alphal[t] + Al.T,axis=1) + aBl[t+1]

        return alphal

    def _messages_backwards(self,aBl):
        betal = np.zeros(aBl.shape)
        Al = np.log(self.transition_distn.A)
        T = aBl.shape[0]

        for t in xrange(T-2,-1,-1):
            np.logaddexp.reduce(Al + betal[t+1] + aBl[t+1],axis=1,out=betal[t])

        return betal

    ### Gibbs sampling

    def sample_forwards(self,aBl,betal):
        T = aBl.shape[0]
        stateseq = np.zeros(T,dtype=np.int32)
        nextstate_unsmoothed = self.initial_distn.pi_0
        A = self.transition_distn.A

        for idx in xrange(T):
            logdomain = betal[idx] + aBl[idx]
            logdomain[nextstate_unsmoothed == 0] = -np.inf # to enforce constraints in the trans matrix
            stateseq[idx] = sample_discrete(nextstate_unsmoothed * np.exp(logdomain - np.amax(logdomain)))
            nextstate_unsmoothed = A[stateseq[idx]]

        self.stateseq = stateseq

    def resample(self):
        if self.data is None:
            print 'ERROR: can only call resample on %s instances with data' % type(self)
        else:
            data = self.data

        aBl = self.get_aBl(data)
        betal = self._messages_backwards(aBl)
        self.sample_forwards(aBl,betal)

        return self

    ### EM

    def E_step(self):
        aBl = self.aBl = self.get_aBl(self.data) # saving aBl makes transition distn job easier

        alphal = self.alphal = self.messages_forwards(aBl)
        betal = self.betal = self._messages_backwards(aBl)
        expectations = self.expectations = alphal + betal

        expectations -= expectations.max(1)[:,na]
        np.exp(expectations,out=expectations)
        expectations /= expectations.sum(1)[:,na]

    ### plotting

    def plot(self,colors_dict=None,vertical_extent=(0,1),**kwargs):
        from matplotlib import pyplot as plt
        states,durations = util.rle(self.stateseq)
        X,Y = np.meshgrid(np.hstack((0,durations.cumsum())),vertical_extent)

        if colors_dict is not None:
            C = np.array([[colors_dict[state] for state in states]])
        else:
            C = states[na,:]

        plt.pcolor(X,Y,C,vmin=0,vmax=1,**kwargs)
        plt.ylim(vertical_extent)
        plt.xlim((0,durations.sum()))
        plt.yticks([])

class HMMStatesEigen(HMMStatesPython):
    def __init__(self,T,state_dim,obs_distns,transition_distn,initial_distn,
            stateseq=None,data=None,censoring=True,
            initialize_from_prior=True,**kwargs):
        # TODO shouldn't this use parent's init?
        self.state_dim = state_dim
        self.obs_distns = obs_distns
        self.transition_distn = transition_distn
        self.initial_distn = initial_distn
        self.T = T
        self.data = data

        self.messages_backwards_codestr = hmm_messages_backwards_codestr % {'M':state_dim}
        self.sample_forwards_codestr = hmm_sample_forwards_codestr % {'M':state_dim}

        if stateseq is not None:
            self.stateseq = stateseq
        else:
            if data is not None and not initialize_from_prior:
                self.resample()
            else:
                self.generate_states()

    def _messages_backwards(self,aBl):
        T = self.T
        AT = self.transition_distn.A.T.copy()
        betal = np.zeros((self.T,self.state_dim))

        scipy.weave.inline(self.messages_backwards_codestr,
                           ['AT','betal','aBl','T'],
                           headers=['<Eigen/Core>'],
                           include_dirs=['/usr/include/eigen3'],
                           extra_compile_args=['-O3','-DNDEBUG'])

        return betal

    def sample_forwards(self,aBl,betal):
        T = self.T
        A = self.transition_distn.A
        pi0 = self.initial_distn.pi_0

        stateseq = np.zeros(T,dtype=np.int32)
        
        scipy.weave.inline(self.sample_forwards_codestr,
                           ['A','T','pi0','stateseq','aBl','betal'],
                           headers=['<Eigen/Core>','<limits>'],
                           include_dirs=['/usr/include/eigen3'],
                           extra_compile_args=['-O3','-DNDEBUG'])
        
        #scipy.weave.inline(self.sample_forwards_codestr,['A','T','pi0','stateseq','aBl','betal'],
                #headers=['<Eigen/Core>','<limits>'],include_dirs=['/usr/local/include/eigen3'],
                #extra_compile_args=['-O3','-DNDEBUG'])  ## /build_dir

        self.stateseq = stateseq
'''
class HSMMStatesPossibleChangepoints(HSMMStatesPython):
'''

################################
#  use_eigen stuff below here  #
################################

# default is python
HSMMStates = HSMMStatesPython
HMMStates = HMMStatesPython

def use_eigen(useit=True):
    # TODO this method is probably dumb; should get rid of it and just make the
    #class usage explicit
    global HSMMStates, HMMStates, hmm_sample_forwards_codestr, \
            hsmm_sample_forwards_codestr, hmm_messages_backwards_codestr

    if useit:
        import os
        eigen_code_dir = os.path.join(os.path.dirname(__file__),'cpp_eigen_code/')
        with open(os.path.join(eigen_code_dir,'hsmm_sample_forwards.cpp')) as infile:
            hsmm_sample_forwards_codestr = infile.read()
        with open(os.path.join(eigen_code_dir,'hmm_messages_backwards.cpp')) as infile:
            hmm_messages_backwards_codestr = infile.read()
        with open(os.path.join(eigen_code_dir,'hmm_sample_forwards.cpp')) as infile:
            hmm_sample_forwards_codestr = infile.read()

            HSMMStates = HSMMStatesEigen
            HMMStates = HMMStatesEigen
    else:
        HSMMStates = HSMMStatesPython
        HMMStates = HMMStatesPython

