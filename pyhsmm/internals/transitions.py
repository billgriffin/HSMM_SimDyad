from __future__ import division
import numpy as np
import scipy.stats as stats
from numpy import newaxis as na
np.seterr(invalid='raise')
import operator

from ..basic.distributions import DirGamma
from ..util.general import rle

##########
#  misc  #
##########

# TODO scaling by self.state_dim in concresampling is the confusing result of
# having a DirGamma object and not a WLDPGamma object! make one

class ConcentrationResampling(object):
    def __init__(self,state_dim,alpha_a_0,alpha_b_0,gamma_a_0,gamma_b_0):
        self.gamma_obj = DirGamma(state_dim,gamma_a_0,gamma_b_0)
        self.alpha_obj = DirGamma(state_dim,alpha_a_0,alpha_b_0)

    def resample(self):
        # multiply by state_dim because the trans objects divide by it (since
        # their parameters correspond to the DP parameters, and so they convert
        # into weak limit scaling)
        self.alpha_obj.resample(self.trans_counts,weighted_cols=self.beta,niter=5)
        self.alpha = self.alpha_obj.concentration*self.state_dim
        self.gamma_obj.resample(self.m,niter=5)
        self.gamma = self.gamma_obj.concentration*self.state_dim

######################
#  HDP-HMM classes  #
######################

class HDPHMMTransitions(object):
    def __init__(self,state_dim,alpha,gamma,beta=None,A=None):
        self.state_dim = state_dim
        self.alpha = alpha
        self.gamma = gamma
        if A is None or beta is None:
            self.resample()
        else:
            self.A = A
            self.beta = beta

    ### Gibbs sampling

    def resample(self,states_list=[]):
        trans_counts = self._count_transitions(states_list)
        m = self._get_m(trans_counts)

        self._resample_beta(m)
        self._resample_A(trans_counts)

    def _resample_beta(self,m):
        self.beta = np.random.dirichlet(self.gamma / self.state_dim + m.sum(0) + 1e-2)

    def _resample_A(self,trans_counts):
        self.A = stats.gamma.rvs(self.alpha * self.beta + trans_counts + 1e-2)
        self.A /= self.A.sum(1)[:,na]

    def _count_transitions(self,states_list):
        trans_counts = np.zeros((self.state_dim,self.state_dim),dtype=np.int32)
        for states in states_list:
            if len(states) >= 2:
                for idx in xrange(len(states)-1):
                    trans_counts[states[idx],states[idx+1]] += 1
        self.trans_counts = trans_counts
        return trans_counts

    def _get_m(self,trans_counts):
        m = np.zeros((self.state_dim,self.state_dim),dtype=np.int32)
        if not (0 == trans_counts).all():
            for (rowidx, colidx), val in np.ndenumerate(trans_counts):
                if val > 0:
                    m[rowidx,colidx] = (np.random.rand(val) < self.alpha * self.beta[colidx] \
                            /(np.arange(val) + self.alpha*self.beta[colidx])).sum()
        self.m = m
        return m

    ### max likelihood

    def max_likelihood(self,expectations_list):
        trans_softcounts = self._count_weighted_transitions(expectations_list,self.A)

        errs = np.seterr(invalid='ignore',divide='ignore')
        self.A = trans_softcounts / trans_softcounts.sum(1)[:,na]
        np.seterr(**errs)

        self.A[np.isnan(self.A)] = 0.

    # NOTE: only needs aBl because the message computation saves betal and not
    # betastarl TODO compute betastarl like a civilized gentleman
    @staticmethod
    def _count_weighted_transitions(expectations_list,A):
        trans_softcounts = np.zeros_like(A)
        Al = np.log(A)

        for alphal, betal, aBl in expectations_list:
            log_joints = alphal[:-1,:,na] + (betal[1:,na,:] + aBl[1:,na,:]) + Al[na,...]
            log_joints -= np.logaddexp.reduce(alphal[0] + betal[0]) # p(y)
            joints = np.exp(log_joints,out=log_joints)

            trans_softcounts += joints.sum(0)

        return trans_softcounts


class HDPHMMTransitionsConcResampling(HDPHMMTransitions,ConcentrationResampling):
    def __init__(self,state_dim,
            alpha_a_0,alpha_b_0,gamma_a_0,gamma_b_0,
            **kwargs):

        ConcentrationResampling.__init__(self,state_dim,alpha_a_0,alpha_b_0,gamma_a_0,gamma_b_0)
        super(HDPHMMTransitionsConcResampling,self).__init__(state_dim,
                alpha=self.alpha_obj.concentration*state_dim,
                gamma=self.gamma_obj.concentration*state_dim,
                **kwargs)

    def resample(self,*args,**kwargs):
        super(HDPHMMTransitionsConcResampling,self).resample(*args,**kwargs)
        ConcentrationResampling.resample(self)


######################
#  HDP-HSMM classes  #
######################

class HDPHSMMTransitions(HDPHMMTransitions):
    '''
    HDPHSMM transition distribution class.
    Uses a weak-limit HDP prior. Zeroed diagonal to forbid self-transitions.

    Hyperparameters follow the notation in Fox et al.
        gamma: concentration paramter for beta
        alpha: total mass concentration parameter for each row of trans matrix

    Parameters are the shared transition vector beta, the full transition matrix,
    and the matrix with the diagonal zeroed.
        beta, A, fullA
    '''

    def __init__(self,state_dim,alpha,gamma,beta=None,A=None,fullA=None):
        self.alpha = alpha
        self.gamma = gamma
        self.state_dim = state_dim
        if A is None or fullA is None or beta is None:
            self.resample()
        else:
            self.A = A
            self.beta = beta
            self.fullA = fullA

    def resample(self,stateseqs=[]):
        states_noreps = map(operator.itemgetter(0),map(rle, stateseqs))

        augmented_data = self._augment_data(self._count_transitions(states_noreps))
        m = self._get_m(augmented_data)

        self._resample_beta(m)
        self._resample_A(augmented_data)

    def _augment_data(self,trans_counts):
        trans_counts = trans_counts.copy()
        if trans_counts.sum() > 0:
            froms = trans_counts.sum(1)
            self_transitions = [np.random.geometric(1-pi_ii,size=n).sum() if n > 0 else 0
                    for pi_ii,n in zip(self.fullA.diagonal(),froms)]
            trans_counts += np.diag(self_transitions)
        self.trans_counts = trans_counts
        return trans_counts

    def _resample_A(self,augmented_data):
        super(HDPHSMMTransitions,self)._resample_A(augmented_data)
        self.fullA = self.A.copy()
        self.A.flat[::self.A.shape[0]+1] = 0
        self.A /= self.A.sum(1)[:,na]


class HDPHSMMTransitionsConcResampling(HDPHSMMTransitions,ConcentrationResampling):
    def __init__(self,state_dim,
            alpha_a_0,alpha_b_0,gamma_a_0,gamma_b_0,
            **kwargs):

        ConcentrationResampling.__init__(self,state_dim,alpha_a_0,alpha_b_0,gamma_a_0,gamma_b_0)
        super(HDPHSMMTransitionsConcResampling,self).__init__(state_dim,
                alpha=self.alpha_obj.concentration*state_dim,
                gamma=self.gamma_obj.concentration*state_dim,
                **kwargs)

    def resample(self,*args,**kwargs):
        super(HDPHSMMTransitionsConcResampling,self).resample(*args,**kwargs)
        ConcentrationResampling.resample(self)


############################
#  Sticky HDP-HMM classes  #
############################

# deleted