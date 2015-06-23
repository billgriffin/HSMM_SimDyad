from __future__ import division
import numpy as np

from ..basic.abstractions import GibbsSampling, MaxLikelihood
from ..basic.distributions import Multinomial

class InitialState(Multinomial):
    def __init__(self,state_dim,rho,pi_0=None):
        super(InitialState,self).__init__(alpha_0=rho,K=state_dim,weights=pi_0)

    @property
    def pi_0(self):
        return self.weights
'''
class StartInZero(GibbsSampling,MaxLikelihood):
'''