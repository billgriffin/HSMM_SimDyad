from __future__ import division
import numpy as np
import scipy.stats as stats
import scipy.special as special

from pybasicbayes.distributions import *
from abstractions import DurationDistribution


class PoissonDuration(Poisson, DurationDistribution):
    def __repr__(self):
        return 'PoissonDuration(lmbda=%0.2f,mean=%0.2f)' % (self.lmbda,self.lmbda+1)

    def log_sf(self,x):
        return stats.poisson.logsf(x-1,self.lmbda) # TODO reimplement

    def log_likelihood(self,x):
        return super(PoissonDuration,self).log_likelihood(x-1)

    def rvs(self,size=None):
        return super(PoissonDuration,self).rvs(size=size) + 1

    def _get_statistics(self,data):
        n, tot = super(PoissonDuration,self)._get_statistics(data)
        tot -= n
        return n, tot

'''
class GeometricDuration(Geometric, DurationDistribution):

class NegativeBinomialDuration(NegativeBinomial, DurationDistribution):
'''