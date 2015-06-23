from __future__ import division
import numpy as np
np.seterr(divide='ignore')
from numpy import newaxis as na
from numpy.core.umath_tests import inner1d
import scipy.weave
import scipy.stats as stats
import scipy.special as special
import matplotlib.pyplot as plt
import abc
from warnings import warn

from abstractions import Distribution, GibbsSampling,\
        MeanField, Collapsed, MaxLikelihood
from util.stats import sample_niw, invwishart_entropy,\
        invwishart_log_partitionfunction, sample_discrete,\
        sample_discrete_from_log, getdatasize, flattendata,\
        getdatadimension
import util.general

################
#  Continuous  #
################

class Gaussian(GibbsSampling, MeanField, Collapsed, MaxLikelihood):
    '''
    Multivariate Gaussian distribution class.

    NOTE: Only works for 2 or more dimensions. For a scalar Gaussian, use one of
    the scalar classes.  Uses a conjugate Normal/Inverse-Wishart prior.

    Hyperparameters mostly follow Gelman et al.'s notation in Bayesian Data
    Analysis, except sigma_0 is proportional to expected covariance matrix:
        nu_0, sigma_0
        mu_0, kappa_0

    Parameters are mean and covariance matrix:
        mu, sigma
    '''

    def __init__(self,mu_0,sigma_0,kappa_0,nu_0,mu=None,sigma=None):
        # xun: [0.1] sigma_0 <--- lmbda_0 is just a name change, it works the same as in old code 
        self.mu_0    = mu_0
        self.sigma_0 = sigma_0
        self.kappa_0 = kappa_0
        self.nu_0    = nu_0

        self.D = mu_0.shape[0]
        assert sigma_0.shape == (self.D,self.D) and self.D >= 2

        if mu is None or sigma is None:
            self.resample()
        else:
            self.mu = mu
            self.sigma = sigma
        self._mu_mf = self.mu
        self._sigma_mf = self.sigma
        self._kappa_mf = kappa_0
        self._nu_mf = nu_0

    def num_parameters(self):
        return self.D*(self.D+1)/2

    def rvs(self,size=None):
        return np.random.multivariate_normal(mean=self.mu,cov=self.sigma,size=size)

    def log_likelihood(self,x):
        mu, sigma, D = self.mu, self.sigma, self.D
        x = np.reshape(x,(-1,D)) - mu
        xs,LT = util.general.solve_chofactor_system(sigma,x.T,overwrite_b=True)
        return -1./2. * inner1d(xs.T,xs.T) - D/2*np.log(2*np.pi) - np.log(LT.diagonal()).sum()

    def _posterior_hypparams(self,n,xbar,sumsq):
        mu_0, sigma_0, kappa_0, nu_0 = self.mu_0, self.sigma_0, self.kappa_0, self.nu_0
        if n > 0:
            mu_n = self.kappa_0 / (self.kappa_0 + n) * self.mu_0 + n / (self.kappa_0 + n) * xbar
            kappa_n = self.kappa_0 + n
            nu_n = self.nu_0 + n
            sigma_n = self.sigma_0 + sumsq + \
                    self.kappa_0*n/(self.kappa_0+n) * np.outer(xbar-self.mu_0,xbar-self.mu_0)

            return mu_n, sigma_n, kappa_n, nu_n
        else:
            return mu_0, sigma_0, kappa_0, nu_0

    ### Gibbs sampling

    def resample(self,data=[]):
        self._mu_mf, self._sigma_mf = self.mu, self.sigma = \
                sample_niw(*self._posterior_hypparams(*self._get_statistics(data,self.D)))

    @staticmethod
    def _get_statistics(data,D):
        n = getdatasize(data)
        if n > 0:
            if isinstance(data,np.ndarray):
                xbar = np.reshape(data,(-1,D)).mean(0)
                centered = data - xbar
                sumsq = np.dot(centered.T,centered)
            else:
                xbar = sum(np.reshape(d,(-1,D)).sum(0) for d in data) / n
                sumsq = sum(np.dot((np.reshape(d,(-1,D))-xbar).T,(np.reshape(d,(-1,D))-xbar))
                        for d in data)
        else:
            xbar, sumsq = None, None
        return n, xbar, sumsq

    ### Mean Field

    # NOTE my sumsq is Bishop's Nk*Sk

    def meanfieldupdate(self,data,weights):
        # update
        self._mu_mf, self._sigma_mf, self._kappa_mf, self._nu_mf = \
                self._posterior_hypparams(*self._get_weighted_statistics(data,weights,self.D))
        self._sigma_mf_chol = None
        self.mu, self.sigma = self._mu_mf, self._sigma_mf/(self._nu_mf - self.D - 1) # for plotting

    def _get_sigma_mf_chol(self):
        if not hasattr(self,'_sigma_mf_chol') or self._sigma_mf_chol is None:
            self._sigma_mf_chol = util.general.cholesky(self._sigma_mf)
        return self._sigma_mf_chol

    def get_vlb(self):
        # return avg energy plus entropy, our contribution to the mean field
        # variational lower bound
        D = self.D
        loglmbdatilde = self._loglmbdatilde()
        chol = self._get_sigma_mf_chol()

        # see Eq. 10.77 in Bishop
        q_entropy = -0.5 * (loglmbdatilde + self.D * (np.log(self._kappa_mf/(2*np.pi))-1)) \
                + invwishart_entropy(self._sigma_mf,self._nu_mf,chol)
        # see Eq. 10.74 in Bishop, we aren't summing over K
        p_avgengy = 0.5 * (D * np.log(self.kappa_0/(2*np.pi)) + loglmbdatilde \
                - D*self.kappa_0/self._kappa_mf - self.kappa_0*self._nu_mf*\
                np.dot(self._mu_mf -
                    self.mu_0,util.general.solve_psd(self._sigma_mf,self._mu_mf - self.mu_0,chol=chol))) \
                + invwishart_log_partitionfunction(self.sigma_0,self.nu_0) \
                + (self.nu_0 - D - 1)/2*loglmbdatilde - 1/2*self._nu_mf*\
                util.general.solve_psd(self._sigma_mf,self.sigma_0,chol=chol).trace()

        return p_avgengy + q_entropy

    def expected_log_likelihood(self,x):
        mu_n, sigma_n, kappa_n, nu_n = self._mu_mf, self._sigma_mf, self._kappa_mf, self._nu_mf
        D = self.D
        x = np.reshape(x,(-1,D)) - mu_n # x is now centered
        chol = self._get_sigma_mf_chol()
        xs = util.general.solve_triangular(chol,x.T,overwrite_b=True)

        # see Eqs. 10.64, 10.67, and 10.71 in Bishop
        return self._loglmbdatilde()/2 - D/(2*kappa_n) - nu_n/2 * \
                inner1d(xs.T,xs.T) - D/2*np.log(2*np.pi)

    def _loglmbdatilde(self):
        # see Eq. 10.65 in Bishop
        chol = self._get_sigma_mf_chol()
        return special.digamma((self._nu_mf-np.arange(self.D))/2).sum() \
                + self.D*np.log(2) - 2*np.log(chol.diagonal()).sum()

    @staticmethod
    def _get_weighted_statistics(data,weights,D):
        # NOTE: _get_statistics is special case with all weights being 1
        # this is kept as a separate method for speed and modularity
        if isinstance(data,np.ndarray):
            neff = weights.sum()
            if neff > 0:
                xbar = np.dot(weights,np.reshape(data,(-1,D))) / neff
                centered = np.reshape(data,(-1,D)) - xbar
                sumsq = np.dot(centered.T,(weights[:,na] * centered))
            else:
                xbar, sumsq = None, None
        else:
            neff = sum(w.sum() for w in weights)
            if neff > 0:
                xbar = sum(np.dot(w,np.reshape(d,(-1,D))) for w,d in zip(weights,data)) / neff
                sumsq = sum(np.dot((np.reshape(d,(-1,D))-xbar).T,w[:,na]*(np.reshape(d,(-1,D))-xbar))
                        for w,d in zip(weights,data))
            else:
                xbar, sumsq = None, None

        return neff, xbar, sumsq

    ### Collapsed

    def log_marginal_likelihood(self,data):
        n, D = getdatasize(data), self.D
        return self._log_partition_function(*self._posterior_hypparams(*self._get_statistics(data,self.D))) \
                - self._log_partition_function(self.mu_0,self.sigma_0,self.kappa_0,self.nu_0) \
                - n*D/2 * np.log(2*np.pi)

    def _log_partition_function(self,mu,sigma,kappa,nu):
        D = self.D
        chol = util.general.cholesky(sigma)
        return nu*D/2*np.log(2) + special.multigammaln(nu/2,D) + D/2*np.log(2*np.pi/kappa) \
                - nu*np.log(chol.diagonal()).sum()

    ### Max likelihood

    # NOTE: could also use sumsq/(n-1) as the covariance estimate, which would
    # be unbiased but not max likelihood, but if we're in the regime where that
    # matters we've got bigger problems!

    def max_likelihood(self,data,weights=None):
        D = self.D
        if weights is None:
            n, muhat, sumsq = self._get_statistics(data,D)
        else:
            n, muhat, sumsq = self._get_weighted_statistics(data,weights,D)

        # this SVD is necessary to check if the max likelihood solution is
        # degenerate, which can happen in the EM algorithm
        if n < D or (np.linalg.svd(sumsq,compute_uv=False) > 1e-6).sum() < D:
            # broken!
            self.mu = 99999999*np.ones(D)
            self.sigma = np.eye(D)
        else:
            self.mu = muhat
            self.sigma = sumsq/n

    @classmethod
    def max_likelihood_constructor(cls,data,weights=None):
        D = getdatadimension(data)
        if weights is None:
            n, muhat, sumsq = cls._get_statistics(data,D)
        else:
            n, muhat, sumsq = cls._get_weighted_statistics(data,weights,D)
        assert n >= D

        return cls(muhat,sumsq/n,n,n,mu=muhat,sigma=sumsq/n)

    def max_likelihood_withprior(self,data,weights=None):
        D = self.D
        if weights is None:
            n, muhat, sumsq = self._get_statistics(data,D)
        else:
            n, muhat, sumsq = self._get_weighted_statistics(data,weights,D)

        self.mu, self.sigma, _, _ = self._posterior_hypparams(n,muhat,sumsq)

    ### Misc

    # TODO get rid of this; just make it happen the first time plot is called
    @classmethod
    def _plot_setup(cls,instance_list):
        # must set cls.vecs to be a reasonable 2D space to project onto
        # so that the projection is consistent across instances
        # for now, i'll just make it random if there are more than 2 dimensions
        assert len(instance_list) > 0
        assert len(set([len(o.mu) for o in instance_list])) == 1, \
                'must have consistent dimensions across instances'
        dim = len(instance_list[0].mu)
        if dim > 2:
            vecs = np.random.randn((dim,2))
            vecs /= np.sqrt((vecs**2).sum())
        else:
            vecs = np.eye(2)

        for o in instance_list:
            o.global_vecs = vecs

    def plot(self,data=None,color='b',plot_params=True,label='',cmap=None):
        from util.plot import project_data, plot_gaussian_projection, pca
        if data is not None:
            data = flattendata(data)

        try:
            vecs = self.global_vecs
        except AttributeError:
            dim = len(self.mu)
            if dim == 2:
                vecs = np.eye(2)
            elif data is not None:
                assert dim > 2
                vecs = pca(data,num_components=2)
            else:
                vecs = np.random.randn(2,2)

        if data is not None:
            projected_data = project_data(data,vecs)
            if cmap==None:
                plt.plot(projected_data[:,0],projected_data[:,1],marker='.',linestyle=' ',color=color)
            else:
                colors = [cmap(sum(i)/16) for i in projected_data]
                plt.scatter(projected_data[:,0],projected_data[:,1], s=20, c=colors,edgecolors='none')                

        if plot_params:
            plot_gaussian_projection(self.mu,self.sigma,vecs,color=color,label=label)

    def to_json_dict(self):
        assert self.D == 2
        U,s,_ = np.linalg.svd(self.sigma)
        U /= np.linalg.det(U)
        theta = np.arctan2(U[0,0],U[0,1])*180/np.pi
        return {'x':self.mu[0],'y':self.mu[1],'rx':np.sqrt(s[0]),'ry':np.sqrt(s[1]),'theta':theta}

'''
# TODO collapsed, meanfield, max_likelihood
class DiagonalGaussian(GibbsSampling):

# TODO collapsed, meanfield, max_likelihood
class IsotropicGaussian(GibbsSampling):

class ScalarGaussian(Distribution):
    
# TODO collapsed, meanfield, max_likelihood
class ScalarGaussianNIX(ScalarGaussian, GibbsSampling, Collapsed):

class ScalarGaussianNonconjNIX(ScalarGaussian, GibbsSampling):
  
# TODO collapsed, meanfield, max_likelihood
class ScalarGaussianFixedvar(ScalarGaussian, GibbsSampling):
'''

##############
#  Discrete  #
##############

# TODO this should be called categorical!
class Multinomial(GibbsSampling, MeanField, MaxLikelihood):
    '''
    This class represents a categorical distribution over labels, where the
    parameter is weights and the prior is a Dirichlet distribution.
    For example, if K == 3, then five samples may look like
        [0,1,0,2,1]
    Each entry is the label of a sample, like the outcome of die rolls. In other
    words, data are not indicator variables! (Except when they need to be, like
    in the mean field update or the weighted max likelihood (EM) update.)

    This can be used as a weak limit approximation for a DP, particularly by
    calling __init__ with alpha_0 and K arguments, in which case the prior will be
    a symmetric Dirichlet with K components and parameter alpha_0/K; K is then the
    weak limit approximation parameter.

    Hyperparaemters:
        alphav_0 (vector) OR alpha_0 (scalar) and K

    Parameters:
        weights, a vector encoding a discrete pmf
    '''
    def __repr__(self):
        return 'Multinomial(weights=%s)' % (self.weights,)

    def __init__(self,weights=None,alpha_0=None,K=None,alphav_0=None):
        assert (isinstance(alphav_0,np.ndarray) and alphav_0.ndim == 1) ^ \
                (K is not None and alpha_0 is not None)

        if alphav_0 is not None:
            self.alphav_0 = alphav_0
            self.K = alphav_0.shape[0]
        else:
            self.K = K
            self.alphav_0 = np.repeat(alpha_0/K,K)

        if weights is not None:
            self.weights = weights
        else:
            self.resample()
        self._alpha_mf = self.weights * self.alphav_0.sum()

    def num_parameters(self):
        return self.K

    def rvs(self,size=None):
        return sample_discrete(self.weights,size)

    def log_likelihood(self,x):
        return np.log(self.weights)[x]

    def _posterior_hypparams(self,counts):
        return self.alphav_0 + counts

    ### Gibbs sampling

    def resample(self,data=[],count_data=None):
        if count_data is None:
            hypparams = self._posterior_hypparams(*self._get_statistics(data,self.K))
        else:
            hypparams = self._posterior_hypparams(count_data)
        self.weights = np.random.dirichlet(np.where(hypparams>1e-2,hypparams,1e-2))

    @staticmethod
    def _get_statistics(data,K):
        assert isinstance(data,np.ndarray) or \
                (isinstance(data,list) and all(isinstance(d,np.ndarray) for d in data))

        if isinstance(data,np.ndarray):
            if np.version.version > '1.6':
                counts = np.bincount(data,minlength=K)
            else:
                tmp = np. bincount(d)
                tmp.resize((K,))
                counts = tmp
        else:
            if np.version.version > '1.9':
                counts = sum(np.bincount(d,minlength=K) for d in data)
            else:
                bincounts = []
                for d in data:
                    tmp = np. bincount(d)
                    tmp.resize((K,))
                    bincounts.append(tmp)
                counts = sum(bincounts)
        return counts,

    ### Mean Field

    def meanfieldupdate(self,data,weights):
        # update
        self._alpha_mf = self._posterior_hypparams(*self._get_weighted_statistics(data,weights))
        self.weights = self._alpha_mf / self._alpha_mf.sum() # for plotting

    def get_vlb(self):
        # return avg energy plus entropy, our contribution to the vlb
        # see Eq. 10.66 in Bishop
        logpitilde = self.expected_log_likelihood(np.arange(self.K))
        q_entropy = -1* ((logpitilde*(self._alpha_mf-1)).sum() \
                + special.gammaln(self._alpha_mf.sum()) - special.gammaln(self._alpha_mf).sum())
        p_avgengy = special.gammaln(self.alphav_0.sum()) - special.gammaln(self.alphav_0).sum() \
                + ((self.alphav_0-1)*logpitilde).sum()

        return p_avgengy + q_entropy

    def expected_log_likelihood(self,x):
        # this may only make sense if np.all(x == np.arange(self.K))...
        return special.digamma(self._alpha_mf[x]) - special.digamma(self._alpha_mf.sum())

    @staticmethod
    def _get_weighted_statistics(data,weights):
        # data is just a placeholder; technically it should be
        # np.arange(K)[na,:].repeat(N,axis=0)
        assert isinstance(weights,np.ndarray) or \
                (isinstance(weights,list) and
                        all(isinstance(w,np.ndarray) for w in weights))

        if isinstance(weights,np.ndarray):
            counts = weights.sum(0)
        else:
            counts = sum(w.sum(0) for w in weights)
        return counts,

    ### Max likelihood

    def max_likelihood(self,data,weights=None):
        K = self.K
        if weights is None:
            counts, = self._get_statistics(data,K)
        else:
            counts, = self._get_weighted_statistics(data,weights)

        self.weights = counts/counts.sum()

    def max_likelihood_countdata(self,counts):
        self.weights = counts /counts.sum()

    def max_likelihood_withprior(self,data,weights=None):
        K = self.K
        if weights is None:
            counts, = self._get_statistics(data,K)
        else:
            counts, = self._get_weighted_statistics(data,weights)

        self.weights = counts/counts.sum()

    # TODO weighted max likelihood!


class MultinomialConcentration(Multinomial):
    '''
    Multinomial with resampling of the symmetric Dirichlet concentration
    parameter.

        concentration ~ Gamma(a_0,b_0)

    The Dirichlet prior over pi is then

        pi ~ Dir(concentration/K)
    '''
    def __init__(self,a_0,b_0,K,concentration=None,weights=None):
        self.concentration = DirGamma(a_0=a_0,b_0=b_0,K=K,
                concentration=concentration)
        super(MultinomialConcentration,self).__init__(alpha_0=self.concentration.concentration,
                K=K,weights=weights)

    def resample(self,data=[],count_data=None,niter=20):
        if count_data is None:
            if isinstance(data,list):
                counts = map(np.bincount,data)
            else:
                counts = np.bincount(data)
        else:
            counts = count_data

        for itr in range(niter):
            self.concentration.resample(counts,niter=1)
            self.alphav_0 = np.ones(self.K) * self.concentration.concentration
            super(MultinomialConcentration,self).resample(data)

    def meanfieldupdate(self,*args,**kwargs): # TODO
        warn('MeanField not implemented for %s; concentration parameter will stay fixed')
        super(MultinomialConcentration,self).meanfieldupdate(*args,**kwargs)

    def max_likelihood(self,*args,**kwargs):
        raise NotImplementedError, "max_likelihood doesn't make sense on this object"

'''
class Geometric(GibbsSampling, Collapsed):
'''  

class Poisson(GibbsSampling, Collapsed):
    '''
    Poisson distribution with a conjugate Gamma prior.

    NOTE: the support is {0,1,2,...}

     # xun: [0.2] from Wiki, the notation should be alpha_0 and beta_0, not alpha_0(k), theta_0 in olde code
    Hyperparameters (following Wikipedia's notation):
        alpha_0, beta_0

    Parameter is the mean/variance parameter:
        lmbda
    '''
    def __repr__(self):
        return 'Poisson(lmbda=%0.2f)' % (self.lmbda,)

    def __init__(self,alpha_0,beta_0,lmbda=None):
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        if lmbda is not None:
            self.lmbda = lmbda
        else:
            self.resample()

    def _posterior_hypparams(self,n,tot):
        return self.alpha_0 + tot, self.beta_0 + n

    def rvs(self,size=None):
        return np.random.poisson(self.lmbda,size=size)

    def log_likelihood(self,x):
        lmbda = self.lmbda
        x = np.array(x,ndmin=1)
        raw = np.empty(x.shape)
        raw[x>=0] = -lmbda + x[x>=0]*np.log(lmbda) - special.gammaln(x[x>=0]+1)
        raw[x<0] = -np.inf
        return raw if isinstance(x,np.ndarray) else raw[0]

    ### Gibbs Sampling

    def resample(self,data=[]):
        alpha_n, beta_n = self._posterior_hypparams(*self._get_statistics(data))
        self.lmbda = np.random.gamma(alpha_n,1/beta_n)

    def _get_statistics(self,data):
        if isinstance(data,np.ndarray):
            n = data.shape[0]
            tot = data.sum()
        elif isinstance(data,list):
            n = sum(d.shape[0] for d in data)
            tot = sum(d.sum() for d in data)
        else:
            assert isinstance(data,int)
            n = 1
            tot = data
        return n, tot

    ### Collapsed

    def log_marginal_likelihood(self,data):
        return self._log_partition_function(*self._posterior_hypparams(*self._get_statistics(data))) \
                - self._log_partition_function(self.alpha_0,self.beta_0) \
                - self._get_sum_of_gammas(data)

    def _log_partition_function(self,alpha,beta):
        return special.gammaln(alpha) - alpha * np.log(beta)

    def _get_sum_of_gammas(self,data):
        if isinstance(data,np.ndarray):
            return special.gammaln(data+1).sum()
        elif isinstance(data,list):
            return sum(special.gammaln(d+1).sum() for d in data)
        else:
            assert isinstance(data,int)
            return special.gammaln(data+1)

'''
class NegativeBinomial(GibbsSampling):
 '''

################################
#  Special Case Distributions  #
################################

# TODO maybe move these to another module? priors with funny likelihoods

class CRPGamma(GibbsSampling):
    '''
    Implements Gamma(a,b) prior over DP/CRP concentration parameter given
    CRP data (integrating out the weights). NOT for Gamma/Poisson, which would
    be called Poisson.
    see appendix A of http://www.cs.berkeley.edu/~jordan/papers/hdp.pdf
    and appendix C of Emily Fox's PhD thesis
    the notation of w's and s's follows from the HDP paper
    '''
    def __repr__(self):
        return 'CRPGamma(concentration=%0.2f)' % self.concentration

    def __init__(self,a_0,b_0,concentration=None):
        self.a_0 = a_0
        self.b_0 = b_0

        if concentration is not None:
            self.concentration = concentration
        else:
            self.resample(niter=1)

    def log_likelihood(self,x):
        raise NotImplementedError # TODO product of gammas

    def rvs(self,customer_counts):
        '''
        Number of distinct tables. Not complete CRPs. customer_counts is a list
        of customer counts, and len(customer_counts) is the number of
        restaurants.
        '''
        assert isinstance(customer_counts,list) or isinstance(customer_counts,int)
        if isinstance(customer_counts,int):
            customer_counts = [customer_counts]

        restaurants = []
        for num in customer_counts:
            tables = []
            for c in range(num):
                newidx = sample_discrete(np.array(tables + [self.concentration]))
                if newidx == len(tables):
                    tables += [1]
                else:
                    tables[newidx] += 1
            restaurants.append(tables)
        return restaurants if len(restaurants) > 1 else restaurants[0]

    def resample(self,data=[],niter=20):
        for itr in range(niter):
            a_n, b_n = self._posterior_hypparams(*self._get_statistics(data))
            self.concentration = np.random.gamma(a_n,scale=1./b_n)

    def _posterior_hypparams(self,sample_numbers,total_num_distinct):
        # NOTE: this is a stochastic function
        if total_num_distinct > 0:
            sample_numbers = np.array(sample_numbers)
            sample_numbers = sample_numbers[sample_numbers > 0]

            wvec = np.random.beta(self.concentration+1,sample_numbers)
            svec = np.array(stats.bernoulli.rvs(sample_numbers/(sample_numbers+self.concentration)))
            return self.a_0 + total_num_distinct-svec.sum(), (self.b_0 - np.log(wvec).sum())
        else:
            return self.a_0, self.b_0

    def _get_statistics(self,data):
        # data is a list of CRP samples, each of which is written as a list of
        # counts of customers at tables, i.e.
        # [5 7 2 ... 1 ]
        assert isinstance(data,list)
        if len(data) == 0:
            sample_numbers = 0
            total_num_distinct = 0
        else:
            if isinstance(data[0],list):
                sample_numbers = np.array(map(sum,data))
                total_num_distinct = sum(map(len,data))
            else:
                sample_numbers = np.array(sum(data))
                total_num_distinct = len(data)
        return sample_numbers, total_num_distinct


class DirGamma(CRPGamma):
    '''
    Implements a Gamma(a_0,b_0) prior over finite dirichlet concentration
    parameter. The concentration is scaled according to the weak-limit according
    to the number of dimensions K.

    For each set of counts i, the model is
        concentration ~ Gamma(a_0,b_0)
        pi_i ~ Dir(concentration/K)
        data_i ~ Multinomial(pi_i)
    '''
    def __repr__(self):
        return 'DirGamma(concentration=%0.2f/%d)' % (self.concentration*self.K,self.K)

    def __init__(self,K,a_0,b_0,concentration=None):
        self.K = K
        super(DirGamma,self).__init__(a_0=a_0,b_0=b_0,
                concentration=concentration)

    def rvs(self,sample_counts):
        if isinstance(sample_counts,int):
            sample_counts = [sample_counts]
        out = np.empty((len(sample_counts),self.K),dtype=int)
        for idx,c in enumerate(sample_counts):
            out[idx] = np.random.multinomial(c,
                np.random.dirichlet(self.concentration * np.ones(self.K)))
        return out if out.shape[0] > 1 else out[0]

    def resample(self,data=[],niter=50,weighted_cols=None):
        if weighted_cols is not None:
            self.weighted_cols = weighted_cols
        else:
            self.weighted_cols = np.ones(self.K)

        if isinstance(data,np.ndarray):
            size = data.sum()
        else:
            assert isinstance(data,list)
            size = sum(d.sum() for d in data)

        if size > 0:
            for itr in range(niter):
                super(DirGamma,self).resample(data,niter=1)
                self.concentration /= self.K
        else:
            super(DirGamma,self).resample(data,niter=1)
            self.concentration /= self.K


    def _get_statistics(self,data):
        counts = np.array(data,ndmin=2)

        # sample m's
        if counts.sum() == 0:
            return 0, 0
        else:
            m = 0
            for (i,j), n in np.ndenumerate(counts):
                m += (np.random.rand(n) < self.concentration*self.K*self.weighted_cols[j] \
                        / (np.arange(n)+self.concentration*self.K*self.weighted_cols[j])).sum()
            return counts.sum(1), m

