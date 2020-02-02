import os
import sys
import numpy as np
import scipy.integrate
import scipy as sp

import theano
import theano.tensor as tt

import vibro_estparam.mixednoise
from vibro_estparam.mixednoise import mixednoise_op,CreateMixedNoise
import pymc3 as pm

# ... by accelerated
#vibro_estparam.mixednoise.use_accel=True


if __name__=="__main__":

    # Some synthetic observed data
    real_sigma_additive = 22e-3
    real_sigma_multiplicative = 0.25
    n=20 # 20
    
    data_logn_mean = np.log(10.0)
    data_logn_sigma = 2.0

    data_samples = np.random.lognormal(mean=data_logn_mean,sigma=data_logn_sigma,size=n)
    
    #data_samples=np.array([13.18510493])
    #data_samples=np.array([14.69922805])
    #data_samples=np.array([51.98531421,51.98531421,])
    
    # model: observed=coefficient*data_samples
    # noisy model: observed = additive + coefficient*data_samples*multiplicative
    real_coefficient = 5.0

    observed_samples = np.random.normal(loc=0.0,scale=real_sigma_additive,size=n) + data_samples*real_coefficient*np.random.lognormal(mean=0.0,sigma=real_sigma_multiplicative,size=n)
    #observed_samples = np.array([225.3339186,225.3339186,])

    
    model=pm.Model()

    with model:
        sigma_additive_prior_mu = 0.0
        sigma_additive_prior_sigma = 1.0
        sigma_multiplicative_prior_mu = np.log(0.5)
        sigma_multiplicative_prior_sigma = 1.0
        coefficient_prior_mu = np.log(3.0)
        coefficient_prior_sigma = 1.0
    
        # priors for sigma_additive and sigma_multiplicative
        sigma_additive = pm.Lognormal("sigma_additive",mu=sigma_additive_prior_mu,sigma=sigma_additive_prior_sigma)
        sigma_additive_prior=pm.Lognormal.dist(mu=sigma_additive_prior_mu,sigma=sigma_additive_prior_sigma)
    
        
        sigma_multiplicative = pm.Lognormal("sigma_multiplicative",mu=sigma_multiplicative_prior_mu,sigma=sigma_multiplicative_prior_sigma)
        sigma_multiplicative_prior = pm.Lognormal.dist(mu=sigma_multiplicative_prior_mu,sigma=sigma_multiplicative_prior_sigma)
        
        
        coefficient = pm.Lognormal("coefficient",mu=coefficient_prior_mu,sigma=coefficient_prior_sigma)
    
        coefficient_prior = pm.Lognormal.dist(mu=coefficient_prior_mu,sigma=coefficient_prior_sigma)
    
    
        like = CreateMixedNoise("like",
                                sigma_additive,
                                sigma_multiplicative,
                                data_samples*coefficient,
                                observed_samples,
                                inhibit_accel_pid=os.getpid())
        
        
        step = pm.NUTS()
        trace = pm.sample(100,step=step,chains=4,cores=4,tune=25)
        #trace = pm.sample(100,step=step,chains=4,cores=1,tune=25)
        pass
    
    from matplotlib import pyplot as pl
    pm.traceplot(trace)
    
    sigma_additive_hist = pl.figure()
    pl.clf()
    pl.hist(trace.get_values("sigma_additive"),bins=30,density=True)
    sa_range=np.linspace(0,pl.axis()[1],100)
    pl.plot(sa_range,np.exp(sigma_additive_prior.logp(sa_range).eval()),'-')
    pl.xlabel('sigma_additive')
    pl.title('median=%f; real value=%f' % (np.median(trace.get_values("sigma_additive")),real_sigma_additive))
    pl.grid()
    
    sigma_multiplicative_hist = pl.figure()
    pl.clf()
    pl.hist(trace.get_values("sigma_multiplicative"),bins=30,density=True)
    sm_range=np.linspace(0,pl.axis()[1],100)
    pl.plot(sm_range,np.exp(sigma_multiplicative_prior.logp(sm_range).eval()),'-')
    pl.xlabel('sigma_multiplicative')
    pl.title('median=%f; real value=%f' % (np.median(trace.get_values("sigma_multiplicative")),real_sigma_multiplicative))
    pl.grid()

    coefficient_hist = pl.figure()
    pl.clf()
    pl.hist(trace.get_values("coefficient"),bins=30,density=True)
    c_range=np.linspace(0,pl.axis()[1],100)
    pl.plot(c_range,np.exp(coefficient_prior.logp(c_range).eval()),'-')
    pl.xlabel('coefficient')
    pl.title('median=%f; real value=%f' % (np.median(trace.get_values("coefficient")),real_coefficient))
    pl.grid()


        
    pl.show()

