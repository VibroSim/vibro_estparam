import numpy as np
import scipy.integrate
import scipy as sp

import theano
import theano.tensor as tt

import vibro_estparam.mixednoise
from vibro_estparam.mixednoise import mixednoise_op
import pymc3 as pm

check_gradient = True
if check_gradient:
    import theano.tests.unittest_tools
    pass

# Some synthetic observed data
real_sigma_additive = 22e-3
real_sigma_multiplicative = 0.25
n=1 # 20

data_logn_mean = np.log(10.0)
data_logn_sigma = 2.0

data_samples = np.random.lognormal(mean=data_logn_mean,sigma=data_logn_sigma,size=n)

#data_samples=np.array([13.18510493])
#data_samples=np.array([14.69922805])
data_samples=np.array([51.98531421])

# model: observed=coefficient*data_samples
# noisy model: observed = additive + coefficient*data_samples*multiplicative
real_coefficient = 5.0

#observed_samples = np.random.normal(loc=0.0,scale=real_sigma_additive,size=n) + data_samples*real_coefficient*np.random.lognormal(mean=0.0,sigma=real_sigma_multiplicative,size=n)
observed_samples = np.array([225.3339186])

#observed_samples = np.array([128.15821403])
#observed_samples=np.array([0.6180382])
#observed_samples=np.array([80.76799627])

# Verify that integrate_lognormal_normal_kernel() is a pdf over observed_indexed, i.e. that it integrates to 1.0

#pdf_integral = scipy.integrate.quad(lambda obs: mixednoise_op.integrate_kernel(mixednoise_op.lognormal_normal_convolution_integral_y_zero_to_eps,mixednoise_op.lognormal_normal_convolution_kernel,real_sigma_additive,real_sigma_multiplicative,3.0,obs),.0001,np.inf)[0]
#print("pdf_integral=%g (should be 1.0)" % (pdf_integral))
    
MixedNoiseOp=mixednoise_op(observed_samples) 
    
orig_ctv = theano.config.compute_test_value
theano.config.compute_test_value = "off"
#theano.config.optimizer="None" # "fast_compile" # disable optimizations

#theano.config.exception_verbosity="high"

# Parameters: (sigma_additive, sigma_multiplicative, prediction)

# Evaluation through the Theano Op
#res=MixedNoiseOp(theano.shared(real_sigma_additive),theano.shared(real_sigma_multiplicative),theano.shared(data_samples*real_coefficient)).eval()

#assert((~np.isnan(res)).all())

# Evaluation directly
#res2=np.array([np.log(mixednoise_op.integrate_kernel(mixednoise_op.lognormal_normal_convolution_integral_y_zero_to_eps,mixednoise_op.lognormal_normal_convolution_kernel,real_sigma_additive,real_sigma_multiplicative,data_sample*real_coefficient,observed_sample)) for (data_sample,observed_sample) in zip(data_samples,observed_samples)])

##assert((res==res2).all())  # Two evaluations should match exactly
## two evaluations are slightly different because res2 always uses the unaccelerated integrator
#assert(np.linalg.norm(res-res2)/np.linalg.norm(res) < 1e-10)

# Evaluation of derivative:

# ... by unaccelerated
vibro_estparam.mixednoise.use_accel=False
MixedNoiseOp.evaluation_cache={}

deriv1u = MixedNoiseOp.grad_sigma_additive_op(theano.shared(real_sigma_additive),theano.shared(real_sigma_multiplicative),theano.shared(data_samples*real_coefficient)).eval()
deriv2u = MixedNoiseOp.grad_sigma_multiplicative_op(theano.shared(real_sigma_additive),theano.shared(real_sigma_multiplicative),theano.shared(data_samples*real_coefficient)).eval()
deriv3u = MixedNoiseOp.grad_prediction_op(theano.shared(real_sigma_additive),theano.shared(real_sigma_multiplicative),theano.shared(data_samples*real_coefficient)).eval()

# ... by accelerated
#vibro_estparam.mixednoise.use_accel=True
vibro_estparam.mixednoise.use_accel=False
MixedNoiseOp.evaluation_cache={}


deriv1 = MixedNoiseOp.grad_sigma_additive_op(theano.shared(real_sigma_additive),theano.shared(real_sigma_multiplicative),theano.shared(data_samples*real_coefficient)).eval()
deriv2 = MixedNoiseOp.grad_sigma_multiplicative_op(theano.shared(real_sigma_additive),theano.shared(real_sigma_multiplicative),theano.shared(data_samples*real_coefficient)).eval()
deriv3 = MixedNoiseOp.grad_prediction_op(theano.shared(real_sigma_additive),theano.shared(real_sigma_multiplicative),theano.shared(data_samples*real_coefficient)).eval()

deriv_unaccelerated=np.array((deriv1u,deriv2u,deriv3u))
deriv_accelerated=np.array((deriv1,deriv2,deriv3))
assert(np.linalg.norm(deriv_unaccelerated-deriv_accelerated)/np.linalg.norm(deriv_unaccelerated) < 1e-10)




    

if check_gradient:
    
    # WARNING: gradient tests sometimes marginally fail, but this should
    # not be a problem
    
    theano.tests.unittest_tools.verify_grad(lambda sig_add_val: MixedNoiseOp(sig_add_val,theano.shared(real_sigma_multiplicative),theano.shared(data_samples*real_coefficient)) ,[ real_sigma_additive,],abs_tol=1e-4,rel_tol=1e-5,eps=1e-6) 
    
    theano.tests.unittest_tools.verify_grad(lambda sig_mul_val: MixedNoiseOp(theano.shared(real_sigma_additive),sig_mul_val,theano.shared(data_samples*real_coefficient)) ,[ real_sigma_multiplicative,],abs_tol=1e-12,rel_tol=1e-5) 
    
    theano.tests.unittest_tools.verify_grad(lambda predict_val: MixedNoiseOp(theano.shared(real_sigma_additive),theano.shared(real_sigma_multiplicative),predict_val) ,[ data_samples*real_coefficient,],abs_tol=1e-12,rel_tol=1e-5,eps=1e-7) 
    
    
    theano.tests.unittest_tools.verify_grad(MixedNoiseOp,[real_sigma_additive,real_sigma_multiplicative,data_samples*real_coefficient],abs_tol=1e-12,rel_tol=1e-5) 
    pass



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
                            observed_samples)
    
    
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
        
