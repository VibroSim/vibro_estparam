import numpy as np
import scipy.integrate
import scipy as sp

import theano
import theano.tensor as tt

import vibro_estparam.mixednoise
from vibro_estparam.mixednoise import mixednoise_op,CreateMixedNoise
import pymc3 as pm

check_gradient = False
if check_gradient:
    import theano.tests.unittest_tools
    pass

if __name__=="__main__": # multiprocessing compatiblity
    # Some synthetic observed data
    real_sigma_additive = 22e-3
    real_sigma_multiplicative = 0.25
    n=2 # 20
    
    data_logn_mean = np.log(10.0)
    data_logn_sigma = 2.0

    #data_samples = np.random.lognormal(mean=data_logn_mean,sigma=data_logn_sigma,size=n)
    
    #data_samples=np.array([13.18510493])
    #data_samples=np.array([14.69922805])
    data_samples=np.array([51.98531421,51.98531421,])
    
    # model: observed=coefficient*data_samples
    # noisy model: observed = additive + coefficient*data_samples*multiplicative
    real_coefficient = 5.0

    #observed_samples = np.random.normal(loc=0.0,scale=real_sigma_additive,size=n) + data_samples*real_coefficient*np.random.lognormal(mean=0.0,sigma=real_sigma_multiplicative,size=n)
    observed_samples = np.array([225.3339186,225.3339186,])
    
    #observed_samples = np.array([128.15821403])
    #observed_samples=np.array([0.6180382])
    #observed_samples=np.array([80.76799627])
    
    # Verify that integrate_lognormal_normal_kernel() is a pdf over observed_indexed, i.e. that it integrates to 1.0
    
    pdf_integral = scipy.integrate.quad(lambda obs: mixednoise_op.integrate_kernel(mixednoise_op.lognormal_normal_convolution_integral_y_zero_to_eps,mixednoise_op.lognormal_normal_convolution_kernel,real_sigma_additive,real_sigma_multiplicative,3.0,obs),.0001,np.inf)[0]
    print("pdf_integral=%g (should be 1.0)" % (pdf_integral))
    
    MixedNoiseOp=mixednoise_op(observed_samples) 
    
    orig_ctv = theano.config.compute_test_value
    theano.config.compute_test_value = "off"
    #theano.config.optimizer="None" # "fast_compile" # disable optimizations
    
    #theano.config.exception_verbosity="high"
    
    # Parameters: (sigma_additive, sigma_multiplicative, prediction)
    
    # Evaluation through the Theano Op
    res=MixedNoiseOp(theano.shared(real_sigma_additive),theano.shared(real_sigma_multiplicative),theano.shared(data_samples*real_coefficient)).eval()
    
    #assert((~np.isnan(res)).all())
    
    # Evaluation directly
    res2=np.array([np.log(mixednoise_op.integrate_kernel(mixednoise_op.lognormal_normal_convolution_integral_y_zero_to_eps,mixednoise_op.lognormal_normal_convolution_kernel,real_sigma_additive,real_sigma_multiplicative,data_sample*real_coefficient,observed_sample)) for (data_sample,observed_sample) in zip(data_samples,observed_samples)])

    #assert((res==res2).all())  # Two evaluations should match exactly
    # two evaluations are slightly different because res2 always uses the unaccelerated integrator
    assert(np.linalg.norm(res-res2)/np.linalg.norm(res) < 1e-10)
    
    # Evaluation of derivative:
    
    # ... by unaccelerated
    vibro_estparam.mixednoise.use_accel=False
    MixedNoiseOp.evaluation_cache={}
    
    deriv1u = MixedNoiseOp.grad_sigma_additive_op(theano.shared(real_sigma_additive),theano.shared(real_sigma_multiplicative),theano.shared(data_samples*real_coefficient)).eval()
    deriv2u = MixedNoiseOp.grad_sigma_multiplicative_op(theano.shared(real_sigma_additive),theano.shared(real_sigma_multiplicative),theano.shared(data_samples*real_coefficient)).eval()
    deriv3u = MixedNoiseOp.grad_prediction_op(theano.shared(real_sigma_additive),theano.shared(real_sigma_multiplicative),theano.shared(data_samples*real_coefficient)).eval()
    
    # ... by accelerated
    vibro_estparam.mixednoise.use_accel=True
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
    
    
        theano.tests.unittest_tools.verify_grad(MixedNoiseOp,[real_sigma_additive,real_sigma_multiplicative,data_samples*real_coefficient],abs_tol=1e-8,rel_tol=2e-3,eps=1e-4) 
        pass


    pass
        
