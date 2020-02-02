import sys
import os
import os.path
import glob
import collections
import re
import numpy as np
import scipy as sp
import scipy.integrate
import scipy.special
import theano
import theano.tensor as tt
from theano import gof


use_accel = True
if use_accel:
    from . import mixednoise_accel
    pass

import pymc3 as pm
#import pandas as pd
from theano.compile.ops import as_op
from theano.gradient import grad_not_implemented


class mixednoise_op(gof.Op):
    __props__ = ()
    itypes = None
    otypes = None

    observed = None   # Note that "observed" MUST NOT BE CHANGED unless you clear the evaluation_cache
    evaluation_cache = None
    
    def __init__(self,observed):

        self.observed=observed
        
        self.itypes = [tt.dscalar,tt.dscalar,tt.dvector] # sigma_additive, sigma_multiplicative, prediction
        self.otypes = [tt.dvector]
        
        self.grad_sigma_additive_op = as_op(itypes=[tt.dscalar,tt.dscalar,tt.dvector],otypes=[tt.dvector])(self.grad_sigma_additive,) # infer_shape=lambda node,input_shapes: [ )
        self.grad_sigma_multiplicative_op = as_op(itypes=[tt.dscalar,tt.dscalar,tt.dvector],otypes=[tt.dvector])(self.grad_sigma_multiplicative)
        
        self.grad_prediction_op = as_op(itypes=[tt.dscalar,tt.dscalar,tt.dvector],otypes=[tt.dvector])(self.grad_prediction)

        self.evaluation_cache = {}
        
        pass
        
    def infer_shape(self,node,input_shapes):
        return [ (self.observed.shape[0],) ]

    @staticmethod
    def lognormal_normal_convolution_kernel(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,y):
        # y is variable of integration
        # Formula is for pdf of (a0*n1 + n2) evaluated at x = observed value for a  where a0 = prediction
        # n1 ~ lognormal(0,sigma_multiplicative^2)
        # a0n1 ~ lognormal(ln(a0),sigma_multiplicative^2)
        # n2 ~ normal(0,sigma_additive^2)
        ret = (1.0/(y*sigma_multiplicative*np.sqrt(2.0*np.pi)))*np.exp(-((np.log(y)-np.log(prediction_indexed))**2.0)/(2.0*sigma_multiplicative**2.0))*(1.0/(sigma_additive*np.sqrt(2.0*np.pi)))*np.exp(-((observed_indexed-y)**2.0)/(2.0*sigma_additive**2.0))
        #print("kernel(%g,%g,%g,%g,%g) returns %g\n" % (y,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,ret))
        return ret

    @staticmethod
    def lognormal_normal_convolution_integral_y_zero_to_eps(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,eps):

        # ... Treating y=0 in additive noise exponent (because y presumed small relative to observed): 
        # Integral as y = 0...eps of (1.0/(y*sigma_multiplicative*np.sqrt(2.0*np.pi)))*np.exp(-((np.log(y)-np.log(prediction))**2.0)/(2.0*sigma_multiplicative**2.0))*(1.0/(sigma_additive*np.sqrt(2.0*np.pi)))*np.exp(-((observed)**2.0)/(2.0*sigma_additive**2.0))
        # = (1.0/(sigma_multiplicative*sigma_additive*2*pi)) *np.exp(-((observed)**2.0)/(2.0*sigma_additive**2.0)) * Integral as y = 0...eps of (1.0/y)*np.exp(-((np.log(y)-np.log(prediction))**2.0)/(2.0*sigma_multiplicative**2.0))
        # By Wolfram Alpha
        # = (1.0/(sigma_multiplicative*sigma_additive*2*pi)) *np.exp(-((observed)**2.0)/(2.0*sigma_additive**2.0)) * (1/2)*sqrt(pi)*sqrt(2)*sigma_multiplicative*erf((log(y)-log(prediction))/(sqrt(2)*sigma_multiplicative)) evaluated from y=0...eps
        # = (1.0/(sigma_multiplicative*sigma_additive*2*pi)) *np.exp(-((observed)**2.0)/(2.0*sigma_additive**2.0)) * (1/2)*sqrt(pi)*sqrt(2)*sigma_multiplicative*( erf((log(eps)-log(prediction))/(sqrt(2)*sigma_multiplicative))- erf((log(0)-log(prediction))/(sqrt(2)*sigma_multiplicative))) ... where log(0) is -inf  and erf(-inf)= -1
        # = (1.0/(sigma_multiplicative*sigma_additive*2*pi)) *np.exp(-((observed)**2.0)/(2.0*sigma_additive**2.0)) * (1/2)*sqrt(pi)*sqrt(2)*sigma_multiplicative*( erf((log(eps)-log(prediction))/(sqrt(2)*sigma_multiplicative)) + 1)

        return (1.0/(sigma_multiplicative*sigma_additive*2.0*np.pi)) *np.exp(-((observed_indexed)**2.0)/(2.0*sigma_additive**2.0)) * (1.0/2.0)*np.sqrt(np.pi)*np.sqrt(2.0)*sigma_multiplicative*( scipy.special.erf((np.log(eps)-np.log(prediction_indexed))/(np.sqrt(2.0)*sigma_multiplicative)) + 1.0)
        
    
    @classmethod
    def lognormal_normal_convolution_kernel_deriv_sigma_additive(cls,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,y):
        # y is variable of integration
        # Formula is for pdf of (a0*n1 + n2) evaluated at x = observed value for a  where a0 = prediction
        # n1 ~ lognormal(0,sigma_multiplicative^2)
        # a0n1 ~ lognormal(ln(a0),sigma_multiplicative^2)
        # n2 ~ normal(0,sigma_additive^2)
        res = cls.lognormal_normal_convolution_kernel(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,y)*( (((observed_indexed-y)**2.0)/(sigma_additive**3.0)) - (1.0/sigma_additive))
        print("kernel_dsa_unaccel(%g,%g,%g,%g,%g) returns %g\n" % (y,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,res))
        return res
        

    @classmethod
    def lognormal_normal_convolution_integral_y_zero_to_eps_deriv_sigma_additive(cls,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,eps):
        # ... Treating y=0 in additive noise exponent (because y presumed small relative to observed): 
        # Integral as y = 0...eps of (1.0/(y*sigma_multiplicative*np.sqrt(2.0*np.pi)))*np.exp(-((np.log(y)-np.log(prediction))**2.0)/(2.0*sigma_multiplicative**2.0))*(1.0/(sigma_additive*np.sqrt(2.0*np.pi)))*np.exp(-((observed)**2.0)/(2.0*sigma_additive**2.0)) * ( (((observed_indexed)**2.0)/(sigma_additive**3.0)) - (1.0/sigma_additive) )
        # = (1.0/(sigma_multiplicative*sigma_additive*2*pi)) *np.exp(-((observed)**2.0)/(2.0*sigma_additive**2.0)) * ( (((observed_indexed)**2.0)/(sigma_additive**3.0)) - (1.0/sigma_additive) ) * Integral as y = 0...eps of (1.0/y)*np.exp(-((np.log(y)-np.log(prediction))**2.0)/(2.0*sigma_multiplicative**2.0))
        # By Wolfram Alpha
        # = (1.0/(sigma_multiplicative*sigma_additive*2*pi)) *np.exp(-((observed)**2.0)/(2.0*sigma_additive**2.0)) * ( (((observed_indexed)**2.0)/(sigma_additive**3.0)) - (1.0/sigma_additive) ) * (1/2)*sqrt(pi)*sqrt(2)*sigma_multiplicative*erf((log(y)-log(prediction))/(sqrt(2)*sigma_multiplicative)) evaluated from y=0...eps
        # = (1.0/(sigma_multiplicative*sigma_additive*2*pi)) *np.exp(-((observed)**2.0)/(2.0*sigma_additive**2.0)) * ( (((observed_indexed)**2.0)/(sigma_additive**3.0)) - (1.0/sigma_additive) ) * (1/2)*sqrt(pi)*sqrt(2)*sigma_multiplicative*( erf((log(eps)-log(prediction))/(sqrt(2)*sigma_multiplicative))- erf((log(0)-log(prediction))/(sqrt(2)*sigma_multiplicative))) ... where log(0) is -inf  and erf(-inf)= -1
        # = (1.0/(sigma_multiplicative*sigma_additive*2*pi)) *np.exp(-((observed)**2.0)/(2.0*sigma_additive**2.0)) * ( (((observed_indexed)**2.0)/(sigma_additive**3.0)) - (1.0/sigma_additive) ) * (1/2)*sqrt(pi)*sqrt(2)*sigma_multiplicative*( erf((log(eps)-log(prediction))/(sqrt(2)*sigma_multiplicative)) + 1)
        # ... reduces to ( (((observed_indexed)**2.0)/(sigma_additive**3.0)) - (1.0/sigma_additive) ) * lognormal_normal_convolution_integral_y_zero_to_eps()

        return ( (((observed_indexed)**2.0)/(sigma_additive**3.0)) - (1.0/sigma_additive) ) * cls.lognormal_normal_convolution_integral_y_zero_to_eps(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,eps)
    

    @classmethod
    def lognormal_normal_convolution_kernel_deriv_sigma_multiplicative(cls,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,y):
        # y is variable of integration
        # Formula is for pdf of (a0*n1 + n2) evaluated at x = observed value for a  where a0 = prediction
        # n1 ~ lognormal(0,sigma_multiplicative^2)
        # a0n1 ~ lognormal(ln(a0),sigma_multiplicative^2)
        # n2 ~ normal(0,sigma_additive^2)
        return cls.lognormal_normal_convolution_kernel(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,y)*( (((np.log(y) - np.log(prediction_indexed))**2.0)/(sigma_multiplicative**3.0)) - (1.0/sigma_multiplicative))

    
    @classmethod
    def lognormal_normal_convolution_integral_y_zero_to_eps_deriv_sigma_multiplicative(cls,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,eps):
        # ... Treating y=0 in additive noise exponent (because y presumed small relative to observed):

        # The effect of the derivative is to multiply the original kernel by ( (((log(y) - log(prediction_indexed))**2.0)/(sigma_multiplicative**3.0)) - (1.0/sigma_multiplicative))
        # of those two terms, the -1/sigma_multiplicative term is just a multiplier... so we can treat it separately like we do for
        # the derivative with respect ot sigma_additive
        one_over_sigmamultiplicative_term =  - (1.0/sigma_multiplicative)  * cls.lognormal_normal_convolution_integral_y_zero_to_eps(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,eps)

        # for the other term, (((log(y) - log(prediction_indexed))**2.0)/(sigma_multiplicative**3.0))
        # this terms goes to infinity as y-> 0 (because log(y)-> -infty).
        # So we have to do the integration differently for this term.
        # As worked out by hand, treating y=0 for the sigma_additive term brings that term out in front...
        #  Integrating the remaining integrand form of (1/y)exp(-(ln(y)-ln(a0))^2/s)((ln(y)-ln(a0))^2) via Wolfram Alpha
        # gives us
        # (1/(sigma_additive*sqrt(2*pi)))*exp(-x^2/(2*sigma_additive^2)) * (1/(sigma_multiplicative^4*sqrt(2pi)))* [ (1/4)sqrt(pi)*2sqrt(2)*sigma_multiplicative^3*erf((ln(y)-ln(a0))/(sigma_multiplicative*sqrt(2)))  -  sigma_multiplicative^2*(ln(y)-ln(a0))*(a0^(ln(y)/sigma_multiplicative^2))*exp(-(ln^2(a0) + ln^2(y))/(2*sigma_multiplicative^2)) ] evaluated from y=0 to y=epsilon
        # As y->0 the erf->-1 ; the limit of the right hand term inside the []'s is simply 0, verified by l'Hopital's rule.
        # So we have:
        # (1/(sigma_additive*sqrt(2*pi)))*exp(-x^2/(2*sigma_additive^2)) * (1/(sigma_multiplicative^4*sqrt(2pi)))* [ (1/4)sqrt(pi)*2sqrt(2)*sigma_multiplicative^3*erf((ln(epsilon)-ln(a0))/(sigma_multiplicative*sqrt(2)))  -  sigma_multiplicative^2*(ln(epsilon)-ln(a0))*(a0^(ln(epsilon)/sigma_multiplicative^2))*exp(-(ln^2(a0) + ln^2(epsilon))/(2*sigma_multiplicative^2)) + (1/4)sqrt(pi)*2sqrt(2)*sigma_multiplicative^3  ]
        # simplify a bit...
        # (1/(sigma_additive*sqrt(2*pi)))*exp(-x^2/(2*sigma_additive^2)) * (1/(sigma_multiplicative^2*sqrt(2pi)))* [ (1/2)sqrt(2*pi)*sigma_multiplicative*erf((ln(epsilon)-ln(a0))/(sigma_multiplicative*sqrt(2)))  -  (ln(epsilon)-ln(a0))*(a0^(ln(epsilon)/sigma_multiplicative^2))*exp(-(ln^2(a0) + ln^2(epsilon))/(2*sigma_multiplicative^2)) + (1/2)sqrt(2pi)*sigma_multiplicative  ]
        additive_factor = (1.0/(sigma_additive*np.sqrt(2.0*np.pi)))*np.exp(-observed_indexed**2.0/(2.0*sigma_additive**2.0))
        #integration_term = additive_factor * (1.0/(sigma_multiplicative**2.0*np.sqrt(2.0*np.pi)))* ( (1.0/2.0)*np.sqrt(2.0*np.pi)*sigma_multiplicative*scipy.special.erf((np.log(eps)-np.log(prediction_indexed))/(sigma_multiplicative*np.sqrt(2.0)))  -  (np.log(eps)-np.log(prediction_indexed))*(prediction_indexed**(np.log(eps)/sigma_multiplicative**2.0))*np.exp(-(np.log(prediction_indexed)**2.0 + np.log(eps)**2.0)/(2.0*sigma_multiplicative**2.0)) + (1.0/2.0)*np.sqrt(2.0*np.pi)*sigma_multiplicative  )
        # ... but the prediction_indexed**(np.log(eps)/sigma_multiplicative**2.0))
        # is numerically problematic (overflow from very large numbers.
        # Replace it and the following exp() per handwritten notes
        # on the basis of (a^b)*exp(c) === exp(c + b*log(a)):
        integration_term = additive_factor * (1.0/(sigma_multiplicative**2.0*np.sqrt(2.0*np.pi)))* ( (1.0/2.0)*np.sqrt(2.0*np.pi)*sigma_multiplicative*scipy.special.erf((np.log(eps)-np.log(prediction_indexed))/(sigma_multiplicative*np.sqrt(2.0)))  -  (np.log(eps)-np.log(prediction_indexed))*np.exp(-(np.log(prediction_indexed)**2.0 + np.log(eps)**2.0)/(2.0*sigma_multiplicative**2.0) + (np.log(eps)/sigma_multiplicative**2.0)*np.log(prediction_indexed) ) + (1.0/2.0)*np.sqrt(2.0*np.pi)*sigma_multiplicative  )

        
        #if np.isnan(integration_term) or np.isnan(one_over_sigmamultiplicative_term):
        #    import pdb
        #    pdb.set_trace()
        #    pass
        
        return integration_term + one_over_sigmamultiplicative_term        
    
    @staticmethod
    def lognormal_normal_convolution_integral_y_zero_to_eps_deriv_prediction(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,eps):
        # ... Treating y=0 in additive noise exponent (because y presumed small relative to observed):

        # The effect of the derivative is to multiply the original kernel by (((np.log(y) - np.log(prediction_indexed))**2.0)/((sigma_multiplicative**2.0)*prediction_indexed))
        # As usual the additive factor in the integral can be pulled out as a constant by neglecting y relative to observed:
        additive_factor = (1.0/(sigma_additive*np.sqrt(2.0*np.pi)))*np.exp(-observed_indexed**2.0/(2.0*sigma_additive**2.0))

        # The remaining integral has the form: integral of (1/y)*exp(-(log(y)-log(c))^2/s)*(log(y)-log(c)) dy
        # by Wolfram Alpha this integrates to -(1/2)*s*exp(-((log(c)-log(y))^2)/s)
        # As worked out on paper, we get
        integration = -1.0/(sigma_multiplicative*prediction_indexed*np.sqrt(2.0*np.pi))*np.exp(-((np.log(prediction_indexed)-np.log(eps))**2.0)/(2.0*sigma_multiplicative**2.0))

        return additive_factor*integration
    
    @classmethod
    def lognormal_normal_convolution_kernel_deriv_prediction(cls,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,y):
        # y is variable of integration
        # Formula is for pdf of (a0*n1 + n2) evaluated at x = observed value for a  where a0 = prediction
        # n1 ~ lognormal(0,sigma_multiplicative^2)
        # a0n1 ~ lognormal(ln(a0),sigma_multiplicative^2)
        # n2 ~ normal(0,sigma_additive^2)
        return cls.lognormal_normal_convolution_kernel(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,y)* ((np.log(y) - np.log(prediction_indexed))/((sigma_multiplicative**2.0)*prediction_indexed))

    #@classmethod
    #def integrate_lognormal_normal_kernel(cls,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed):
    #    # Integration of kernel singularity: Kernel is singular at y=0
    #    # Perform integration from y=0 to y=eps analytically.
    #    # where eps presumed small relative to observed.
    #    eps = observed_indexed/100.0
    #    
    #    
    #    singular_portion = cls.lognormal_normal_convolution_integral_y_zero_to_eps(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,eps)
    #    # Break integration into singular portion, portion up to observed value, portion to infinity to help make sure quadrature is accurate. 
    #    p1 = scipy.integrate.quad(lambda y: cls.lognormal_normal_convolution_kernel(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,y),eps,observed_indexed)[0]
    #    p2 = scipy.integrate.quad(lambda y: cls.lognormal_normal_convolution_kernel(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,y),observed_indexed,np.inf)[0]
    #    return singular_portion + p1 + p2


    @staticmethod
    def integrate_kernel(integral_y_zero_to_eps,kernel,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed):
        # Integration of kernel singularity: Kernel is singular at y=0
        # Perform integration from y=0 to y=eps analytically.
        # where eps presumed small relative to observed.
        eps = observed_indexed/100.0

        bound1=observed_indexed-sigma_additive
        if bound1 < eps:
            bound1=eps
            pass
        
        bound2=observed_indexed+sigma_additive
        
        singular_portion = integral_y_zero_to_eps(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,eps)
        # Break integration into singular portion, portion up to observed value, portion to infinity to help make sure quadrature is accurate.
        print("Integration from y=%g... %g" % (eps,bound1))
        (p1,p1err) = scipy.integrate.quad(lambda y: kernel(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,y),eps,bound1,epsabs=3e-15)
        #print("integral from %g to %g, sa=%g, sm=%g, pi=%g, oi=%g,ea=%g" %(eps,observed_indexed-sigma_additive,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,3e-15))

        print("Integration from y=%g... %g" % (bound1,bound2))
        (p2,p2err) = scipy.integrate.quad(lambda y: kernel(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,y),bound1,bound2,epsabs=3e-15)
        #print("integral from %g to %g, sa=%g, sm=%g, pi=%g, oi=%g,ea=%g" %(eps,observed_indexed,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,3e-15))

        print("Integration from y=%g... inf" % (bound2))

        (p3,p3err) = scipy.integrate.quad(lambda y: kernel(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,y),bound2,np.inf,epsabs=1e-24)
        print("integrate_kernel returns %s from %s, %s, %s, and %s; p1err=%g, p2err=%g,p3err=%g" % (str(singular_portion+p1+p2+p3),str(singular_portion),str(p1),str(p2),str(p3),p1err,p2err,p3err))
        #print("kernel(1,1,1,1,1)=%g" % (kernel(1,1,1,1,1)))
        
        return singular_portion + p1 + p2 + p3

    def evaluate_p_from_cache(self,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed):
        # Evaluating the baseline probability is required both in each call to perform() and in the
        # gradient along each axis. This uses a dictionary as a cache so that they don't need to be
        # recomputed
        key = (float(sigma_additive),float(sigma_multiplicative),float(prediction_indexed),float(observed_indexed))
        if not key in self.evaluation_cache:
            
            #p = self.integrate_lognormal_normal_kernel(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed)

            p = self.integrate_kernel(self.lognormal_normal_convolution_integral_y_zero_to_eps,
                                          self.lognormal_normal_convolution_kernel,
                                          sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed)
        
            
            
            self.evaluation_cache[key]=p
            return p 
        return self.evaluation_cache[key]
        
        
    def perform(self,node,inputs_storage,outputs_storage):
        (sigma_additive, sigma_multiplicative, prediction) = inputs_storage

        logp = np.zeros(self.observed.shape[0],dtype='d')

        if use_accel:
            p = mixednoise_accel.integrate_lognormal_normal_convolution(self.lognormal_normal_convolution_integral_y_zero_to_eps,
                                                                        self.evaluation_cache,
                                                                        sigma_additive,sigma_multiplicative,prediction,self.observed)
            
            logp=np.log(p)
            pass
        else:

            for index in range(self.observed.shape[0]):
                prediction_indexed=prediction[index]
                observed_indexed = self.observed[index]
                
                p = self.evaluate_p_from_cache(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed)
                logp[index]=np.log(p)
                
                pass
            pass
        
        outputs_storage[0][0]=logp
        pass

    def grad_sigma_additive(self,sigma_additive,sigma_multiplicative,prediction):
        # gradient of log p is (1/p) dp
        dlogp = np.zeros(self.observed.shape[0],dtype='d')

        if use_accel:
            p = mixednoise_accel.integrate_lognormal_normal_convolution(self.lognormal_normal_convolution_integral_y_zero_to_eps,
                                                                        self.evaluation_cache,
                                                                        sigma_additive,sigma_multiplicative,prediction,self.observed)
            dp = mixednoise_accel.integrate_deriv_sigma_additive(self.lognormal_normal_convolution_integral_y_zero_to_eps_deriv_sigma_additive,
                                                                 sigma_additive,sigma_multiplicative,prediction,self.observed)

            
            dlogp =  dp/p
            print("accel: p=%s; dp=%s; dlogp = %s" % (str(p),str(dp),str(dlogp)))
            
            pass
        else: 
            for index in range(self.observed.shape[0]):
                prediction_indexed=prediction[index]
                observed_indexed = self.observed[index]
            
                p = self.evaluate_p_from_cache(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed)
                
                dp = self.integrate_kernel(self.lognormal_normal_convolution_integral_y_zero_to_eps_deriv_sigma_additive,
                                           self.lognormal_normal_convolution_kernel_deriv_sigma_additive,
                                           sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed)
                dlogp[index] = (1.0/p) * dp
                pass
            print("unaccel: p=%s; dp=%s; dlogp = %s" % (str(p),str(dp),str(dlogp)))

            pass
        
        #print("grad_sigma_additive() returns %s from p = %s and dp = %s" % (str(dlogp),str(p),str(dp)))
        
        return dlogp


    def grad_sigma_multiplicative(self,sigma_additive,sigma_multiplicative,prediction):
        # gradient of log p is (1/p) dp
        dlogp = np.zeros(self.observed.shape[0],dtype='d')
        if use_accel:
            p = mixednoise_accel.integrate_lognormal_normal_convolution(self.lognormal_normal_convolution_integral_y_zero_to_eps,
                                                                        self.evaluation_cache,
                                                                        sigma_additive,sigma_multiplicative,prediction,self.observed)
            dp = mixednoise_accel.integrate_deriv_sigma_multiplicative(self.lognormal_normal_convolution_integral_y_zero_to_eps_deriv_sigma_multiplicative,
                                                                       sigma_additive,sigma_multiplicative,prediction,self.observed)

            dlogp =  dp/p
            pass
        else:
            for index in range(self.observed.shape[0]):
                prediction_indexed=prediction[index]
                observed_indexed = self.observed[index]
                
                p = self.evaluate_p_from_cache(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed)
                
                dp = self.integrate_kernel(self.lognormal_normal_convolution_integral_y_zero_to_eps_deriv_sigma_multiplicative,
                                           self.lognormal_normal_convolution_kernel_deriv_sigma_multiplicative,
                                           sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed)
                dlogp[index] = (1.0/p) * dp
                pass
            pass
        #if np.isnan(dlogp).any():
        #    import pdb
        #    pdb.set_trace()
        
        return dlogp

    def grad_prediction(self,sigma_additive,sigma_multiplicative,prediction):
        # gradient of log p is (1/p) dp
        dlogp = np.zeros(self.observed.shape[0],dtype='d')

        if use_accel:
            p = mixednoise_accel.integrate_lognormal_normal_convolution(self.lognormal_normal_convolution_integral_y_zero_to_eps,
                                                                        self.evaluation_cache,
                                                                        sigma_additive,sigma_multiplicative,prediction,self.observed)
            dp = mixednoise_accel.integrate_deriv_prediction(self.lognormal_normal_convolution_integral_y_zero_to_eps_deriv_prediction,
                                                             sigma_additive,sigma_multiplicative,prediction,self.observed)
            
            dlogp =  dp/p
            pass
        else:
            for index in range(self.observed.shape[0]):
                prediction_indexed=prediction[index]
                observed_indexed = self.observed[index]
                
                p = self.evaluate_p_from_cache(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed)
                
                dp = self.integrate_kernel(self.lognormal_normal_convolution_integral_y_zero_to_eps_deriv_prediction,
                                           self.lognormal_normal_convolution_kernel_deriv_prediction,
                                           sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed)
                dlogp[index] = (1.0/p) * dp
                pass
            
            #print("grad_prediction() returns %s from p = %s and dp = %s" % (str(dlogp),str(p),str(dp)))
            pass
    
        return dlogp
        
    
    def grad(self,inputs,output_grads):
        (sigma_additive, sigma_multiplicative, prediction) = inputs
        # output_grads has a single element output_grads[0] corresponding to our single vector output
        #
        # ... In evaluating grad_x G(x),
        # where G(x) is representable as C(f(x)) and f(x) is predict_crackheating_op.
        #
        # Then grad_x G(x) = dC/df * df/dx
        # If f is a vector 
        # Then grad_x G(x) = sum_i dC/df_i * df_i/dx
        
        # If x is also a vector 
        # Then grad_xj G(x) = sum_i dC/df_i * df_i/dxj

        # And if xj can be x0 (mu) or x1 (log_msqrtR)
        # Then grad_xj G(x) = (sum_i dC/df_i * df_i/dx0, sum_i dC/df_i * df_i/dx1), 
        
        # 
        # We are supposed to return the tensor product dC/df_i * df_i/dxj where
        #          * dC/df_i is output_gradients[0],
        #          * df_i/dx0 is predict_crackheating_grad_mu_op, and
        #          * df_i/dx1 is predict_crackheating_grad_log_msqrtR_op.
        # Since f is a vectors, dC_df is also a vector
        #    and returning the tensor product means summing over the elements i.
        # From the Theano documentation "The grad method must return a list containing
        #                                one Variable for each input. Each returned Variable
        #                                represents the gradient with respect to that input
        #                                computed based on the symbolic gradients with
        #                                respect to each output."
        # So we return a list indexed over input j

        #import pdb
        #pdb.set_trace()
        
        return [ (self.grad_sigma_additive_op(*inputs)*output_grads[0]).sum(),  (self.grad_sigma_multiplicative_op(*inputs)*output_grads[0]).sum(), (self.grad_prediction_op(*inputs)*output_grads[0]) ]
    

    
    pass



def CreateMixedNoise(name,
                     sigma_additive,
                     sigma_multiplicative,
                     prediction,
                     observed):

    MixedNoiseOp=mixednoise_op(observed) 
    
    def MixedNoiseLogP(sigma_additive,
                       sigma_multiplicative,
                       prediction):
        # captures "MixedNoiseOp"
        return MixedNoiseOp(sigma_additive,sigma_multiplicative,prediction)
    
    
    return pm.DensityDist(name,
                          MixedNoiseLogP,
                          observed={
                              "sigma_additive": sigma_additive,
                              "sigma_multiplicative": sigma_multiplicative,
                              "prediction": prediction
                              })

                              

                              
if __name__=="__main__":
    pass
