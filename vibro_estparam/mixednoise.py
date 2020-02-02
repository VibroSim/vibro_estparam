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

check_gradient = False
if check_gradient:
    import theano.tests.unittest_tools
    pass

import pymc3 as pm
import pandas as pd
from theano.compile.ops import as_op
from theano.gradient import grad_not_implemented


class mixednoise_op(gof.Op):
    __props__ = ()
    itypes = None
    otypes = None

    observed = None
    
    def __init__(self,observed):

        self.observed=observed
        
        self.itypes = [tt.dscalar,tt.dscalar,tt.dvector] # sigma_additive, sigma_multiplicative, prediction
        self.otypes = [tt.dvector]
        
        self.grad_sigma_additive_op = as_op(itypes=[tt.dscalar,tt.dscalar,tt.dvector],otypes=[tt.dvector])(self.grad_sigma_additive,) # infer_shape=lambda node,input_shapes: [ )
        self.grad_sigma_multiplicative_op = as_op(itypes=[tt.dscalar,tt.dscalar,tt.dvector],otypes=[tt.dvector])(self.grad_sigma_multiplicative)
        
        self.grad_prediction_op = as_op(itypes=[tt.dscalar,tt.dscalar,tt.dvector],otypes=[tt.dvector])(self.grad_prediction)
        
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
        return (1.0/(y*sigma_multiplicative*np.sqrt(2.0*np.pi)))*np.exp(-((np.log(y)-np.log(prediction_indexed))**2.0)/(2.0*sigma_multiplicative**2.0))*(1.0/(sigma_additive*np.sqrt(2.0*np.pi)))*np.exp(-((observed_indexed-y)**2.0)/(2.0*sigma_additive**2.0))

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
        return cls.lognormal_normal_convolution_kernel(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,y)*( (((observed_indexed-y)**2.0)/(sigma_additive**3.0)) - (1.0/sigma_additive))


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

        singular_portion = integral_y_zero_to_eps(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,eps)
        # Break integration into singular portion, portion up to observed value, portion to infinity to help make sure quadrature is accurate. 
        (p1,p1err) = scipy.integrate.quad(lambda y: kernel(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,y),eps,observed_indexed,epsabs=3e-15)
        (p2,p2err) = scipy.integrate.quad(lambda y: kernel(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,y),observed_indexed,np.inf,epsabs=1e-24)
        #print("integrate_kernel returns %s from %s, %s, and %s; p1err=%g, p2err=%g" % (str(singular_portion+p1+p2),str(singular_portion),str(p1),str(p2),p1err,p2err))
        
        return singular_portion + p1 + p2
    
        
    def perform(self,node,inputs_storage,outputs_storage):
        (sigma_additive, sigma_multiplicative, prediction) = inputs_storage

        logp = np.zeros(self.observed.shape[0],dtype='d')

        for index in range(self.observed.shape[0]):
            prediction_indexed=prediction[index]
            observed_indexed = self.observed[index]

            #p = self.integrate_lognormal_normal_kernel(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed)
            p = self.integrate_kernel(self.lognormal_normal_convolution_integral_y_zero_to_eps,
                                      self.lognormal_normal_convolution_kernel,
                                      sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed)
            logp[index]=np.log(p)
            
            pass

        outputs_storage[0][0]=logp
        pass

    def grad_sigma_additive(self,sigma_additive,sigma_multiplicative,prediction):
        # gradient of log p is (1/p) dp
        dlogp = np.zeros(self.observed.shape[0],dtype='d')
        for index in range(self.observed.shape[0]):
            prediction_indexed=prediction[index]
            observed_indexed = self.observed[index]
            
            p = self.integrate_kernel(self.lognormal_normal_convolution_integral_y_zero_to_eps,
                                      self.lognormal_normal_convolution_kernel,
                                      sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed)
            
            dp = self.integrate_kernel(self.lognormal_normal_convolution_integral_y_zero_to_eps_deriv_sigma_additive,
                                       self.lognormal_normal_convolution_kernel_deriv_sigma_additive,
                                       sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed)
            dlogp[index] = (1.0/p) * dp
            pass

        #print("grad_sigma_additive() returns %s from p = %s and dp = %s" % (str(dlogp),str(p),str(dp)))
        
        return dlogp


    def grad_sigma_multiplicative(self,sigma_additive,sigma_multiplicative,prediction):
        # gradient of log p is (1/p) dp
        dlogp = np.zeros(self.observed.shape[0],dtype='d')
        for index in range(self.observed.shape[0]):
            prediction_indexed=prediction[index]
            observed_indexed = self.observed[index]
            
            p = self.integrate_kernel(self.lognormal_normal_convolution_integral_y_zero_to_eps,
                                      self.lognormal_normal_convolution_kernel,
                                      sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed)
            
            dp = self.integrate_kernel(self.lognormal_normal_convolution_integral_y_zero_to_eps_deriv_sigma_multiplicative,
                                       self.lognormal_normal_convolution_kernel_deriv_sigma_multiplicative,
                                       sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed)
            dlogp[index] = (1.0/p) * dp
            pass

        #if np.isnan(dlogp).any():
        #    import pdb
        #    pdb.set_trace()
        
        return dlogp

    def grad_prediction(self,sigma_additive,sigma_multiplicative,prediction):
        # gradient of log p is (1/p) dp
        dlogp = np.zeros(self.observed.shape[0],dtype='d')
        for index in range(self.observed.shape[0]):
            prediction_indexed=prediction[index]
            observed_indexed = self.observed[index]
            
            p = self.integrate_kernel(self.lognormal_normal_convolution_integral_y_zero_to_eps,
                                      self.lognormal_normal_convolution_kernel,
                                      sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed)
            
            dp = self.integrate_kernel(self.lognormal_normal_convolution_integral_y_zero_to_eps_deriv_prediction,
                                       self.lognormal_normal_convolution_kernel_deriv_prediction,
                                       sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed)
            dlogp[index] = (1.0/p) * dp
            pass

        #print("grad_prediction() returns %s from p = %s and dp = %s" % (str(dlogp),str(p),str(dp)))

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

    # Some synthetic observed data
    real_sigma_additive = 22e-3
    real_sigma_multiplicative = 0.25
    n=20

    data_logn_mean = np.log(10.0)
    data_logn_sigma = 2.0

    data_samples = np.random.lognormal(mean=data_logn_mean,sigma=data_logn_sigma,size=n)

    #data_samples=np.array([14.69922805])
    #data_samples=np.array([0.15086759])
    
    # model: observed=coefficient*data_samples
    # noisy model: observed = additive + coefficient*data_samples*multiplicative
    real_coefficient = 5.0
    
    observed_samples = np.random.normal(loc=0.0,scale=real_sigma_additive,size=n) + data_samples*real_coefficient*np.random.lognormal(mean=0.0,sigma=real_sigma_multiplicative,size=n)
    #observed_samples = np.array([128.15821403])
    #observed_samples=np.array([0.6180382])
    
    # Verify that integrate_lognormal_normal_kernel() is a pdf over observed_indexed, i.e. that it integrates to 1.0
    
    pdf_integral = scipy.integrate.quad(lambda obs: mixednoise_op.integrate_kernel(mixednoise_op.lognormal_normal_convolution_integral_y_zero_to_eps,mixednoise_op.lognormal_normal_convolution_kernel,real_sigma_additive,real_sigma_multiplicative,3.0,obs),.0001,np.inf)[0]
    print("pdf_integral=%g (should be 1.0)" % (pdf_integral))
    
    MixedNoiseOp=mixednoise_op(observed_samples) 
    
    orig_ctv = theano.config.compute_test_value
    #theano.config.compute_test_value = "off"
    theano.config.optimizer="None" # "fast_compile" # disable optimizations

    theano.config.exception_verbosity="high"
    
    # Parameters: (sigma_additive, sigma_multiplicative, prediction)

    # Evaluation through the Theano Op
    res=MixedNoiseOp(theano.shared(real_sigma_additive),theano.shared(real_sigma_multiplicative),theano.shared(data_samples*real_coefficient)).eval()

    # Evaluation directly
    res2=np.array([np.log(mixednoise_op.integrate_kernel(mixednoise_op.lognormal_normal_convolution_integral_y_zero_to_eps,mixednoise_op.lognormal_normal_convolution_kernel,real_sigma_additive,real_sigma_multiplicative,data_sample*real_coefficient,observed_sample)) for (data_sample,observed_sample) in zip(data_samples,observed_samples)])

    #assert((res==res2).all())  # Two evaluations should match exactly
    

    # Evaluation of derivative:
    
    deriv1 = MixedNoiseOp.grad_sigma_additive_op(theano.shared(real_sigma_additive),theano.shared(real_sigma_multiplicative),theano.shared(data_samples*real_coefficient)).eval()
    deriv2 = MixedNoiseOp.grad_sigma_multiplicative_op(theano.shared(real_sigma_additive),theano.shared(real_sigma_multiplicative),theano.shared(data_samples*real_coefficient)).eval()
    deriv3 = MixedNoiseOp.grad_prediction_op(theano.shared(real_sigma_additive),theano.shared(real_sigma_multiplicative),theano.shared(data_samples*real_coefficient)).eval()


    
    
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
        
    pass
