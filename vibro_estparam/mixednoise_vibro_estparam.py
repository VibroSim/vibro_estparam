# mixednoise_vibro_estparam --
# a mixed noise model that leverages the general-purpose mixednoise.py implementation
# to do an application-specific calculation with differentiation based on finite differences
# that numerically bypasses all of the intermediate partial derivatives


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
from theano.printing import Print
from theano import gof

import pymc3 as pm
#import pandas as pd
from theano.compile.ops import as_op

from . import mixednoise
from .mixednoise_accel import integrate_lognormal_normal_convolution

def gradient(location, function, rel_precision=1e-3,abs_precision=1e-8,initialrelstep=1e-4,initialabsstep=1e-5,minrelstep=1e-9,debug=False):
    
    n_axes = location.shape[0]
    grad = np.zeros(n_axes,dtype='d')
    non_convergences = 0

    center_val = function(location)

    if not np.isfinite(center_val):
        non_convergences+=n_axes
        print("gradients: function value @ location=%s is not finite; treating gradient as zero" % (str(location)))
        return (grad,non_convergences)
    
    for axis in range(n_axes):

        max_flipflops = 10
        initial_step = abs(location[axis])*initialrelstep

        if initial_step < initialabsstep:
            # don't let initial step distance be less than initialabsstep
            initial_step = initialabsstep
            pass

        step = initial_step
        done=False
        armed=False
        num_flipflops = 0
        last_deriv=0.0
        
        if debug:
            print("gradient: axis=%d; location=%g; initial_step=%g" % (axis,location[axis],initial_step))
            pass

        while not done:

            location_1 = location.copy()
            location_1[axis] -= step/2
            val_1 = function(location_1)

            location_2 = location.copy()
            location_2[axis] += step/2
            val_2 = function(location_2)

            apparent_derivatives = np.array(((val_2 - val_1)/step,
                                             (val_2 - center_val)*2.0/step,
                                             (center_val - val_1)*2.0/step),dtype='d')

            if debug:
                print("apparent_derivatives=%g; %g %g " % (apparent_derivatives[0],apparent_derivatives[1],apparent_derivatives[2]))
                pass
                
            deriv_diffs = apparent_derivatives[:,np.newaxis]-apparent_derivatives[np.newaxis,:]
            deriv_mean = np.mean(apparent_derivatives)
            max_deriv_diff = np.amax(abs(deriv_diffs))
            if (max_deriv_diff > rel_precision*abs(deriv_mean) and max_deriv_diff > abs_precision) or not np.isfinite(apparent_derivatives).all() :
                armed=False
                gooditer=False
                pass
            else:
                gooditer=True
                pass

            # check for flip/flop
            deriv_ratio = last_deriv/apparent_derivatives[0]
            if np.isfinite(deriv_ratio) and deriv_ratio < 0.0:
                num_flipflops += 1
                pass

            if debug:
                print("armed=%s; gooditer=%s; flip_flop=%s" % (str(armed),str(gooditer),str(np.isfinite(deriv_ratio) and deriv_ratio < 0.0)))
                pass

            if armed and gooditer:
                deriv_diff = abs(apparent_derivatives[0] - last_deriv)
                print("deriv_diff = %g; rel_prec=%g; appar_dev=%g; last_deriv=%g" % (deriv_diff,rel_precision,apparent_derivatives[0],last_deriv))
                if deriv_diff < rel_precision*abs(apparent_derivatives[0]) or deriv_diff < abs_precision:
                    # Match... success!
                    derivative = apparent_derivatives[0]
                    done=True
                    pass
                pass


            
            # Prepare for next iteration
            if gooditer:
                armed=True
                pass
            last_deriv = apparent_derivatives[0]
            
            step=step/2.0
            if step < abs(location[axis])*minrelstep or num_flipflops > max_flipflops:
                # step size too small or too many flipflops
                print("gradients: gradient on axis %d did not converge @ location=%s; treating as zero" % (axis,location))
                non_convergences += 1
                derivative = 0.0
                done=True
                pass
            
            pass
        if debug:
            print("grad[%d]=%g" % (axis,derivative))
            pass
        
        grad[axis]=derivative
        pass
    
    return (grad,non_convergences)


class mixednoise_vibro_estparam_op(gof.Op):
    __props__ = ()
    itypes = None
    otypes = None

    observed = None   # Note that "observed" MUST NOT BE CHANGED unless you clear the evaluation_cache
    prediction_function = None # prediction function should take two arguments: mu and log_msqrtR, and return an array of the same size as "observed"
    evaluation_cache = None
    inhibit_accel_pid = None # Set this to a pid to prevent acceleration from happening in this pid. Used to prevent openmp parallelism in main process that causes python multiprocessing (used by pymc3) to bork.
    
    non_convergences=None
    num_derivatives=None
    
    def __init__(self,observed,prediction_function,inhibit_accel_pid=None):

        self.observed=observed
        self.prediction_function = prediction_function
        self.inhibit_accel_pid=inhibit_accel_pid
        
        self.itypes = [tt.dvector] # Sigma_additive, sigma_multiplicative, mu, log_msqrtR collected into a vector
        self.otypes = [tt.dscalar]

        self.numerical_grad_op = as_op(itypes=[tt.dvector],otypes=[tt.dvector])(self.numerical_grad)

        self.evaluation_cache = {}
        self.non_convergences=0
        self.num_derivatives=0
        pass

    def evaluate_logp_total_from_cache(self,sigma_additive,sigma_multiplicative,mu,log_msqrtR):
        # Evaluating the baseline probability is required both in each call to perform() and in the
        # gradient along each axis. This uses a dictionary as a cache so that they don't need to be
        # recomputed
        key = (float(sigma_additive),float(sigma_multiplicative),float(mu),float(log_msqrtR))
        if not key in self.evaluation_cache:
            
            #p = self.integrate_lognormal_normal_kernel(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed)

            prediction = self.prediction_function(mu,log_msqrtR)

            #print("prediction=%s" % (str(prediction)))

            if mixednoise.use_accel and self.inhibit_accel_pid != os.getpid():
                p_array = integrate_lognormal_normal_convolution(mixednoise.mixednoise_op.lognormal_normal_convolution_integral_y_zero_to_eps,
                                                                                  None, # our evaluation cache is not compatible with that expected by mixednoise_accel
                                                                                  sigma_additive,
                                                                                  sigma_multiplicative,
                                                                                  prediction,
                                                                                  self.observed)
                #print("p_array=%s" % (str(p_array)))
                logp_total = np.sum(np.log(p_array))
                pass

            else:
                # acceleration not available... just iterate
                logp_total=0.0
                for index in range(self.observed.shape[0]):
                    prediction_indexed = prediction[index];
                    observed_indexed=self.observed[index]
            
                    p = mixednoise.mixednoise_op.integrate_kernel(mixednoise.mixednoise_op.lognormal_normal_convolution_integral_y_zero_to_eps,
                                                                  mixednoise.mixednoise_op.lognormal_normal_convolution_kernel,
                                                                  sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed)
                    
                    #print("p[%d]=%s" % (index,str(p)))
                    logp_total += np.log(p)
                    pass
                pass
            
            
            self.evaluation_cache[key]=logp_total
            return logp_total 
        return self.evaluation_cache[key]

    


    def perform(self,node,inputs_storage,outputs_storage):
        (sigma_additive, sigma_multiplicative, mu, log_msqrtR) = inputs_storage[0]  # inputs_storage[0] is first (only) parameter, which should be a 4-element vector
        
        logp = self.evaluate_logp_total_from_cache(sigma_additive,sigma_multiplicative,mu,log_msqrtR)
        outputs_storage[0][0]=np.array(logp)
        pass



    def grad(self,inputs,output_grads):
        (sa_sm_mu_lm,) = inputs   # sigma_additive, etc 
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
        
        return [ self.numerical_grad_op(sa_sm_mu_lm)*output_grads[0] ]
    
    def numerical_grad(self,sa_sm_mu_lm):
        (sigma_additive, sigma_multiplicative, mu, log_msqrtR) = sa_sm_mu_lm

        if sigma_additive < 1e-4 or sigma_multiplicative < 1e-4:
            # ridiculously small and outside reasonable domain; treat gradient as zero
            grad=np.zeros(4,dtype='d')
            self.non_convergences +=4 
            self.num_derivatives +=4
            return grad


        origfcn = lambda sa_sm_mu_lm : self.evaluate_logp_total_from_cache(*sa_sm_mu_lm)

        (grad,non_convergences) = gradient(sa_sm_mu_lm, origfcn,initialrelstep=1e-2,rel_precision=1e-2)

        self.non_convergences += non_convergences
        self.num_derivatives += sa_sm_mu_lm.shape[0]
        
        return grad

    
    pass

def CreateMixedNoiseVibroEstparam(name,sa_sm_mu_lm,
                                  scaled_observed,
                                  scaled_prediction_function,
                                  inhibit_accel_pid = None):

    MixedNoiseOp=mixednoise_vibro_estparam_op(observed=scaled_observed,prediction_function=scaled_prediction_function,inhibit_accel_pid=inhibit_accel_pid) 

    def MixedNoiseLogP(sa_sm_mu_lm):
        # captures "MixedNoiseOp"
        return Print('MixedNoiseLogP')(MixedNoiseOp(sa_sm_mu_lm))
    
    # uncomment these next three lines to enable grad verification
    #import theano.tests.unittest_tools
    #theano.config.compute_test_value = "off"
    #theano.tests.unittest_tools.verify_grad(MixedNoiseOp,[np.array((100.0,1.0,0.5,4.0),dtype='d')],abs_tol=1e-3,rel_tol=1e-3,eps=1e-4)

    return (MixedNoiseOp,pm.DensityDist(name,
                                        MixedNoiseLogP,
                                        observed={ "sa_sm_mu_lm": sa_sm_mu_lm }))

    
