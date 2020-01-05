import sys
import os
import os.path
import glob
import collections
import re
import numpy as np
import theano
import theano.tensor as tt
from theano import gof
#from theano import pp

# check_perspecimen_gradient is optional because it requires
# substantial extra software to be installed. We have already
# verified it to be correct.

check_perspecimen_gradient = False
if check_perspecimen_gradient:
    import theano.tests.unittest_tools
    pass

import pymc3 as pm
import pandas as pd
from theano.compile.ops import as_op

from crackheat_surrogate.load_surrogate import nonnegative_denormalized_surrogate



class predict_crackheating_op(gof.Op):
    # Custom theano operation representing
    # the predict_crackheating() function
    
    # Properties attribute
    __props__ = ()
    itypes = None
    otypes = None
    estparam_obj = None

    def perform(self,node,inputs_storage,outputs_storage):
        out = self.estparam_obj.predict_crackheating(*inputs_storage)

        outputs_storage[0][0] = out
        pass

    def grad(self,inputs,output_grads):
        mu=inputs[0]
        msqrtR=inputs[1]

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

        # And if xj can be x0 (mu) or x1 (msqrtR)
        # Then grad_xj G(x) = (sum_i dC/df_i * df_i/dx0, sum_i dC/df_i * df_i/dx1), 
        
        # 
        # We are supposed to return the tensor product dC/df_i * df_i/dxj where
        #          * dC/df_i is output_gradients[0],
        #          * df_i/dx0 is predict_crackheating_grad_mu_op, and
        #          * df_i/dx1 is predict_crackheating_grad_msqrtR_op.
        # Since f is a vectors, dC_df is also a vector
        #    and returning the tensor product means summing over the elements i.
        # From the Theano documentation "The grad method must return a list containing
        #                                one Variable for each input. Each returned Variable
        #                                represents the gradient with respect to that input
        #                                computed based on the symbolic gradients with
        #                                respect to each output."
        # So we return a list indexed over input j
        return [ (self.predict_crackheating_grad_mu_op(*inputs)*output_grads[0]).sum(), (self.predict_crackheating_grad_msqrtR_op(*inputs)*output_grads[0]).sum() ]   
    
    def infer_shape(self,node,input_shapes):
        return [ (self.estparam_obj.crackheat_table.shape[0],) ]

    

    def __init__(self,estparam_obj):
        self.itypes = [tt.dscalar,tt.dscalar]
        self.otypes = [tt.dvector]
        self.estparam_obj = estparam_obj
        
        self.predict_crackheating_grad_mu_op = as_op(itypes=[tt.dscalar,tt.dscalar], otypes = [tt.dmatrix])(self.estparam_obj.predict_crackheating_grad_mu)
        self.predict_crackheating_grad_msqrtR_op = as_op(itypes=[tt.dscalar,tt.dscalar], otypes = [tt.dmatrix])(self.estparam_obj.predict_crackheating_grad_msqrtR)

        pass
    pass




class predict_crackheating_per_specimen_op(gof.Op):
    # Custom theano operation representing
    # the predict_crackheating_per_specimen() function
    
    # Properties attribute
    __props__ = ()
    itypes = None
    otypes = None
    estparam_obj = None

    def perform(self,node,inputs_storage,outputs_storage):
        out = self.estparam_obj.predict_crackheating_per_specimen(*inputs_storage)

        outputs_storage[0][0] = out
        pass

    def grad(self,inputs,output_grads):
        mu=inputs[0]  # mu is now a vector
        msqrtR=inputs[1] # msqrtR is now a vector


        # output_grads has a single element output_grads[0] corresponding to our single vector output
        #
        # ... In evaluating grad_x G(x),
        # where G(x) is representable as C(f(x)) and f(x) is predict_crackheating_op.
        #
        # Then grad_x G(x) = dC/df * df/dx
        # If f is a vector 
        # Then grad_x G(x) = sum_i dC/df_i * df_i/dx
        
        # ... x is now TWO vectors! 
        # Then grad_xjk G(x) = sum_i dC/df_i * df_i/dxjk

        # And if xjk can be x0k (mu) or x1k (msqrtR)
        # Then grad_xjk G(x) = (sum_i dC/df_i * df_i/dx0k, sum_i dC/df_i * df_i/dx1k), 
        
        # 
        # We are supposed to return the tensor product dC/df_i * df_i/dxjk where
        #          * dC/df_i is output_gradients[0],
        #          * df_i/dx0k is predict_crackheating_grad_mu_op, and
        #          * df_i/dx1k is predict_crackheating_grad_msqrtR_op.
        # Since f is a vectors, dC_df is also a vector
        #    and returning the tensor product means summing over the elements i. 
        # From the Theano documentation "The grad method must return a list containing
        #                                one Variable for each input. Each returned Variable
        #                                represents the gradient with respect to that input
        #                                computed based on the symbolic gradients with
        #                                respect to each output."
        # So we return a list indexed over input j.

        return [ (self.predict_crackheating_per_specimen_grad_mu_op(*inputs).T*output_grads[0]).sum(1), (self.predict_crackheating_per_specimen_grad_msqrtR_op(*inputs).T*output_grads[0]).sum(1) ]  
    
    def infer_shape(self,node,input_shapes):
        return [ (self.estparam_obj.crackheat_table.shape[0],) ] #self.estparam_obj.M) ]

    

    def __init__(self,estparam_obj):
        self.itypes = [tt.dvector,tt.dvector]
        self.otypes = [tt.dvector]
        self.estparam_obj = estparam_obj
        
        self.predict_crackheating_per_specimen_grad_mu_op = as_op(itypes=[tt.dvector,tt.dvector], otypes = [tt.dmatrix])(self.estparam_obj.predict_crackheating_per_specimen_grad_mu)
        self.predict_crackheating_per_specimen_grad_msqrtR_op = as_op(itypes=[tt.dvector,tt.dvector], otypes = [tt.dmatrix])(self.estparam_obj.predict_crackheating_per_specimen_grad_msqrtR)

        pass
    pass




class estparam(object):
    """Perform parameter estimation for vibrothermography crack heating
    based on crack surrogates and heating data. """
    
    # member variables
    # inputs:
    crack_specimens=None  # Array of specimen names (strings)
    crackheatfiles=None  # Array of crack heat CSV file names (strings)
    surrogatefiles=None  # Array of surrogate JSON file names (strings)
    accel_trisolve_devs=None # Optional GPU Acceleration parameters

    # results of load_data() method
    crack_surrogates=None # List of surrogate objects corresponding to each surrogate file (list of nonnegative_denormalized_surrogate objects)
    crackheatfile_dataframes=None # List of Pandas dataframes corresponding to each crack heat file
    crackheat_table = None # Unified Pandas dataframe from all crackheat files; lines with NaN's are omitted

    # parameter_estimation variables
    model=None # pymc3 model
    mu=None
    dummy_observed_variable=None
    msqrtR=None
    bending_stress=None
    dynamic_stress=None
    cracknum=None
    predicted_crackheating=None
    y_like=None  # y_likelihood function; was just "crackheating" 

    M=None # number of unique specimens
    N=None # number of data rows
    
    step=None
    trace=None

    # estimated values
    mu_estimate=None
    msqrtR_estimate=None

    # Predicted and actual heatings based on estimates
    predicted=None
    actual=None

    # instance of predict_crackheating_op class (above)
    predict_crackheating_op_instance=None
    
    def __init__(self,**kwargs):
        for argname in kwargs:
            if hasattr(self,argname):
                setattr(self,argname,kwargs[argname])
                pass
            else:
                raise ValueError("Unknown attribute: %s" % (argname))
            pass
        pass

    @classmethod
    def fromfilelists(cls,crack_specimens,crackheatfiles,surrogatefiles,accel_trisolve_devs=None):
        """Load crack data from parallel lists
        of specimens, crack heating file names, and surrogate .json file names"""
        return cls(crack_specimens=crack_specimens,
                   crackheatfiles=crackheatfiles,
                   surrogatefiles=surrogatefiles,
                   accel_trisolve_devs=accel_trisolve_devs)

    def load_data(self,filter_outside_closure_domain=True):
        """Given the lists of crack_specimens, crackheatfiles, surrogatefiles,
        loaded into appropriate crack members, load the data from the files
        into into self.crack_surrogates, and self.crackheatfile_dataframes. 
        The crackheatfile_dataframes are joined into a single table, 
        self.crackheat_table. An additional column specimen_nums is 
        added to self.crackheat_table identifying the specimen index. 

        Rows in the crackheat_table with NaN heating are removed. 

        If the optional parameter filter_outside_closure_domain is set
        (default True), then [rows in the crackheat table with bending
        stress lower than the closure_lowest_avg_load_used variable
        stored in the surrogate] are removed. 
        """

        self.crack_surrogates = [ nonnegative_denormalized_surrogate.fromjsonfile(filename) for filename in self.surrogatefiles ]  

        self.crackheatfile_dataframes = [ pd.read_csv(crackheat_file) for crackheat_file in self.crackheatfiles ]

        self.crackheat_table = pd.concat(self.crackheatfile_dataframes,ignore_index=True) # NOTE: Pandas warning about sorting along non-concatenation axis should go away once all data has timestamps


        assert(np.all(self.crackheat_table["DynamicNormalStressAmpl (Pa)"] > 100*self.crackheat_table["DynamicShearStressAmpl (Pa)"])) # Assumption that shear stress is negligible compared to normal stress for all of this data

        # Add specimen numbers to crackheat table
        # If this next line is slow, can easily be accelerated with a dictionary!
        specimen_nums = np.array([ self.crack_specimens.index(specimen) for specimen in self.crackheat_table["Specimen"].values ],dtype=np.uint32)
        self.crackheat_table["specimen_nums"]=specimen_nums

        # Look up closure_lowest_avg_load_used from values loaded into surrogate
        closure_lowest_avg_load_used = np.array([ self.crack_surrogates[specimen_num].closure_lowest_avg_load_used for specimen_num in self.crackheat_table["specimen_nums"].values ])
        self.crackheat_table["closure_lowest_avg_load_used"]=closure_lowest_avg_load_used
        
        # Drop rows in crackheat_table with NaN ThermalPower
        NaNrownums = np.where(np.isnan(self.crackheat_table["ThermalPower (W)"].values))[0]
    
        self.crackheat_table.drop(NaNrownums,axis=0,inplace=True)
        self.crackheat_table.reset_index(drop=True,inplace=True)

        if filter_outside_closure_domain:
            # Drop rows in crackheat_table with bending stress less than closure_lowest_avg_load_used
            outside_closure_domain_rownums = np.where((~np.isnan(self.crackheat_table["closure_lowest_avg_load_used"].values)) & (self.crackheat_table["closure_lowest_avg_load_used"].values > self.crackheat_table["BendingStress (Pa)"].values))[0]
            self.crackheat_table.drop(outside_closure_domain_rownums,axis=0,inplace=True)
            self.crackheat_table.reset_index(drop=True,inplace=True)
                                                       
            pass


        specimen_nums_filtered = self.crackheat_table["specimen_nums"]

        
        # number of unique specimens
        self.M = np.unique(specimen_nums_filtered).shape[0]

        # total number of observations 
        self.N = specimen_nums_filtered.shape[0]


        
        pass
    

    def predict_crackheating_grad_mu(self,mu,msqrtR):
        """Predict derivative of crackheating with respect to mu for each row in self.crackheat_table,
        given hypothesized values for mu and msqrtR"""
        
        retval = np.zeros(self.crackheat_table.shape[0],dtype='d')
        
        # Could parallelize this loop!
        for index in range(self.crackheat_table.shape[0]):
            #datagrid=np.array(((mu,self.crackheat_table["BendingStress (Pa)"].values[index],self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values[index],msqrtR),),dtype='d')

            datagrid = pd.DataFrame(columns=["mu","bendingstress","dynamicstress","msqrtR"])
            datagrid = datagrid.append({"mu": mu,
                                        "bendingstress": self.crackheat_table["BendingStress (Pa)"].values[index],
                                        "dynamicstress": self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values[index],
                                        "msqrtR": msqrtR},ignore_index=True)
            
            retval[index]=self.crack_surrogates[self.crackheat_table["specimen_nums"].values[index]].evaluate_derivative(datagrid,0,accel_trisolve_devs=self.accel_trisolve_devs)[0]
            pass
        return retval


    def predict_crackheating_grad_msqrtR(self,mu,msqrtR):
        """Predict derivative of crackheating with respect to mu for each row in self.crackheat_table,
        given hypothesized values for mu and msqrtR"""
        
        retval = np.zeros(self.crackheat_table.shape[0],dtype='d')
        
        # Could parallelize this loop!
        for index in range(self.crackheat_table.shape[0]):
            #datagrid=np.array(((mu,self.crackheat_table["BendingStress (Pa)"].values[index],self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values[index],msqrtR),),dtype='d')

            datagrid = pd.DataFrame(columns=["mu","bendingstress","dynamicstress","msqrtR"])
            datagrid = datagrid.append({"mu": mu,
                                        "bendingstress": self.crackheat_table["BendingStress (Pa)"].values[index],
                                        "dynamicstress": self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values[index],
                                        "msqrtR": msqrtR},ignore_index=True)
            
            retval[index]=self.crack_surrogates[self.crackheat_table["specimen_nums"].values[index]].evaluate_derivative(datagrid,3,accel_trisolve_devs=self.accel_trisolve_devs)[0]
            pass
        return retval
    
    
    
    def predict_crackheating(self,mu,msqrtR):
        """Predict crackheating for each row in self.crackheat_table,
        given hypothesized single values for mu and msqrtR across the entire dataset"""
        
        retval = np.zeros(self.crackheat_table.shape[0],dtype='d')
        
        # Could parallelize this loop!
        for index in range(self.crackheat_table.shape[0]):
            #datagrid=np.array(((mu,self.crackheat_table["BendingStress (Pa)"].values[index],self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values[index],msqrtR),),dtype='d')
            datagrid = pd.DataFrame(columns=["mu","bendingstress","dynamicstress","msqrtR"])
            datagrid = datagrid.append({"mu": mu,
                                        "bendingstress": self.crackheat_table["BendingStress (Pa)"].values[index],
                                        "dynamicstress": self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values[index],
                                        "msqrtR": msqrtR},ignore_index=True)
            retval[index]=self.crack_surrogates[self.crackheat_table["specimen_nums"].values[index]].evaluate(datagrid,meanonly=True,accel_trisolve_devs=self.accel_trisolve_devs)["mean"][0]
            pass
        return retval


        
    def predict_crackheating_per_specimen(self,mu,msqrtR):
        """Predict crackheating for each row in self.crackheat_table,
        given hypothesized values for mu and msqrtR per specimen"""
        
        retval = np.zeros(self.crackheat_table.shape[0],dtype='d')
        
        # Could parallelize this loop!
        for index in range(self.crackheat_table.shape[0]):  # ... going from 0...N-1 -- iterating over all the rows in the data
            specimen_num=self.crackheat_table["specimen_nums"].values[index]
            
            #datagrid=np.array(((mu[specimen_num],self.crackheat_table["BendingStress (Pa)"].values[index],self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values[index],msqrtR[specimen_num]),),dtype='d')
            
            datagrid = pd.DataFrame(columns=["mu","bendingstress","dynamicstress","msqrtR"])
            datagrid = datagrid.append({"mu": mu[specimen_num],
                                        "bendingstress": self.crackheat_table["BendingStress (Pa)"].values[index],
                                        "dynamicstress": self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values[index],
                                        "msqrtR": msqrtR[specimen_num]},ignore_index=True)
            
            retval[index]=self.crack_surrogates[specimen_num].evaluate(datagrid,meanonly=True,accel_trisolve_devs=self.accel_trisolve_devs)["mean"][0]
            pass
        return retval


    def predict_crackheating_per_specimen_grad_mu(self,mu,msqrtR):
        """Predict derivative of crackheating with respect to mu vector for each row in self.crackheat_table,
        given hypothesized values for mu and msqrtR"""
        
        retval = np.zeros((self.crackheat_table.shape[0],self.M),dtype='d')
        
        # Could parallelize this loop!
        for index in range(self.crackheat_table.shape[0]):
            specimen_num=self.crackheat_table["specimen_nums"].values[index]
            #datagrid=np.array(((mu[specimen_num],self.crackheat_table["BendingStress (Pa)"].values[index],self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values[index],msqrtR[specimen_num]),),dtype='d')
            
            datagrid = pd.DataFrame(columns=["mu","bendingstress","dynamicstress","msqrtR"])
            datagrid = datagrid.append({"mu": mu[specimen_num],
                                        "bendingstress": self.crackheat_table["BendingStress (Pa)"].values[index],
                                        "dynamicstress": self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values[index],
                                        "msqrtR": msqrtR[specimen_num]},ignore_index=True)
            
            retval[index,specimen_num]=self.crack_surrogates[specimen_num].evaluate_derivative(datagrid,0,accel_trisolve_devs=self.accel_trisolve_devs)[0]
            pass
        return retval



    def predict_crackheating_per_specimen_grad_msqrtR(self,mu,msqrtR):
        """Predict derivative of crackheating with respect to mu vector for each row in self.crackheat_table,
        given hypothesized per-specimen vectors for mu and msqrtR"""
        
        retval = np.zeros((self.crackheat_table.shape[0],self.M),dtype='d')
        
        # Could parallelize this loop!
        for index in range(self.crackheat_table.shape[0]):
            specimen_num=self.crackheat_table["specimen_nums"].values[index]

            #datagrid=np.array(((mu[specimen_num],self.crackheat_table["BendingStress (Pa)"].values[index],self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values[index],msqrtR[specimen_num]),),dtype='d')
            
            datagrid = pd.DataFrame(columns=["mu","bendingstress","dynamicstress","msqrtR"])
            datagrid = datagrid.append({"mu": mu[specimen_num],
                                        "bendingstress": self.crackheat_table["BendingStress (Pa)"].values[index],
                                        "dynamicstress": self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values[index],
                                        "msqrtR": msqrtR[specimen_num]},ignore_index=True)
            
            retval[index,specimen_num]=self.crack_surrogates[specimen_num].evaluate_derivative(datagrid,3,accel_trisolve_devs=self.accel_trisolve_devs)[0]
            pass
        return retval
    

    
    
    def posterior_estimation(self,steps_per_chain,num_chains,cores=None,tune=500):
        """Build and execute PyMC3 Model to obtain self.trace which 
        holds the chain samples"""

        self.model = pm.Model()
    
        with self.model:
            #mu = pm.Uniform('mu',lower=0.01,upper=3.0)
            #msqrtR = pm.Uniform('msqrtR',lower=500000,upper=50e6)
            
            self.mu = pm.Lognormal('mu',mu=0.0,sigma=1.0)
            self.msqrtR = pm.Lognormal('msqrtR',mu=np.log(20e6),sigma=1.0)

            # Add in a dummy observed variable that does not interact with the model
            # This is in place because arviz (used for traceplot(), others)
            # does a conversion called convert_to_dataset() on the PyMC3 
            # samples. This conversion calls:
            #   io_pymc3.py/to_inference_data() which calls 
            #   io_pymc3.py/sample_stats_to_xarray() which calls 
            #   io_pymc3.py/_extract_log_likelihood()
            # In our context, extracting the log_likelihood is very slow
            # because it seems to require many calls to the 
            # predict_crackheating() function. Because these calls are in 
            # the main process, not a worker process, they are not delegated
            # to the GPU (GPU is disabled in main process, because if it 
            # fork()s after GPU access subprocess GPU access will fail) 
            # and evaluating the log_likelihood can take hours 
            # even for a data set that was sampled in 5-10 minutes. 
            #
            # Within _extract_log_likelihood() there is a test:
            #         if len(model.observed_RVs) != 1:
            # That disables likelihood evaluation if there is more 
            # than one observed random variable. 
            #
            # To trigger that test and disable likelihood evaluation, 
            # we add this additional observed random variable that 
            # is otherwise unused in the model
            #
            # Other workarounds might be possible; for example it seems
            # likely that the evaluations would be for exactly 
            # the same parameter values used in the sampling. 
            # If so, somehow transferring a cache of predicted heating
            # values to the main process would probably address the
            # slowdown as well. 
            self.dummy_observed_variable = pm.Normal('dummy_observed_variable',mu=0,sigma=1,observed=0.0)
            
            #self.bending_stress = pm.Normal('bending_stress',mu=50e6, sigma=10e6, observed=self.crackheat_table["BendingStress (Pa)"].values)
            #self.dynamic_stress = pm.Normal('dynamic_stress',mu=20e6, sigma=5e6,observed=self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values)
            #self.cracknum = pm.DiscreteUniform('cracknum',lower=0,upper=len(self.crack_specimens)-1,observed=self.crackheat_table["specimen_nums"].values)
            #self.predict_crackheating_op_instance = as_op(itypes=[tt.dscalar,tt.dscalar], otypes = [tt.dvector])(self.predict_crackheating)

            self.predict_crackheating_op_instance = predict_crackheating_op(self)

            # Verify that our op correctly calculates the gradient
            #theano.tests.unittest_tools.verify_grad(self.predict_crackheating_op_instance,[ np.array(0.3), np.array(5e6)]) # mu=0.3, msqrtR=5e6

            mu_testval = tt.dscalar('mu_testval')
            mu_testval.tag.test_value = 0.3 # pymc3 turns on theano's config.compute_test_value switch, so we have to provide a value
            
            msqrtR_testval = tt.dscalar('msqrtR_testval')
            msqrtR_testval.tag.test_value = 5e6 # pymc3 turns on theano's config.compute_test_value switch, so we have to provide a value

            test_function = theano.function([mu_testval,msqrtR_testval],self.predict_crackheating_op_instance(mu_testval,msqrtR_testval))
            jac_mu = tt.jacobian(self.predict_crackheating_op_instance(mu_testval,msqrtR_testval),mu_testval)
            jac_mu_analytic = jac_mu.eval({ mu_testval: 0.3, msqrtR_testval: 5e6})
            jac_mu_numeric = (test_function(0.301,5e6)-test_function(0.300,5e6))/.001
            assert(np.linalg.norm(jac_mu_analytic-jac_mu_numeric)/np.linalg.norm(jac_mu_analytic) < .05)

            jac_msqrtR = tt.jacobian(self.predict_crackheating_op_instance(mu_testval,msqrtR_testval),msqrtR_testval)
            jac_msqrtR_analytic = jac_msqrtR.eval({ mu_testval: 0.3, msqrtR_testval: 5e6})
            jac_msqrtR_numeric = (test_function(0.300,5.01e6)-test_function(0.300,5.00e6))/.01e6
            assert(np.linalg.norm(jac_msqrtR_analytic-jac_msqrtR_numeric)/np.linalg.norm(jac_msqrtR_analytic) < .05)

            
            # Create pymc3 predicted_crackheating expression
            self.predicted_crackheating = self.predict_crackheating_op_instance(self.mu,self.msqrtR)
            #self.predicted_crackheating = pm.Deterministic('predicted_crackheating',predict_crackheating_op(self.mu,self.msqrtR))
            self.crackheating = pm.Normal('crackheating', mu=self.predicted_crackheating, sigma=1e-9, observed=self.crackheat_table["ThermalPower (W)"].values/self.crackheat_table["ExcFreq (Hz)"].values) # ,shape=specimen_nums.shape[0])
        
            #self.step = pm.Metropolis()
            self.step=pm.NUTS()
            self.trace = pm.sample(steps_per_chain, step=self.step,chains=num_chains, cores=cores,tune=tune) # discard_tuned_samples=False,tune=0)
            pass
        pass



    
    def plot_and_estimate(self,mu_zone=(0.05,0.5),msqrtR_zone=(.36e8,.43e8),marginal_bins=50,joint_bins=(230,200)):
        """ Create diagnostic histograms. Also return coordinates of 
        joint histogram peak as estimates of mu and msqrtR"""

        from matplotlib import pyplot as pl
        import cycler

        #traceplots=pl.figure()
        traceplot_axes = pm.traceplot(self.trace)
        traceplots = traceplot_axes[0,0].figure

        mu_vals=self.trace.get_values("mu")
        msqrtR_vals = self.trace.get_values("msqrtR")
        
        mu_hist = pl.figure()
        pl.clf()
        pl.hist(mu_vals,bins=marginal_bins)
        pl.xlabel('mu')
        pl.grid()
        
        
        msqrtR_hist = pl.figure()
        pl.clf()
        pl.hist(msqrtR_vals,bins=marginal_bins)
        pl.xlabel('m*sqrtR')
        pl.grid()
        
    
        joint_hist = pl.figure()
        pl.clf()
        (hist,hist_mu_edges,hist_msqrtR_edges,hist_image)=pl.hist2d(mu_vals,msqrtR_vals,range=(mu_zone,msqrtR_zone),bins=joint_bins)
        pl.grid()
        pl.colorbar()
        pl.xlabel('mu')
        pl.ylabel('m*sqrt(R) (sqrt(m)/m^2)')
        
        histpeakpos = np.unravel_index(np.argmax(hist,axis=None),hist.shape)
        self.mu_estimate = (hist_mu_edges[histpeakpos[0]]+hist_mu_edges[histpeakpos[0]+1])/2.0
        self.msqrtR_estimate = (hist_msqrtR_edges[histpeakpos[1]]+hist_msqrtR_edges[histpeakpos[1]+1])/2.0
    
        # Compare
        self.predicted = self.predict_crackheating(self.mu_estimate,self.msqrtR_estimate)*self.crackheat_table["ExcFreq (Hz)"].values
        
        # add to crackheat_table
        
        self.crackheat_table["predicted"]=self.predicted

        ## with:
        #self.actual = self.crackheat_table["ThermalPower (W)"].values

        # Group predicted and actual heating by specimen
        specimen_grouping = self.crackheat_table.groupby("Specimen")

        specimen_groups = [ specimen_grouping.get_group(specimen) for specimen in specimen_grouping.groups ]
        predicted_by_specimen = [ specimen_group["predicted"].values for specimen_group in specimen_groups ]
        actual_by_specimen = [ specimen_group["ThermalPower (W)"].values for specimen_group in specimen_groups ]
        specimen_group_specimens = list(specimen_grouping.groups) 

        markerstyle_cycler=cycler.cycler(marker=['o','v','^','<','>','s','p','+','x','D'])()
        

        prediction_plot = pl.figure()
        pl.clf()
        #pl.plot(self.predicted,self.actual,'x',
        #        (0,np.max(self.predicted)),(0,np.max(self.predicted)),'-')
        [ pl.plot(predicted_by_specimen[idx]*1e3,actual_by_specimen[idx]*1e3,linestyle='',**next(markerstyle_cycler)) for idx in range(len(specimen_group_specimens)) ] 
        pl.plot((0,np.max(self.predicted)*1e3),(0,np.max(self.predicted)*1e3),'-')
        pl.legend(specimen_group_specimens,loc='best')
        pl.xlabel('Predicted heating from model (mW)')
        pl.ylabel('Actual heating from experiment (mW)')
        pl.title('mu_estimate=%g; msqrtR_estimate=%g' % (self.mu_estimate,self.msqrtR_estimate))
        pl.grid()
        
        return (self.mu_estimate,self.msqrtR_estimate,traceplots,mu_hist,msqrtR_hist,joint_hist,prediction_plot)



    def posterior_estimation_partial_pooling(self,steps_per_chain,num_chains,cores=None,tune=500):
        """Build and execute PyMC3 Model to obtain self.trace which 
        holds the chain samples"""

        ThermalPowerPerHz = self.crackheat_table["ThermalPower (W)"].values/self.crackheat_table["ExcFreq (Hz)"].values

        
        self.model = pm.Model()
        with self.model:

            #
            # prior distribution for the mean vector of the BVN
            #
            theta = pm.MvNormal('theta', mu=np.array([-1.2, 17.7]), 
                                cov = np.array([[4.0, 0.0], [0.0, 16.0]]), shape = (1, 2))
            
            #
            # prior distribution for covariance matrix in Cholesky form
            #
            sd_dist = pm.HalfCauchy.dist(beta=2.5, shape=3)
            packed_L = pm.LKJCholeskyCov('packed_L', n=2,
                                         eta=1.2, sd_dist=sd_dist)
            L = pm.expand_packed_triangular(2, packed_L, lower=True)
            # If we needed it could get the covariance matrix from
            #      Sigma = pm.Deterministic('Sigma', L.dot(L.T))
            #
            # prior distribution for the error standard deviation
            #
            sigmaError = pm.HalfFlat("sigmaError")
            pass
        
        with self.model:
            # crack random effects (log intercept and log slope in pseudo heating model)
            # 
            # for reparameterization start with standardized normal (to keep sampler happy)
            LambdaStd = pm.MvNormal('LambdaStd', mu=np.zeros(2), cov=np.identity(2),
                                    shape= (self.M, 2))
            #
            #  add in the mean vector and multiply by the Cholesky matrix
            #  The 'Deterministic' is a sum of the mean (theta) with the mvnormal LambdaStd (standardized) multiplied by L, the lower diagonal Cholesky factor representing the covariance structure in the bivariate normal. 
            
            # Lambda represents the logarithms of the parameters  (mu and msqrtR, identified by column)
            # and which specimen, identified by row. 
            Lambda = pm.Deterministic('Lambda',  theta + tt.dot(L, LambdaStd.T).T)

            # logmu is M rows 
            logmu = Lambda[:,0]  
            logmsqrtR = Lambda[:,1]

            # self.mu is M rows
            self.mu=np.exp(logmu)
            self.msqrtR = np.exp(logmsqrtR)

            #
            # specify the predicted heating from the parameters Lambda (now extracted into mu and msqrtR)... Should be 1 column by N rows. 
            #
            # Create pymc3 predicted_crackheating expression
            self.predict_crackheating_per_specimen_op_instance = predict_crackheating_per_specimen_op(self)
            self.predicted_crackheating = self.predict_crackheating_per_specimen_op_instance(self.mu,self.msqrtR)


            #
            # specify the likelihood
            #
            self.y_like = pm.Normal('y_like',mu=self.predicted_crackheating,sigma=sigmaError*np.ones(self.N),observed=ThermalPowerPerHz)  # may need to broadcast sigmaError over N rows of mu
            
            
            # Add in a dummy observed variable that does not interact with the model
            # This is in place because arviz (used for traceplot(), others)
            # does a conversion called convert_to_dataset() on the PyMC3 
            # samples. This conversion calls:
            #   io_pymc3.py/to_inference_data() which calls 
            #   io_pymc3.py/sample_stats_to_xarray() which calls 
            #   io_pymc3.py/_extract_log_likelihood()
            # In our context, extracting the log_likelihood is very slow
            # because it seems to require many calls to the 
            # predict_crackheating() function. Because these calls are in 
            # the main process, not a worker process, they are not delegated
            # to the GPU (GPU is disabled in main process, because if it 
            # fork()s after GPU access subprocess GPU access will fail) 
            # and evaluating the log_likelihood can take hours 
            # even for a data set that was sampled in 5-10 minutes. 
            #
            # Within _extract_log_likelihood() there is a test:
            #         if len(model.observed_RVs) != 1:
            # That disables likelihood evaluation if there is more 
            # than one observed random variable. 
            #
            # To trigger that test and disable likelihood evaluation, 
            # we add this additional observed random variable that 
            # is otherwise unused in the model
            #
            # Other workarounds might be possible; for example it seems
            # likely that the evaluations would be for exactly 
            # the same parameter values used in the sampling. 
            # If so, somehow transferring a cache of predicted heating
            # values to the main process would probably address the
            # slowdown as well. 
            self.dummy_observed_variable = pm.Normal('dummy_observed_variable',mu=0,sigma=1,observed=0.0)
            

            if check_perspecimen_gradient:
                # verify_grad() only works when theano.config.compute_test_value is "off"
                try:
                    orig_ctv = theano.config.compute_test_value
                    theano.config.compute_test_value = "off"
                    # Verify that our op correctly calculates the gradient
                    print("grad_mu_values = %s" % (str(self.predict_crackheating_per_specimen_grad_mu(np.array([0.3]*self.M), np.array([5e6]*self.M)))))
                    print("grad_msqrtR_values = %s" % (str(self.predict_crackheating_per_specimen_grad_msqrtR(np.array([0.3]*self.M), np.array([5e6]*self.M)))))

                    # Test gradient with respect to mu
                    theano.tests.unittest_tools.verify_grad(lambda mu_val: self.predict_crackheating_per_specimen_op_instance(mu_val, theano.shared(np.array([5e6]*self.M))) ,[ np.array([0.3]*self.M),],abs_tol=1e-12,rel_tol=1e-5) # mu=0.3, msqrtR=5e6
                    # Test gradient with respect to msqrtR
                    theano.tests.unittest_tools.verify_grad(lambda msqrtR_val: self.predict_crackheating_per_specimen_op_instance(theano.shared(np.array([0.3]*self.M)),msqrtR_val) ,[ np.array([5e6]*self.M)],abs_tol=1e-20,rel_tol=1e-8,eps=1.0) # mu=0.3, msqrtR=5e6  NOTE: rel_tol is very tight here because Theano gradient.py/abs_rel_err() lower bounds the relative divisor to 1.e-8 and if we are not tight, we don't actually diagnose errors. 

                    print("\n\n\nVerify_grad() completed!!!\n\n\n")
                    pass
                finally:
                    theano.config.compute_test_value = orig_ctv
                    pass
                pass
            
            pass

        # set up for sampling
        
        with self.model:
        
            #self.step = pm.Metropolis()
            self.step=pm.NUTS(target_accept=0.80)
            self.trace = pm.sample(steps_per_chain, step=self.step,chains=num_chains, cores=cores,tune=tune) # discard_tuned_samples=False,tune=0)
            pass
        pass



    def plot_and_estimate_partial_pooling(self,mu_zone=(0.05,1.0),msqrtR_zone=(29.5e6,48.6e6),marginal_bins=50,joint_bins=(230,200)):
        """ Create diagnostic histograms. Also return coordinates of 
        joint histogram peak as estimates of mu and msqrtR"""

        from matplotlib import pyplot as pl
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        import cycler

        #traceplots=pl.figure()
        #traceplot_axes = pm.traceplot(self.trace,figsize=(12,self.M*4+12))
        traceplot_axes = pm.traceplot(self.trace,figsize=(3,9.2),textsize=2.0)
        traceplots = traceplot_axes[0,0].figure
        


        theta_vals = self.trace.get_values("theta")

        # Unpack packed_L: elements L0 L1 L2 ->
        #  [ L0   0  ]
        #  [ L1   L2 ]
        #packed_L = self.trace.get_values("packed_L")

        trace_frame=pd.DataFrame(index=pd.RangeIndex(theta_vals.shape[0]))
        
        for varname in self.trace.varnames:
            vardata = self.trace.get_values(varname)
            it=np.nditer(vardata[0,...],flags=['multi_index'])
            while not it.finished:
                colname = varname
                if it.multi_index != ():
                    colname += str(it.multi_index).replace(", ","__")
                    pass
                colindex = (slice(None),)+it.multi_index
                trace_frame.insert(len(trace_frame.columns),colname,vardata[colindex])
                it.iternext()
            vardims = vardata.shape[1:]
            pass
        #trace_frame.to_csv('/tmp/trace.csv')


        # Bill's traceplots: 
        theta_L_sigmaerror = pl.figure()
        theta_L_sigmaerror_datatitles = [ "theta(0__0)", "packed_L(1,)","theta(0__1)","packed_L(2,)","packed_L(0,)","sigmaError"]
        for subplotnum in range(len(theta_L_sigmaerror_datatitles)):
            datatitle=theta_L_sigmaerror_datatitles[subplotnum]
            pl.subplot(3,2,subplotnum+1)
            pl.plot(trace_frame.index,trace_frame[datatitle])
            pl.xlabel('Sample index')
            pl.ylabel(datatitle)
            pl.grid(True)
            pass

        lambdaplots = []
        lambdasubplotrows = 4
        lambdacnt = 0

        while lambdacnt < self.trace["Lambda"].shape[1]:
            lambdaplot = pl.figure()

            for subplotrownum in range(lambdasubplotrows):
                if lambdacnt >= self.trace["Lambda"].shape[1]:
                    break

                pl.subplot(lambdasubplotrows,2,subplotrownum*2+1)
                pl.plot(trace_frame.index,self.trace["Lambda"][:,lambdacnt,0])
                pl.xlabel('Sample index')
                pl.ylabel('Lambda(%d__0)' % (lambdacnt))
                if subplotrownum==0:
                    pl.title('per-sample ln(mu)')
                    pass
                pl.grid(True)

                pl.subplot(lambdasubplotrows,2,subplotrownum*2+2)
                pl.plot(trace_frame.index,self.trace["Lambda"][:,lambdacnt,1])
                pl.xlabel('Sample index')
                pl.ylabel('Lambda(%d__1)' % (lambdacnt))
                if subplotrownum==0:
                    pl.title('per-sample ln(msqrtR)')
                    pass
                pl.grid(True)
                
                lambdacnt+=1

                pass
            lambdaplots.append(lambdaplot)
            pass
        
        histograms=collections.OrderedDict()

        # First create the histograms for the traces in theta_L_sigmaerror
        for datatitle in theta_L_sigmaerror_datatitles:
            histfig=pl.figure()
            pl.hist(trace_frame[datatitle],bins=18)
            pl.title(datatitle)
            pl.ylabel('Frequency')
            pl.grid(True)
            histograms[datatitle]=histfig
            pass


        # Now create histograms for the lambdas
        for lambdacnt in range(self.trace["Lambda"].shape[1]):
            histfig=pl.figure()
            datatitle = "Lambda[%d,0]" % (lambdacnt)
            pl.hist(self.trace["Lambda"][:,lambdacnt,0],bins=marginal_bins,range=(np.log(mu_zone[0]),np.log(mu_zone[1])))
            pl.title("%s: ln(mu) for crack #%d" % (datatitle,lambdacnt))
            pl.ylabel('Frequency')
            pl.grid(True)

            # draw inset
            ax=pl.gca()
            axins = inset_axes(ax,width="20%",height="20%",loc="upper right")
            axins.hist(self.trace["Lambda"][:,lambdacnt,0],bins=marginal_bins)
            pl.grid(True)
            histograms[datatitle]=histfig


            histfig=pl.figure()
            datatitle = "Lambda[%d,1]" % (lambdacnt)
            pl.hist(self.trace["Lambda"][:,lambdacnt,1],bins=marginal_bins,range=(np.log(msqrtR_zone[0]),np.log(msqrtR_zone[1])))
            pl.title("%s: ln(msqrtR) for crack #%d" % (datatitle,lambdacnt))
            pl.ylabel('Frequency')
            pl.grid(True)

            # draw inset
            ax=pl.gca()
            axins = inset_axes(ax,width="20%",height="20%",loc="upper right")
            axins.hist(self.trace["Lambda"][:,lambdacnt,1],bins=marginal_bins)
            pl.grid(True)

            histograms[datatitle]=histfig
            
            
            pass

        
        
        # Bill: Suggest scatterplot from medians of (mu,sqrtR) of each crack (lambdas). 
        # with ellipse centered at median (theta1) and based on covariances based on its Cholesky decomposition. 
        # illustrates crack-to-crack variability and relationship between parameters
        Lambda0_medians = np.median(self.trace["Lambda"][:,:,0],axis=0)
        Lambda1_medians = np.median(self.trace["Lambda"][:,:,1],axis=0)

        Theta0_median = np.median(self.trace["theta"][:,0,0],axis=0)
        Theta1_median = np.median(self.trace["theta"][:,0,1],axis=0)

        lambda_scatterplot = pl.figure()
        pl.plot(Lambda0_medians,Lambda1_medians,'x')
        pl.plot(Theta0_median,Theta1_median,'o')

        # The lambda distribution is a multivariate normal with covariance matrix capitial sigma = LL'. 
        packed_L_median=np.median(self.trace.get_values("packed_L"),axis=0)
        L=np.array(((packed_L_median[0],0.0),(packed_L_median[1],packed_L_median[2])),dtype='d')
        LLt = np.dot(L,L.T)
        inv_LLt = np.linalg.inv(LLt)
        (L_evals,L_evects) = np.linalg.eig(inv_LLt)
        # Now (L_evects * diag(L_evals) * L_evects.T) = inv_LLt
        # so the exponent of the multivariate normal distribution
        # -(1/2)(x-mean).T * inv_LLt * (x-mean)
        # becomes
        # -(1/2)(x-mean).T * (L_evects * diag(L_evals) * L_evects.T) * (x-mean)
        # Regrouping parentheses
        # -(1/2) ( (x-mean).T * L_evects ) * diag(L_evals) * (L_evects.T * (x-mean) )
        # So the first element of (L_evects.T * (x-mean) ) is squared
        # and multiplied by the first element of L_evals and that is 
        # added to the same thing with the 2nd element... final result
        # multiplied by 1/2. 
        # So setting this equal to a constant... say -1/2 to get 1 standard deviation 
        # says el_1^2*L_eval1/2 + el_2^2 *L_eval2/2 = 1/2
        # or el_1^2*L_eval1 + el_2^2 *L_eval2 = 1
        # So this is the equation of an ellipse where the axis lengths
        # are sqrt(1/L_eval1) and sqrt(1/L_eval2)

        # (1/2) ( (x-mean).T * L_evects ) * diag(L_evals) * (L_evects.T * (x-mean) ) = 1/2 
        #Parametric form of an ellipse: x=a*cos(t); y=b*sin(t)
        # letting u1 and u2 be the basis coordinates for the eigenframe
        # u1=sqrt(1/L_eval1)*cos(t) and u2=sqrt(1/L_eval2)*sin(t)
        # [(x1 - mean1) ; (x2-mean2)] = L_evects * [ u1; u2]
        # [x1 ; x2] = L_evects * [ u1; u2] + [ mean1; mean2] 
        # [x1 ; x2] = L_evects * sqrt(1/L_eval).*[ cos(t); sin(t)]  + [ mean1; mean2] 
        ellipse_param_t = np.linspace(0.0,2*np.pi,180)
        ellipse_coords = np.dot(L_evects,np.sqrt(1.0/L_evals[:,np.newaxis])*np.array((np.cos(ellipse_param_t),np.sin(ellipse_param_t)),dtype='d')) + np.array((Theta0_median,Theta1_median),dtype='d')[:,np.newaxis]
        
        pl.plot(ellipse_coords[0,:],ellipse_coords[1,:],'-')

        pl.legend(('Lambdas','Theta'),loc="best")
        pl.xlabel('Lambda[:,0]')
        pl.ylabel('Lambda[:,1]')
        pl.grid(True)

        mu_msqrtR_scatterplot = pl.figure()
        pl.plot(np.exp(Lambda0_medians),np.exp(Lambda1_medians),'x')
        pl.plot(np.exp(Theta0_median),np.exp(Theta1_median),'o')
        pl.legend(('Individual cracks','Ensemble'),loc="best")
        pl.xlabel('mu')
        pl.ylabel('msqrtR')
        pl.grid(True)

        # Also look at sampler output for isolated draws (divergent transitions?)

        # 2D scatterplot (2D histogram) of draws for theta1 and theta2 to 
        # tell us how certain we are about the means of those parameters. 

        mu_vals = np.exp(theta_vals[:,0,0])

        msqrtR_vals = np.exp(theta_vals[:,0,1])
        
        mu_hist = pl.figure()
        pl.clf()
        pl.hist(mu_vals,bins=marginal_bins,range=mu_zone)
        pl.xlabel('mu')
        pl.ylabel('Frequency')
        pl.grid()
        # draw inset
        ax=pl.gca()
        axins = inset_axes(ax,width="20%",height="20%",loc="upper right")
        axins.hist(mu_vals,bins=marginal_bins)
        pl.grid(True)
        
        
        msqrtR_hist = pl.figure()
        pl.clf()
        pl.hist(msqrtR_vals,bins=marginal_bins,range=msqrtR_zone)
        pl.xlabel('m*sqrtR')
        pl.ylabel('Frequency')
        pl.grid()
        # draw inset
        ax=pl.gca()
        axins = inset_axes(ax,width="20%",height="20%",loc="upper right")
        axins.hist(msqrtR_vals,bins=marginal_bins)
        pl.grid(True)
        
    
        joint_hist = pl.figure()
        pl.clf()
        (hist,hist_mu_edges,hist_msqrtR_edges,hist_image)=pl.hist2d(mu_vals,msqrtR_vals,range=(mu_zone,msqrtR_zone),bins=joint_bins)
        pl.grid()
        pl.colorbar()
        pl.xlabel('mu')
        pl.ylabel('m*sqrt(R) (sqrt(m)/m^2)')
        # draw inset
        ax=pl.gca()
        axins = inset_axes(ax,width="20%",height="20%",loc="upper right")
        axins.hist2d(mu_vals,msqrtR_vals,bins=joint_bins)
        pl.grid(True)
        
        #histpeakpos = np.unravel_index(np.argmax(hist,axis=None),hist.shape)
        #self.mu_estimate = (hist_mu_edges[histpeakpos[0]]+hist_mu_edges[histpeakpos[0]+1])/2.0
        
        #self.msqrtR_estimate = (hist_msqrtR_edges[histpeakpos[1]]+hist_msqrtR_edges[histpeakpos[1]+1])/2.0
    
        self.mu_estimate = np.median(mu_vals)
        self.msqrtR_estimate = np.median(msqrtR_vals)
        
        #theta0_estimate = np.median(theta_vals[:,0,0])
        #theta1_estimate=np.median(theta_vals[:,0,1])



        # Compare
        self.predicted = self.predict_crackheating(self.mu_estimate,self.msqrtR_estimate)*self.crackheat_table["ExcFreq (Hz)"].values
        
        # add to crackheat_table
        
        self.crackheat_table["predicted"]=self.predicted

        # with:
        self.actual = self.crackheat_table["ThermalPower (W)"].values

        # Group predicted and actual heating by specimen
        specimen_grouping = self.crackheat_table.groupby("Specimen")

        specimen_groups = [ specimen_grouping.get_group(specimen) for specimen in specimen_grouping.groups ]
        predicted_by_specimen = [ specimen_group["predicted"].values for specimen_group in specimen_groups ]
        actual_by_specimen = [ specimen_group["ThermalPower (W)"].values for specimen_group in specimen_groups ]
        specimen_group_specimens = list(specimen_grouping.groups) 

        markerstyle_cycler=cycler.cycler(marker=['o','v','^','<','>','s','p','+','x','D'])()
        

        prediction_plot = pl.figure()
        pl.clf()
        #pl.plot(self.predicted,self.actual,'x',
        #        (0,np.max(self.predicted)),(0,np.max(self.predicted)),'-')
        [ pl.plot(predicted_by_specimen[idx]*1e3,actual_by_specimen[idx]*1e3,linestyle='',**next(markerstyle_cycler)) for idx in range(len(specimen_group_specimens)) ] 
        pl.plot((0,np.max(self.predicted)*1e3),(0,np.max(self.predicted)*1e3),'-')
        pl.legend(specimen_group_specimens,loc='best')
        pl.xlabel('Predicted heating from model (mW)')
        pl.ylabel('Actual heating from experiment (mW)')
        pl.title('mu_estimate=%g; msqrtR_estimate=%g' % (self.mu_estimate,self.msqrtR_estimate))
        pl.grid()
        
        return (self.mu_estimate,self.msqrtR_estimate,Theta1_median,packed_L_median,trace_frame,traceplots,theta_L_sigmaerror,lambdaplots,histograms,lambda_scatterplot,mu_msqrtR_scatterplot,mu_hist,msqrtR_hist,joint_hist,prediction_plot)
    
    pass
