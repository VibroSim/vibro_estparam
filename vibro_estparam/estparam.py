import sys
import os
import os.path
import glob
import collections
import re
import scipy
import scipy.special
import scipy.integrate
import numpy as np
import theano
import theano.tensor as tt
from theano.printing import Print
from theano import gof
#from theano import pp

# check_perspecimen_gradient is optional because it requires
# substantial extra software to be installed. We have already
# verified it to be correct.

check_perspecimen_gradient = False
check_gradient = False
if check_perspecimen_gradient or check_gradient:
    import theano.tests.unittest_tools
    pass

import pymc3 as pm
import pandas as pd
from theano.compile.ops import as_op

from crackheat_surrogate2.load_surrogate import load_denorm_surrogates_from_jsonfile
from crackheat_surrogate2.load_surrogate_shear import load_denorm_surrogates_shear_from_jsonfile
from .mixednoise import CreateMixedNoise
from .mixednoise_vibro_estparam import CreateMixedNoiseVibroEstparam

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
        log_msqrtR=inputs[1]

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
        return [ (self.predict_crackheating_grad_mu_op(*inputs)*output_grads[0]).sum(), (self.predict_crackheating_grad_log_msqrtR_op(*inputs)*output_grads[0]).sum() ]   
    
    def infer_shape(self,node,input_shapes):
        return [ (self.estparam_obj.crackheat_table.shape[0],) ]

    

    def __init__(self,estparam_obj):
        self.itypes = [tt.dscalar,tt.dscalar]
        self.otypes = [tt.dvector]
        self.estparam_obj = estparam_obj
        
        self.predict_crackheating_grad_mu_op = as_op(itypes=[tt.dscalar,tt.dscalar], otypes = [tt.dvector])(self.estparam_obj.predict_crackheating_grad_mu)
        self.predict_crackheating_grad_log_msqrtR_op = as_op(itypes=[tt.dscalar,tt.dscalar], otypes = [tt.dvector])(self.estparam_obj.predict_crackheating_grad_log_msqrtR)

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
        log_msqrtR=inputs[1] # log_msqrtR is now a vector


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

        # And if xjk can be x0k (mu) or x1k (log_msqrtR)
        # Then grad_xjk G(x) = (sum_i dC/df_i * df_i/dx0k, sum_i dC/df_i * df_i/dx1k), 
        
        # 
        # We are supposed to return the tensor product dC/df_i * df_i/dxjk where
        #          * dC/df_i is output_gradients[0],
        #          * df_i/dx0k is predict_crackheating_grad_mu_op, and
        #          * df_i/dx1k is predict_crackheating_grad_log_msqrtR_op.
        # Since f is a vectors, dC_df is also a vector
        #    and returning the tensor product means summing over the elements i. 
        # From the Theano documentation "The grad method must return a list containing
        #                                one Variable for each input. Each returned Variable
        #                                represents the gradient with respect to that input
        #                                computed based on the symbolic gradients with
        #                                respect to each output."
        # So we return a list indexed over input j.

        return [ (self.predict_crackheating_per_specimen_grad_mu_op(*inputs).T*output_grads[0]).sum(1), (self.predict_crackheating_per_specimen_grad_log_msqrtR_op(*inputs).T*output_grads[0]).sum(1) ]  
    
    def infer_shape(self,node,input_shapes):
        return [ (self.estparam_obj.crackheat_table.shape[0],) ] #self.estparam_obj.M) ]

    

    def __init__(self,estparam_obj):
        self.itypes = [tt.dvector,tt.dvector]
        self.otypes = [tt.dvector]
        self.estparam_obj = estparam_obj
        
        self.predict_crackheating_per_specimen_grad_mu_op = as_op(itypes=[tt.dvector,tt.dvector], otypes = [tt.dmatrix])(self.estparam_obj.predict_crackheating_per_specimen_grad_mu)
        self.predict_crackheating_per_specimen_grad_log_msqrtR_op = as_op(itypes=[tt.dvector,tt.dvector], otypes = [tt.dmatrix])(self.estparam_obj.predict_crackheating_per_specimen_grad_log_msqrtR)

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
    ignore_datapoints=None # list of (specimen,measnum) values for data points to ignore
    accel_trisolve_devs=None # Optional GPU Acceleration parameters

    # results of load_data() method
    crack_surrogates=None # List of surrogate objects corresponding to each surrogate file (list of nonnegative_denormalized_surrogate objects)
    crackheatfile_dataframes=None # List of Pandas dataframes corresponding to each crack heat file
    crackheat_table = None # Unified Pandas dataframe from all crackheat files; lines with NaN's are omitted

    # parameter_estimation variables
    model=None # pymc3 model
    mu=None
    dummy_observed_variable=None
    log_msqrtR=None
    cracknum=None
    predicted_crackheating=None
    y_like=None  # y_likelihood function; was just "crackheating" 

    M=None # number of unique specimens
    N=None # number of data rows
    
    step=None
    trace=None

    # estimated values
    mu_estimate=None
    log_msqrtR_estimate=None

    # Predicted and actual heatings based on estimates
    predicted=None
    actual=None

    # instance of predict_crackheating_op class (above)
    predict_crackheating_op_instance=None
    

    sigma_additive = None
    sigma_multiplicative = None

    MixedNoiseOp = None

    # prior/parameters
    crackheat_scalefactor = None # Scaling factor multiplied by  heating in Joules/Cycle (or W/Hz) to get values in the model
    mu_prior_mu = None
    mu_prior_sigma = None
    mu_prior = None
    msqrtR_prior_mu = None
    msqrtR_prior_sigma = None
    msqrtR_prior = None
    predicted_crackheating_lower_bound = None # negligible value added to predicted crackheating values. 
    sigma_additive_prior_sigma = None
    sigma_additive_prior = None
    sigma_multiplicative_prior_mu = None
    sigma_multiplicative_prior_sigma = None
    sigma_multiplicative_prior = None

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
    def fromfilelists(cls,crack_specimens,crackheatfiles,surrogatefiles,ignore_datapoints,accel_trisolve_devs=None):
        """Load crack data from parallel lists
        of specimens, crack heating file names, and surrogate .json file names"""
        return cls(crack_specimens=crack_specimens,
                   crackheatfiles=crackheatfiles,
                   surrogatefiles=surrogatefiles,
                   ignore_datapoints=ignore_datapoints,
                   accel_trisolve_devs=accel_trisolve_devs)

    def load_data(self,filter_outside_closure_domain=True,shear=False):
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

        if shear:
            self.crack_surrogates = [ load_denorm_surrogates_shear_from_jsonfile(filename,nonneg=True) for filename in self.surrogatefiles ]  
            pass
        else:
            self.crack_surrogates = [ load_denorm_surrogates_from_jsonfile(filename,nonneg=True) for filename in self.surrogatefiles ]  
            pass
        self.crackheatfile_dataframes = [ pd.read_csv(crackheat_file,dtype={"DynamicNormalStressAmpl (Pa)": str, "DynamicShearStressAmpl (Pa)": str, "BendingStress (Pa)": str}) for crackheat_file in self.crackheatfiles ]
        
        #import pdb
        #pdb.set_trace()

        self.crackheat_table = pd.concat(self.crackheatfile_dataframes,ignore_index=True) # NOTE: Pandas warning about sorting along non-concatenation axis should go away once all data has timestamps


        #assert(np.all(self.crackheat_table["DynamicNormalStressAmpl (Pa)"].map(float) > 100*self.crackheat_table["DynamicShearStressAmpl (Pa)"].map(float))) # Assumption that shear stress is negligible compared to normal stress for all of this data ASSUMPTION ELIMINATED!!!

        # Add specimen numbers to crackheat table
        # If this next line is slow, can easily be accelerated with a dictionary!
        specimen_nums = np.array([ self.crack_specimens.index(specimen) for specimen in self.crackheat_table["Specimen"].values ],dtype=np.uint32)
        self.crackheat_table["specimen_nums"]=specimen_nums

        # Look up closure_lowest_avg_load_used from values loaded into surrogate
        closure_lowest_avg_load_used = np.array([ self.crack_surrogates[specimen_num][list(self.crack_surrogates[specimen_num].keys())[0]].closure_lowest_avg_load_used for specimen_num in self.crackheat_table["specimen_nums"].values ])
        self.crackheat_table["closure_lowest_avg_load_used"]=closure_lowest_avg_load_used
        
        # Drop rows in crackheat_table with NaN ThermalPower
        NaNrownums = np.where(np.isnan(self.crackheat_table["ThermalPower (W)"].values))[0]
    
        self.crackheat_table.drop(NaNrownums,axis=0,inplace=True)
        self.crackheat_table.reset_index(drop=True,inplace=True)

        # Drop rows in crackheat_table corresponding to (specimen,measnum) indicated in ignore_datapoints list
        for (id_specimen,id_measnum) in self.ignore_datapoints:
            ignore_datapoint_rownums = np.where((self.crackheat_table["measnum"].astype(np.int32) == int(id_measnum)) & (self.crackheat_table["Specimen"] == id_specimen))[0]
            self.crackheat_table.drop(ignore_datapoint_rownums,axis=0,inplace=True)
            pass
        self.crackheat_table.reset_index(drop=True,inplace=True)
        

        if filter_outside_closure_domain:
            # Drop rows in crackheat_table with bending stress less than closure_lowest_avg_load_used
            outside_closure_domain_rownums = np.where((~np.isnan(self.crackheat_table["closure_lowest_avg_load_used"].values)) & (self.crackheat_table["closure_lowest_avg_load_used"].values > self.crackheat_table["BendingStress (Pa)"].map(float).values))[0]
            self.crackheat_table.drop(outside_closure_domain_rownums,axis=0,inplace=True)
            self.crackheat_table.reset_index(drop=True,inplace=True)
                                                       
            pass


        specimen_nums_filtered = self.crackheat_table["specimen_nums"]

        
        # number of unique specimens
        self.M = np.unique(specimen_nums_filtered).shape[0]

        # total number of observations 
        self.N = specimen_nums_filtered.shape[0]


        
        pass
    

    def predict_crackheating_grad_mu(self,mu,log_msqrtR): # NOTE: Will need to be updated to accommodate log_mu!!!
        """Predict derivative of crackheating with respect to mu for each row in self.crackheat_table,
        given hypothesized values for mu and log_msqrtR"""

        assert(0)
        retval = np.zeros(self.crackheat_table.shape[0],dtype='d')
        
        # Could parallelize this loop!
        for index in range(self.crackheat_table.shape[0]):
            #datagrid=np.array(((mu,self.crackheat_table["BendingStress (Pa)"].values[index],self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values[index],log_msqrtR),),dtype='d')


            surrogate_key = "bs_pa_" + self.crackheat_table["BendingStress (Pa)"].iloc[index] + "_dnsa_pa_" + self.crackheat_table["DynamicNormalStressAmpl (Pa)"].iloc[index] + "_dssa_pa_" + self.crackheat_table["DynamicShearStressAmpl (Pa)"].iloc[index]

            datagrid = pd.DataFrame(columns=["mu","log_msqrtR"]).astype({"mu":np.float64,"log_msqrtR":np.float64})
            datagrid = datagrid.append({"mu": float(mu),
                                        #"bendingstress": self.crackheat_table["BendingStress (Pa)"].values[index],
                                        #"dynamicstress": self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values[index],
                                        "log_msqrtR": float(log_msqrtR)},ignore_index=True)
            
            retval[index]=self.crack_surrogates[self.crackheat_table["specimen_nums"].values[index]][surrogate_key].evaluate_derivative(datagrid,"mu",accel_trisolve_devs=self.accel_trisolve_devs)[0]
            pass
        return retval


    def predict_crackheating_grad_log_msqrtR(self,log_mu,log_msqrtR):
        """Predict derivative of crackheating with respect to mu for each row in self.crackheat_table,
        given hypothesized values for mu and log_msqrtR"""
        
        retval = np.zeros(self.crackheat_table.shape[0],dtype='d')
        
        # Could parallelize this loop!
        for index in range(self.crackheat_table.shape[0]):
            #datagrid=np.array(((mu,self.crackheat_table["BendingStress (Pa)"].values[index],self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values[index],log_msqrtR),),dtype='d')
            surrogate_key = "bs_pa_" + self.crackheat_table["BendingStress (Pa)"].iloc[index] + "_dnsa_pa_" + self.crackheat_table["DynamicNormalStressAmpl (Pa)"].iloc[index] + "_dssa_pa_" + self.crackheat_table["DynamicShearStressAmpl (Pa)"].iloc[index]


            datagrid = pd.DataFrame(columns=["log_mu","log_msqrtR"]) # .astype({"mu":np.float64,"log_msqrtR":np.float64})
            datagrid = datagrid.append({"log_mu": float(log_mu),
                                        #"bendingstress": self.crackheat_table["BendingStress (Pa)"].values[index],
                                        #"dynamicstress": self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values[index],
                                        "log_msqrtR": float(log_msqrtR)},ignore_index=True)
            
            retval[index]=self.crack_surrogates[self.crackheat_table["specimen_nums"].values[index]][surrogate_key].evaluate_derivative(datagrid,"log_msqrtR",accel_trisolve_devs=self.accel_trisolve_devs)[0]
            pass
        return retval
    
    
    
    def predict_crackheating(self,log_mu,log_msqrtR,log_crack_model_shear_factor=None):
        """Predict crackheating for each row in self.crackheat_table,
        given hypothesized single values for mu and log_msqrtR across the entire dataset"""
        
        retval = np.zeros(self.crackheat_table.shape[0],dtype='d')
        
        # Could parallelize this loop!
        for index in range(self.crackheat_table.shape[0]):
            #datagrid=np.array(((mu,self.crackheat_table["BendingStress (Pa)"].values[index],self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values[index],log_msqrtR),),dtype='d')
            surrogate_key = "bs_pa_" + self.crackheat_table["BendingStress (Pa)"].iloc[index] + "_dnsa_pa_" + self.crackheat_table["DynamicNormalStressAmpl (Pa)"].iloc[index] + "_dssa_pa_" + self.crackheat_table["DynamicShearStressAmpl (Pa)"].iloc[index]

            datagrid = pd.DataFrame(columns=["mu","log_msqrtR"]) #.astype({"mu":np.float64,"log_msqrtR":np.float64})
            if log_crack_model_shear_factor is None:
                datagrid = datagrid.append({"log_mu": float(log_mu),
                                            #"bendingstress": self.crackheat_table["BendingStress (Pa)"].values[index],
                                            #"dynamicstress": self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values[index],
                                            "log_msqrtR": float(log_msqrtR)},ignore_index=True)
                pass
            else:
                datagrid = datagrid.append({"log_mu": float(log_mu),
                                            #"bendingstress": self.crackheat_table["BendingStress (Pa)"].values[index],
                                            #"dynamicstress": self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values[index],
                                            "log_msqrtR": float(log_msqrtR),
                                            "log_crack_model_shear_factor": float(log_crack_model_shear_factor),
                                        },ignore_index=True)                
                pass
            retval[index]=self.crack_surrogates[self.crackheat_table["specimen_nums"].values[index]][surrogate_key].evaluate(datagrid,meanonly=True,accel_trisolve_devs=self.accel_trisolve_devs)["mean"][0]
            if not np.isfinite(retval[index]):
                raise ValueError("Surrogate %s for specimen #%d evaluated to %g at %s" % (surrogate_key,self.crackheat_table["specimen_nums"].values[index],retval[index],str(datagrid)))
            pass
            
        return retval
        

        
    def predict_crackheating_per_specimen(self,mu,log_msqrtR):
        """Predict crackheating for each row in self.crackheat_table,
        given hypothesized values for mu and log_msqrtR per specimen"""
        assert(0)  # !!! Needs to be updated to support log_mu!!!***
        retval = np.zeros(self.crackheat_table.shape[0],dtype='d')
        
        # Could parallelize this loop!
        for index in range(self.crackheat_table.shape[0]):  # ... going from 0...N-1 -- iterating over all the rows in the data
            specimen_num=self.crackheat_table["specimen_nums"].values[index]
            
            surrogate_key = "bs_pa_" + self.crackheat_table["BendingStress (Pa)"].iloc[index] + "_dnsa_pa_" + self.crackheat_table["DynamicNormalStressAmpl (Pa)"].iloc[index] + "_dssa_pa_" + self.crackheat_table["DynamicShearStressAmpl (Pa)"].iloc[index]
            
            #datagrid=np.array(((mu[specimen_num],self.crackheat_table["BendingStress (Pa)"].values[index],self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values[index],log_msqrtR[specimen_num]),),dtype='d')
            
            datagrid = pd.DataFrame(columns=["mu","log_msqrtR"])
            datagrid = datagrid.append({"mu": float(mu[specimen_num]),
                                        #"bendingstress": self.crackheat_table["BendingStress (Pa)"].values[index],
                                        #"dynamicstress": self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values[index],
                                        "log_msqrtR": float(log_msqrtR[specimen_num])},ignore_index=True)
            
            retval[index]=self.crack_surrogates[specimen_num][surrogate_key].evaluate(datagrid,meanonly=True,accel_trisolve_devs=self.accel_trisolve_devs)["mean"][0]
            pass
        return retval


    def predict_crackheating_per_specimen_grad_mu(self,mu,log_msqrtR):
        """Predict derivative of crackheating with respect to mu vector for each row in self.crackheat_table,
        given hypothesized values for mu and log_msqrtR"""
        assert(0) # !!!*** needs to be updated to support log_mu
        retval = np.zeros((self.crackheat_table.shape[0],self.M),dtype='d')
        
        # Could parallelize this loop!
        for index in range(self.crackheat_table.shape[0]):
            specimen_num=self.crackheat_table["specimen_nums"].values[index]
            #datagrid=np.array(((mu[specimen_num],self.crackheat_table["BendingStress (Pa)"].values[index],self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values[index],log_msqrtR[specimen_num]),),dtype='d')
            surrogate_key = "bs_pa_" + self.crackheat_table["BendingStress (Pa)"].iloc[index] + "_dnsa_pa_" + self.crackheat_table["DynamicNormalStressAmpl (Pa)"].iloc[index] + "_dssa_pa_" + self.crackheat_table["DynamicShearStressAmpl (Pa)"].iloc[index]
            

            datagrid = pd.DataFrame(columns=["mu","log_msqrtR"])
            datagrid = datagrid.append({"mu": float(mu[specimen_num]),
                                        #"bendingstress": self.crackheat_table["BendingStress (Pa)"].values[index],
                                        #"dynamicstress": self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values[index],
                                        "log_msqrtR": float(log_msqrtR[specimen_num])},ignore_index=True)
            
            retval[index,specimen_num]=self.crack_surrogates[specimen_num][surrogate_key].evaluate_derivative(datagrid,"mu",accel_trisolve_devs=self.accel_trisolve_devs)[0]
            pass
        return retval



    def predict_crackheating_per_specimen_grad_log_msqrtR(self,mu,log_msqrtR):
        """Predict derivative of crackheating with respect to mu vector for each row in self.crackheat_table,
        given hypothesized per-specimen vectors for mu and log_msqrtR"""
        assert(0) # !!!*** needs to be updated to support log_mu
        
        retval = np.zeros((self.crackheat_table.shape[0],self.M),dtype='d')
        
        # Could parallelize this loop!
        for index in range(self.crackheat_table.shape[0]):
            specimen_num=self.crackheat_table["specimen_nums"].values[index]

            surrogate_key = "bs_pa_" + self.crackheat_table["BendingStress (Pa)"].iloc[index] + "_dnsa_pa_" + self.crackheat_table["DynamicNormalStressAmpl (Pa)"].iloc[index] + "_dssa_pa_" + self.crackheat_table["DynamicShearStressAmpl (Pa)"].iloc[index]
            #datagrid=np.array(((mu[specimen_num],self.crackheat_table["BendingStress (Pa)"].values[index],self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values[index],log_msqrtR[specimen_num]),),dtype='d')
            
            datagrid = pd.DataFrame(columns=["mu","log_msqrtR"])
            datagrid = datagrid.append({"mu": float(mu[specimen_num]),
                                        #"bendingstress": self.crackheat_table["BendingStress (Pa)"].values[index],
                                        #"dynamicstress": self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values[index],
                                        "log_msqrtR": float(log_msqrtR[specimen_num])},ignore_index=True)
            
            retval[index,specimen_num]=self.crack_surrogates[specimen_num][surrogate_key].evaluate_derivative(datagrid,"log_msqrtR",accel_trisolve_devs=self.accel_trisolve_devs)[0]
            pass
        return retval
    

    
    
    def posterior_estimation(self,steps_per_chain,num_chains,cores=None,tune=500):
        """Build and execute PyMC3 Model to obtain self.trace which 
        holds the chain samples"""

        self.model = pm.Model()
    
        with self.model:
            #mu = pm.Uniform('mu',lower=0.01,upper=3.0)
            #msqrtR = pm.Uniform('msqrtR',lower=500000,upper=50e6)

            self.crackheat_scalefactor=1e10
            self.mu_prior_mu = np.log(0.4)
            self.mu_prior_sigma=0.4
            self.mu = Print('mu')(pm.Lognormal('mu',mu=self.mu_prior_mu,sigma=self.mu_prior_sigma))
            self.mu_prior = pm.Lognormal.dist(mu=self.mu_prior_mu,sigma=self.mu_prior_sigma)

            self.msqrtR_prior_mu=np.log(20e6)
            self.msqrtR_prior_sigma=1.0
            
            self.log_msqrtR = np.log(Print('msqrtR')(pm.Lognormal('msqrtR',mu=self.msqrtR_prior_mu,sigma=self.msqrtR_prior_sigma)))  # !!!*** This might need to be changed.... what is the logarithm of a lognormal distribution??? Is it valid? What about negative numbers???
            self.msqrtR_prior = pm.Lognormal.dist(mu=self.msqrtR_prior_mu,sigma=self.msqrtR_prior_sigma)
            
            #sigma_additive_prior_mu = 0.0
            self.predicted_crackheating_lower_bound=1e-12 # Joules/cycle or W/Hz... this is added to the predicted crack heating and should be in the noise. Used to avoid the problem of predicted heatings that are identically zero, making log(prediction) -infinity. 
            self.sigma_additive_prior_sigma_unscaled = 2.0e-8
            self.sigma_additive_prior_sigma = self.sigma_additive_prior_sigma_unscaled*self.crackheat_scalefactor 
            self.sigma_multiplicative_prior_mu = 0.0 #np.log(0.5)
            self.sigma_multiplicative_prior_sigma = 0.75
            
            # priors for sigma_additive and sigma_multiplicative
            self.sigma_additive = Print('sigma_additive')(pm.HalfNormal("sigma_additive",sigma=self.sigma_additive_prior_sigma))
            self.sigma_additive_prior=pm.HalfNormal.dist(sigma=self.sigma_additive_prior_sigma)
            
            self.sigma_multiplicative = Print('sigma_multiplicative')(pm.Lognormal("sigma_multiplicative",mu=self.sigma_multiplicative_prior_mu,sigma=self.sigma_multiplicative_prior_sigma))
            self.sigma_multiplicative_prior = pm.Lognormal.dist(mu=self.sigma_multiplicative_prior_mu,sigma=self.sigma_multiplicative_prior_sigma)

            
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
            

            if cores > 1:
                inhibit_accel_pid=os.getpid() # work around problems with OpenMP threads and Python multiprocessing by inhibiting threads except in the subprocesses
                pass
            else:
                inhibit_accel_pid=None
                pass

            #self.bending_stress = pm.Normal('bending_stress',mu=50e6, sigma=10e6, observed=self.crackheat_table["BendingStress (Pa)"].values)
            #self.dynamic_stress = pm.Normal('dynamic_stress',mu=20e6, sigma=5e6,observed=self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values)
            #self.cracknum = pm.DiscreteUniform('cracknum',lower=0,upper=len(self.crack_specimens)-1,observed=self.crackheat_table["specimen_nums"].values)
            #self.predict_crackheating_op_instance = as_op(itypes=[tt.dscalar,tt.dscalar], otypes = [tt.dvector])(self.predict_crackheating)
            
            use_fast_mixednoise=False
            if not use_fast_mixednoise: 
                self.predict_crackheating_op_instance = predict_crackheating_op(self)

                if check_gradient:
                    # Verify that our op correctly calculates the gradient
                    #theano.tests.unittest_tools.verify_grad(self.predict_crackheating_op_instance,[ np.array(0.3), np.array(5e6)]) # mu=0.3, msqrtR=5e6
                    
                    mu_testval = tt.dscalar('mu_testval')
                    mu_testval.tag.test_value = 0.3 # pymc3 turns on theano's config.compute_test_value switch, so we have to provide a value
                    
                    log_msqrtR_testval = tt.dscalar('log_msqrtR_testval')
                    log_msqrtR_testval.tag.test_value = np.log(5e6) # pymc3 turns on theano's config.compute_test_value switch, so we have to provide a value
                    
                    test_function = theano.function([mu_testval,log_msqrtR_testval],self.predict_crackheating_op_instance(np.log(mu_testval),log_msqrtR_testval))
                    jac_mu = tt.jacobian(self.predict_crackheating_op_instance(np.log(mu_testval),log_msqrtR_testval),mu_testval)
                    jac_mu_analytic = jac_mu.eval({ mu_testval: 0.3, log_msqrtR_testval: np.log(5e6)})
                    jac_mu_numeric = (test_function(0.301,np.log(5e6))-test_function(0.300,np.log(5e6)))/.001
                    assert(np.linalg.norm(jac_mu_analytic-jac_mu_numeric)/np.linalg.norm(jac_mu_analytic) < .05)
                    
                    jac_log_msqrtR = tt.jacobian(self.predict_crackheating_op_instance(np.log(mu_testval),log_msqrtR_testval),log_msqrtR_testval)
                    jac_log_msqrtR_analytic = jac_log_msqrtR.eval({ mu_testval: 0.3, log_msqrtR_testval: np.log(5e6)})
                    jac_log_msqrtR_numeric = (test_function(0.300,np.log(5.0e6)+.001)-test_function(0.300,np.log(5.0e6)))/.001
                    assert(np.linalg.norm(jac_log_msqrtR_analytic-jac_log_msqrtR_numeric)/np.linalg.norm(jac_log_msqrtR_analytic) < .05)
                    
                    try:
                        orig_ctv = theano.config.compute_test_value
                        theano.config.compute_test_value = "off"
                        # Verify that our op correctly calculates the gradient
                        print("grad_mu_values = %s" % (str(self.predict_crackheating_grad_mu(np.array([0.3]), np.array([5e6])))))
                        print("grad_log_msqrtR_values = %s" % (str(self.predict_crackheating_grad_log_msqrtR(np.array([0.3]), np.array([np.log(5e6)])))))
                        
                        # Test gradient with respect to mu
                        theano.tests.unittest_tools.verify_grad(lambda mu_val: self.predict_crackheating_op_instance(np.log(mu_val), theano.shared(5e6)) ,[ 0.3],abs_tol=1e-12,rel_tol=1e-5) # mu=0.3, msqrtR=5e6
                        # Test gradient with respect to msqrtR
                        theano.tests.unittest_tools.verify_grad(lambda log_msqrtR_val: self.predict_crackheating_op_instance(np.log(theano.shared(0.3)),log_msqrtR_val) ,[ np.log(5e6) ],abs_tol=1e-20,rel_tol=1e-8,eps=1.0e-3) # mu=0.3, msqrtR=5e6  NOTE: rel_tol is very tight here because Theano gradient.py/abs_rel_err() lower bounds the relative divisor to 1.e-8 and if we are not tight, we don't actually diagnose errors.
                        # Test combined gradient
                        theano.tests.unittest_tools.verify_grad(self.predict_crackheating_op_instance,[ np.log(0.3), np.log(5e6) ],abs_tol=1e-20,rel_tol=1e-8,eps=1.0e-3) 
                        print("\n\n\nVerify_grad() completed!!!\n\n\n")
                        pass
                    finally:
                        theano.config.compute_test_value = orig_ctv
                        pass
                    pass
                
                
                
                # Create pymc3 predicted_crackheating expression
                #self.predicted_crackheating = self.predict_crackheating_op_instance(np.log(self.mu),self.log_msqrtR)


                self.predicted_crackheating = pm.Deterministic('predicted_crackheating',self.predict_crackheating_op_instance(np.log(self.mu),self.log_msqrtR)) + self.predicted_crackheating_lower_bound 
                
                
            
                (self.MixedNoiseOp,self.y_like) = CreateMixedNoise('y_like',
                                                                   self.sigma_additive,
                                                                   self.sigma_multiplicative,
                                                                   prediction=self.predicted_crackheating*self.crackheat_scalefactor,
                                                                   observed=self.crackheat_scalefactor*self.crackheat_table["ThermalPower (W)"].values/self.crackheat_table["ExcFreq (Hz)"].values,
                                                                   inhibit_accel_pid=inhibit_accel_pid) # ,shape=specimen_nums.shape[0])
                pass
            else:
                # use_fast_mixednoise case... turned out not to be fast after all... will probably
                # not be used. 
                self.sa_sm_mu_lm = tt.as_tensor_variable([self.sigma_additive,
                                                          self.sigma_multiplicative,
                                                          self.mu,
                                                          self.log_msqrtR])
                (self.MixedNoiseOp,self.y_like) = CreateMixedNoiseVibroEstparam('crackheating', self.sa_sm_mu_lm,
                                                                                self.crackheat_scalefactor*self.crackheat_table["ThermalPower (W)"].values/self.crackheat_table["ExcFreq (Hz)"].values,
                                                                                lambda mu, log_msqrtR: self.crackheat_scalefactor*(self.predict_crackheating(mu,log_msqrtR)+self.predicted_crackheating_lower_bound),
                                                                                inhibit_accel_pid = inhibit_accel_pid)
                pass
            
                
            self.step = pm.Metropolis()
            #self.step=pm.NUTS()
            self.trace = pm.sample(steps_per_chain, step=self.step,chains=num_chains, cores=cores,tune=tune,discard_tuned_samples=True) #tune=tune cores=cores chains=num_chains
            trace_df = pm.backends.tracetab.trace_to_dataframe(self.trace,include_transformed=True)

            excfreq_median = np.median(self.crackheat_table["ExcFreq (Hz)"].values)
            
            return (trace_df,
                    # Scalars
                    self.mu_prior_mu,
                    self.mu_prior_sigma,
                    self.msqrtR_prior_mu,
                    self.msqrtR_prior_sigma,
                    self.sigma_additive_prior_sigma_unscaled,
                    self.sigma_multiplicative_prior_mu,
                    self.sigma_multiplicative_prior_sigma,
                    self.crackheat_scalefactor,
                    excfreq_median,
                    self.predicted_crackheating_lower_bound)
        pass



    
    def plot_and_estimate(self,
                          trace_df,
                          mu_prior_mu,
                          mu_prior_sigma,
                          msqrtR_prior_mu,
                          msqrtR_prior_sigma,
                          sigma_additive_prior_sigma_unscaled,
                          sigma_multiplicative_prior_mu,
                          sigma_multiplicative_prior_sigma,
                          crackheat_scalefactor,
                          excfreq_median,
                          predicted_crackheating_lower_bound,
                          marginal_bins=50,joint_bins=(230,200)):
        """ Create diagnostic histograms. Also return coordinates of 
        joint histogram peak as estimates of mu and msqrtR"""

        #trace_df["crackheat_scalefactor"]=self.crackheat_scalefactor
        #trace_df["Median ExcFreq (Hz)"] = np.median(self.crackheat_table["ExcFreq (Hz)"].values)

        from matplotlib import pyplot as pl
        import cycler

        ## Scalars
        #mu_prior_mu=self.mu_prior_mu
        #mu_prior_sigma=self.mu_prior_sigma
        #msqrtR_prior_mu=self.msqrtR_prior_mu
        #msqrtR_prior_sigma=self.msqrtR_prior_sigma
        ##sigma_additive_prior_mu=self.sigma_additive_prior_mu
        #sigma_additive_prior_sigma=self.sigma_additive_prior_sigma
        #sigma_multiplicative_prior_mu=self.sigma_multiplicative_prior_mu
        #sigma_multiplicative_prior_sigma=self.sigma_multiplicative_prior_sigma

        
        #crackheat_scalefactor=self.crackheat_scalefactor
        #excfreq_median = np.median(self.crackheat_table["ExcFreq (Hz)"].values)

        mu_vals=trace_df["mu"].values # self.trace.get_values("mu")
        msqrtR_vals = trace_df["msqrtR"].values # self.trace.get_values("msqrtR")
        sigma_additive_vals=trace_df["sigma_additive"].values/crackheat_scalefactor # self.trace.get_values("sigma_additive")/self.crackheat_scalefactor
        sigma_multiplicative_vals=trace_df["sigma_multiplicative"].values #self.trace.get_values("sigma_multiplicative")
        
        # # read in predicted_crackheating
        # predicted_crackheating=np.zeros((mu_vals.shape[0],self.crackheat_table["ThermalPower (W)"].values.shape[0]),dtype='d')
        # for datapt_idx in range(self.crackheat_table["ThermalPower (W)"].values.shape[0]):
        #     predicted_crackheating[:,datapt_idx] = trace_df["predicted_crackheating__%d" % (datapt_idx)]
        #     pass

        # prediction_factors = predicted_crackheating/(self.crackheat_table["ThermalPower (W)"].values/self.crackheat_table["ExcFreq (Hz)"].values)[np.newaxis,:]

        
        # Results
        mu_estimate = np.median(mu_vals)
        mu_sd=np.std(mu_vals)
        log_mu_estimate = np.median(np.log(mu_vals))
        log_mu_sd=np.std(np.log(mu_vals))
        msqrtR_estimate = np.median(msqrtR_vals)
        msqrtR_sd=np.std(msqrtR_vals)
        log_msqrtR_estimate = np.median(np.log(msqrtR_vals))
        log_msqrtR_sd = np.std(np.log(msqrtR_vals))
        #self.log_msqrtR_estimate = np.median(np.log(msqrtR_vals))
        #self.log_msqrtR_sd=np.std(np.log(msqrtR_vals))
        sigma_additive_estimate = np.median(sigma_additive_vals)
        sigma_additive_sd=np.std(sigma_additive_vals)
        sigma_multiplicative_estimate = np.median(sigma_multiplicative_vals)
        sigma_multiplicative_sd=np.std(sigma_multiplicative_vals)
        log_sigma_multiplicative_estimate = np.median(np.log(sigma_multiplicative_vals))
        log_sigma_multiplicative_sd=np.std(np.log(sigma_multiplicative_vals))
        
        results = (mu_estimate,
                   mu_sd,
                   log_mu_estimate,
                   log_mu_sd,
                   msqrtR_estimate,
                   msqrtR_sd,
                   log_msqrtR_estimate,
                   log_msqrtR_sd,
                   sigma_additive_estimate,
                   sigma_additive_sd,
                   sigma_multiplicative_estimate,
                   sigma_multiplicative_sd,
                   log_sigma_multiplicative_estimate,
                   log_sigma_multiplicative_sd)
        
        
        #traceplots=pl.figure()

        # Current arviz traceplots do not work because of attempt to access 'observed' member 
        #     https://discourse.pymc.io/t/pm-traceplot-error/3524/8
        #     https://github.com/arviz-devs/arviz/pull/731
        # ... Attempted to apply patch to installed copy, but it did not help
        #traceplot_axes = pm.traceplot(self.trace)
        #traceplots = traceplot_axes[0,0].figure
        
        traceplots = pl.figure()
        pl.subplot(2,2,1)
        pl.plot(mu_vals)
        pl.title('mu')
        pl.subplot(2,2,2)
        pl.plot(msqrtR_vals)
        pl.title('msqrtR')
        pl.subplot(2,2,3)
        pl.plot(sigma_additive_vals)
        pl.title('sigma_additive')
        pl.subplot(2,2,4)
        pl.plot(sigma_multiplicative_vals)
        pl.tight_layout(pad=1.08,h_pad=2.0)
        pl.title('sigma_multiplicative')
        

        
        gaussian = lambda x,mu,sigma : (1.0/(sigma*np.sqrt(2.0*np.pi)))*np.exp(-((x-mu)**2.0)/(2.0*sigma**2.0))
        lognormal = lambda x,mu,sigma : (1.0/(x*sigma*np.sqrt(2.0*np.pi)))*np.exp(-((np.log(x)-mu)**2.0)/(2.0*sigma**2.0))
        halfnormal = lambda x,sigma: (np.sqrt(2)/(sigma*np.sqrt(np.pi)))*np.exp(-(x**2.0)/(2.0*sigma**2.0))        

        mu_hist = pl.figure()
        pl.clf()
        pl.hist(mu_vals,bins=marginal_bins,density=True)
        mu_range=np.linspace(0,pl.axis()[1],100)
        pl.plot(mu_range,lognormal(mu_range,mu_prior_mu,mu_prior_sigma),'-')
        pl.plot(mu_range,gaussian(mu_range,mu_estimate,mu_sd),'-')
        pl.plot(mu_range,lognormal(mu_range,log_mu_estimate,log_mu_sd),'-')
        pl.xlabel('mu (unitless)')
        pl.ylabel('Probability density')
        pl.title('mu lognormal mu = %g;\nmu lognormal sd = %g' % (log_mu_estimate,log_mu_sd))
        pl.legend(('Prior','Posterior approx. (normal)','Posteror approx. (lognormal)','MCMC Histogram'),loc="best")
        pl.grid()
        
        
        msqrtR_hist = pl.figure()
        pl.clf()
        pl.hist(msqrtR_vals/1e7,bins=marginal_bins,density=True)
        msqrtR_range=np.linspace(0,pl.axis()[1]*1e7,100)
        pl.plot(msqrtR_range/1e7,lognormal(msqrtR_range,msqrtR_prior_mu,msqrtR_prior_sigma)*1e7,'-')
        pl.plot(msqrtR_range/1e7,gaussian(msqrtR_range,msqrtR_estimate,msqrtR_sd)*1e7,'-')
        pl.plot(msqrtR_range/1e7,lognormal(msqrtR_range,log_msqrtR_estimate,log_msqrtR_sd)*1e7,'-')
        pl.xlabel('m*sqrtR')
        pl.title('m*sqrtR lognormal mu = %g;\nm*sqrtR lognormal sd = %g' % (log_msqrtR_estimate,log_msqrtR_sd))
        pl.xlabel('m*sqrt(R) (10^7 1/m^(3/2))')
        pl.ylabel('Probability density (1/10^7 m^(3/2))')
        pl.legend(('Prior','Posterior approx. (normal)','Posteror approx. (lognormal)','MCMC Histogram'),loc="best")
        pl.grid()


        #log_msqrtR_hist = pl.figure()
        #pl.clf()
        #pl.hist(np.log(msqrtR_vals),bins=marginal_bins,density=True)
        #log_msqrtR_range=np.linspace(0,pl.axis()[1],100)
        #pl.plot(log_msqrtR_range,np.exp(self.msqrtR_prior.logp(np.exp(log_msqrtR_range))).eval(),'-')
        #pl.plot(log_msqrtR_range,gaussian(log_msqrtR_range,self.log_msqrtR_estimate,self.log_msqrtR_sd),'-')
        #pl.xlabel('log(m*sqrtR)')
        #pl.legend(('Prior','Posterior approx.','MCMC Histogram'),loc="best")
        #pl.grid()

        sigma_additive_hist = pl.figure()
        pl.clf()
        pl.hist(sigma_additive_vals*1e9,bins=marginal_bins,density=True)
        sa_range=np.linspace(0,pl.axis()[1]/1e9,100)
        pl.plot(sa_range*1e9,halfnormal(sa_range,sigma_additive_prior_sigma_unscaled)/1e9,'-')
        pl.plot(sa_range*1e9,gaussian(sa_range,sigma_additive_estimate,sigma_additive_sd)/1e9,'-')
        pl.xlabel('sigma_additive (10^-9 J/cy)')
        pl.ylabel('Probability density (10^9 cy/J)')
        pl.legend(('Prior','Posterior approx.','MCMC Histogram'),loc="best")
        pl.title('sigma_additive normal mu = %g;\nsigma_additive normal sd = %g' % (sigma_additive_estimate,sigma_additive_sd))
        pl.grid()

        sigma_additive_power_hist = pl.figure()
        pl.clf()
        pl.hist(sigma_additive_vals*1e3*excfreq_median,bins=marginal_bins,density=True)
        sap_range=np.linspace(0,pl.axis()[1]/1e3,100) # note: sap_range is in W
        pl.plot(sap_range*1e3,halfnormal(sap_range/excfreq_median,sigma_additive_prior_sigma_unscaled)/1.e3/excfreq_median,'-')
        pl.plot(sap_range*1e3,gaussian(sap_range/excfreq_median,sigma_additive_estimate,sigma_additive_sd)/1.e3/excfreq_median,'-')
        pl.xlabel('sigma_additive (mW) assuming typ freq of %f kHz' % (excfreq_median/1e3))
        pl.ylabel('Probability density (1/mW)')
        pl.legend(('Prior','Posterior approx.','MCMC Histogram'),loc="best")
        pl.title('sigma_additive(W) normal mu = %g;\nsigma_additive(W) normal sd = %g' % (sigma_additive_estimate*excfreq_median,sigma_additive_sd*excfreq_median))
        pl.grid()


        sigma_multiplicative_hist = pl.figure()
        pl.clf()
        pl.hist(sigma_multiplicative_vals,bins=marginal_bins,density=True)
        sm_range=np.linspace(0,pl.axis()[1],100)
        pl.plot(sm_range,lognormal(sm_range,sigma_multiplicative_prior_mu,sigma_multiplicative_prior_sigma),'-')
        pl.plot(sm_range,gaussian(sm_range,sigma_multiplicative_estimate,sigma_multiplicative_sd),'-')
        pl.plot(sm_range,lognormal(sm_range,log_sigma_multiplicative_estimate,log_sigma_multiplicative_sd),'-')
        pl.xlabel('sigma_multiplicative (unitless)')
        pl.xlabel('Probability density (unitless)')
        pl.title('sigma_mul lognormal mu = %g;\nsigma_mul lognormal sd = %g' % (log_sigma_multiplicative_estimate,log_sigma_multiplicative_sd))
        pl.legend(('Prior','Posterior approx. (normal)','Posteror approx. (lognormal)','MCMC Histogram'),loc="best")
        pl.grid()

        
        probit = lambda x: np.sqrt(2)*scipy.special.erfinv(2.0*x-1.0)

        sigma_multiplicative_pdfs = pl.figure()
        # heating is multiplied by multiplicative noise which is presumed to be lognormally distributed with a mu of ln(predicted heating) and variance of sigma_multiplicative_estimate
        # For this plot we represent it relative ot a predicted heating of 1.0 so ln(predicted heating) = 0 
        pl.clf()
        smp_range=np.linspace(0.001,4.0,200)
        smp_function = lambda x,sigma : (1.0/(x*sigma*np.sqrt(2.0*np.pi)))*np.exp(-((np.log(x))**2.0)/(2.0*sigma**2.0))
        pl.plot(smp_range,smp_function(smp_range,np.exp(log_sigma_multiplicative_estimate)),'-',
                smp_range,smp_function(smp_range,np.exp(log_sigma_multiplicative_estimate-probit(.975)*sigma_multiplicative_sd)),'--',
                smp_range,smp_function(smp_range,np.exp(log_sigma_multiplicative_estimate+probit(.975)*sigma_multiplicative_sd)),'--')
        #log_smp_range=np.linspace(-7,1.4,200)
        ## sigma_multiplicative is log-normally distributed so log_sigma_multiplicative should be normally distributed
        #log_smp_function = lambda x,mu,sigma : (1.0/(sigma*np.sqrt(2.0*np.pi)))*np.exp(-((x-mu)**2.0)/(2.0*sigma**2.0))
        #pl.plot(np.exp(log_smp_range),log_smp_function(log_smp_range,0,np.exp(log_sigma_multiplicative_estimate)),'-',
        #        np.exp(log_smp_range),log_smp_function(log_smp_range,0,np.exp(log_sigma_multiplicative_estimate-probit(.975)*log_sigma_multiplicative_sd)),'--',
        #np.exp(log_smp_range),log_smp_function(log_smp_range,0,np.exp(log_sigma_multiplicative_estimate+probit(.975)*log_sigma_multiplicative_sd)),'--')
        pl.xlabel('multiplier')
        pl.title('Probability density for multiplicative error') 
        pl.legend(('Based on best estimate (median)','97.5% lower bound for 95% conf interval','97.5% upper bound for 95% conf interval'))
        pl.grid()

        
    
        joint_hist = pl.figure()
        pl.clf()
        (hist,hist_mu_edges,hist_msqrtR_edges,hist_image)=pl.hist2d(mu_vals,msqrtR_vals,
                                                                    range=((mu_estimate-3.0*mu_sd,
                                                                            mu_estimate+3.0*mu_sd),
                                                                           (msqrtR_estimate-3.0*msqrtR_sd,
                                                                            msqrtR_estimate+3.0*msqrtR_sd)),
                                                                    bins=joint_bins)
        pl.grid()
        pl.colorbar()
        pl.xlabel('mu')
        pl.ylabel('m*sqrt(R) (sqrt(m)/m^2)')
        
        #histpeakpos = np.unravel_index(np.argmax(hist,axis=None),hist.shape)
        #self.mu_estimate = (hist_mu_edges[histpeakpos[0]]+hist_mu_edges[histpeakpos[0]+1])/2.0
        #self.msqrtR_estimate = (hist_msqrtR_edges[histpeakpos[1]]+hist_msqrtR_edges[histpeakpos[1]+1])/2.0
    
        # Compare
        predicted = self.predict_crackheating(mu_estimate,np.log(msqrtR_estimate))*self.crackheat_table["ExcFreq (Hz)"].values
        
        # add to crackheat_table
        
        self.crackheat_table["predicted"]=predicted

        ## with:
        #self.actual = self.crackheat_table["ThermalPower (W)"].values

        # Group predicted and actual heating by specimen
        specimen_grouping = self.crackheat_table.groupby("Specimen")

        specimen_groups = [ specimen_grouping.get_group(specimen) for specimen in specimen_grouping.groups ]
        predicted_by_specimen = [ specimen_group["predicted"].values for specimen_group in specimen_groups ]
        actual_by_specimen = [ specimen_group["ThermalPower (W)"].values for specimen_group in specimen_groups ]
        specimen_group_specimens = list(specimen_grouping.groups) 

        markerstyle_cycler=cycler.cycler(marker=['o','v','^','<','>','s','p','+','x'])()
        prediction_plot = pl.figure()
        pl.clf()
        #pl.plot(predicted,self.actual,'x',
        #        (0,np.max(predicted)),(0,np.max(predicted)),'-')
        [ pl.plot(predicted_by_specimen[idx]*1e3,actual_by_specimen[idx]*1e3,linestyle='',**next(markerstyle_cycler)) for idx in range(len(specimen_group_specimens)) ] 
        pl.plot((0,np.max(predicted)*1.6*1e3),(0,np.max(predicted)*1.6*1e3),'-')
        pl.legend(specimen_group_specimens,loc='lower right')
        pl.xlabel('Predicted heating from model (mW)')
        pl.ylabel('Actual heating from experiment (mW)')
        pl.title('mu_estimate=%.3g; msqrtR_estimate=%.3g  m^(-3/2)' % (mu_estimate,msqrtR_estimate))
        pl.grid()

        markerstyle_cyclerz=cycler.cycler(marker=['o','v','^','<','>','s','p','+','x'])()
        prediction_zoom_plot = pl.figure()
        pl.clf()
        #pl.plot(predicted,self.actual,'x',
        #        (0,np.max(predicted)),(0,np.max(predicted)),'-')
        [ pl.plot(predicted_by_specimen[idx]*1e3,actual_by_specimen[idx]*1e3,linestyle='',**next(markerstyle_cyclerz)) for idx in range(len(specimen_group_specimens)) ] 
        pl.plot((0,np.max(predicted)*1.6*1e3),(0,np.max(predicted)*1.6*1e3),'-')
        pl.legend(specimen_group_specimens,loc='lower right')
        pl.axis((0,np.max(predicted)*0.25*1e3,0,np.max(predicted)*0.2*1e3))
        pl.xlabel('Predicted heating from model (mW)')
        pl.ylabel('Actual heating from experiment (mW)')
        pl.title('mu_estimate=%.3g; msqrtR_estimate=%.3g m^(-3/2)' % (mu_estimate,msqrtR_estimate))
        pl.grid()


        plots = (traceplots,
                 mu_hist,
                 msqrtR_hist,
                 sigma_additive_hist,
                 sigma_additive_power_hist,
                 sigma_multiplicative_hist,
                 sigma_multiplicative_pdfs,
                 joint_hist,
                 prediction_plot,
                 prediction_zoom_plot)

        # self.MixedNoiseOp(theano.shared(self.trace.get_values("sigma_additive")[0]),theano.shared(self.trace.get_values("sigma_multiplicative")[0]),theano.shared(self.trace.get_values("predicted_crackheating")[0,:])).eval() ... gives log(p) = 10
        # self.MixedNoiseOp.evaluate_p_from_cache(self.trace.get_values("sigma_additive")[0],self.trace.get_values("sigma_multiplicative")[0],self.trace.get_values("predicted_crackheating")[0,0],self.MixedNoiseOp.observed[0])
        
        #pdf_integral1 = scipy.integrate.quad(lambda obs: self.MixedNoiseOp.integrate_kernel(self.MixedNoiseOp.lognormal_normal_convolution_integral_y_zero_to_eps,self.MixedNoiseOp.lognormal_normal_convolution_kernel,self.trace.get_values("sigma_additive")[0],self.trace.get_values("sigma_multiplicative")[0],self.trace.get_values("predicted_crackheating")[0,1]*crackheat_scalefactor,obs),150.0,np.inf)[0]

        #pdf_integral2 = scipy.integrate.quad(lambda obs: self.MixedNoiseOp.integrate_kernel(self.MixedNoiseOp.lognormal_normal_convolution_integral_y_zero_to_eps,self.MixedNoiseOp.lognormal_normal_convolution_kernel,self.trace.get_values("sigma_additive")[0],self.trace.get_values("sigma_multiplicative")[0],self.trace.get_values("predicted_crackheating")[0,1]*crackheat_scalefactor,obs),.0000001,150.0)[0]
        
        #import theano.tests.unittest_tools
        #theano.config.compute_test_value = "off"

        #theano.tests.unittest_tools.verify_grad(self.MixedNoiseOp,[ self.trace.get_values("sigma_additive")[0], self.trace.get_values("sigma_multiplicative")[0],self.trace.get_values("predicted_crackheating")[0,:]*self.crackheat_scalefactor ],abs_tol=1e-12,rel_tol=1e-5,eps=1e-7)

        #raise ValueError("Foo!")

        #return (self.mu_estimate,self.msqrtR_estimate,traceplots,mu_hist,msqrtR_hist,joint_hist,prediction_plot)
        

        return (results,plots)






    
    
    def posterior_estimation_shear(self,mu_prior_mu,mu_prior_sigma,msqrtR_prior_mu,msqrtR_prior_sigma,sigma_additive_prior_mu_unscaled,sigma_additive_prior_sigma_unscaled,sigma_multiplicative_prior_mu,sigma_multiplicative_prior_sigma,steps_per_chain,num_chains,cores=None,tune=500):
        """Build and execute PyMC3 Model to obtain self.trace which 
        holds the chain samples

        NOTE: Do not run posterior_estimation shear and posterior_estimation 
        at the same time on the same estparam object!!!"""

        self.model = pm.Model()
    
        with self.model:
            #mu = pm.Uniform('mu',lower=0.01,upper=3.0)
            #msqrtR = pm.Uniform('msqrtR',lower=500000,upper=50e6)

            self.crackheat_scalefactor=1e10
            self.mu_prior_mu = mu_prior_mu
            self.mu_prior_sigma=mu_prior_sigma
            self.mu = Print('mu')(pm.Lognormal('mu',mu=self.mu_prior_mu,sigma=self.mu_prior_sigma))
            self.mu_prior = pm.Lognormal.dist(mu=self.mu_prior_mu,sigma=self.mu_prior_sigma)

            self.msqrtR_prior_mu=msqrtR_prior_mu
            self.msqrtR_prior_sigma=msqrtR_prior_sigma
            
            self.log_msqrtR = np.log(Print('msqrtR')(pm.Lognormal('msqrtR',mu=self.msqrtR_prior_mu,sigma=self.msqrtR_prior_sigma)))  # !!!*** This might need to be changed.... the logarithm of a lognormal distribution is a normal distribution...
            self.msqrtR_prior = pm.Lognormal.dist(mu=self.msqrtR_prior_mu,sigma=self.msqrtR_prior_sigma)
            
            self.crack_model_shear_factor_prior_mu=2.2
            self.crack_model_shear_factor_prior_sigma=3.0
            self.crack_model_shear_factor=pm.Lognormal('crack_model_shear_factor',mu=self.crack_model_shear_factor_prior_mu,sigma=self.crack_model_shear_factor_prior_sigma)
            self.crack_model_shear_factor_prior=pm.Lognormal.dist(mu=self.crack_model_shear_factor_prior_mu,sigma=self.crack_model_shear_factor_prior_sigma)
            self.log_crack_model_shear_factor = np.log(self.crack_model_shear_factor)
            

            #sigma_additive_prior_mu = 0.0
            self.predicted_crackheating_lower_bound=1e-12 # Joules/cycle or W/Hz... this is added to the predicted crack heating and should be in the noise. Used to avoid the problem of predicted heatings that are identically zero, making log(prediction) -infinity. 

            self.sigma_additive_prior_sigma_unscaled = sigma_additive_prior_sigma_unscaled
            self.sigma_additive_prior_sigma = self.sigma_additive_prior_sigma_unscaled*self.crackheat_scalefactor 
            self.sigma_additive_prior_mu_unscaled = sigma_additive_prior_mu_unscaled
            self.sigma_additive_prior_mu = self.sigma_additive_prior_mu_unscaled*self.crackheat_scalefactor 
            self.sigma_multiplicative_prior_mu = sigma_multiplicative_prior_mu #np.log(0.5)
            self.sigma_multiplicative_prior_sigma = sigma_multiplicative_prior_sigma
            
            # priors for sigma_additive and sigma_multiplicative
            # Note that here (shear case) we use a normal prior for sigma_additive because the values have already been estimated for the normal case
            self.sigma_additive = Print('sigma_additive')(pm.Normal("sigma_additive",mu=self.sigma_additive_prior_mu,sigma=self.sigma_additive_prior_sigma))
            self.sigma_additive_prior=pm.HalfNormal.dist(sigma=self.sigma_additive_prior_sigma)
            
            self.sigma_multiplicative = Print('sigma_multiplicative')(pm.Lognormal("sigma_multiplicative",mu=self.sigma_multiplicative_prior_mu,sigma=self.sigma_multiplicative_prior_sigma))
            self.sigma_multiplicative_prior = pm.Lognormal.dist(mu=self.sigma_multiplicative_prior_mu,sigma=self.sigma_multiplicative_prior_sigma)

            
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
            

            if cores > 1:
                inhibit_accel_pid=os.getpid() # work around problems with OpenMP threads and Python multiprocessing by inhibiting threads except in the subprocesses
                pass
            else:
                inhibit_accel_pid=None
                pass

            #self.bending_stress = pm.Normal('bending_stress',mu=50e6, sigma=10e6, observed=self.crackheat_table["BendingStress (Pa)"].values)
            #self.dynamic_stress = pm.Normal('dynamic_stress',mu=20e6, sigma=5e6,observed=self.crackheat_table["DynamicNormalStressAmpl (Pa)"].values)
            #self.cracknum = pm.DiscreteUniform('cracknum',lower=0,upper=len(self.crack_specimens)-1,observed=self.crackheat_table["specimen_nums"].values)
            #self.predict_crackheating_op_instance = as_op(itypes=[tt.dscalar,tt.dscalar], otypes = [tt.dvector])(self.predict_crackheating)
            
            # Shear calculation doesn't support derivatives so we can just use as_op()
            self.predict_crackheating_op_instance = as_op(itypes=[tt.dscalar,tt.dscalar,tt.dscalar], otypes=[tt.dvector])(self.predict_crackheating)
            
                
                
            # Create pymc3 predicted_crackheating expression
            #self.predicted_crackheating = self.predict_crackheating_op_instance(self.mu,self.log_msqrtR)


            self.predicted_crackheating = pm.Deterministic('predicted_crackheating',self.predict_crackheating_op_instance(np.log(self.mu),self.log_msqrtR,self.log_crack_model_shear_factor)) + self.predicted_crackheating_lower_bound 
                
                
            
            (self.MixedNoiseOp,self.y_like) = CreateMixedNoise('y_like',
                                                               self.sigma_additive,
                                                               self.sigma_multiplicative,
                                                               prediction=self.predicted_crackheating*self.crackheat_scalefactor,
                                                               observed=self.crackheat_scalefactor*self.crackheat_table["ThermalPower (W)"].values/self.crackheat_table["ExcFreq (Hz)"].values,
                                                               inhibit_accel_pid=inhibit_accel_pid) # ,shape=specimen_nums.shape[0])
            
            self.step = pm.Metropolis()
            #self.step=pm.NUTS()
            self.trace = pm.sample(steps_per_chain, step=self.step,chains=num_chains, cores=cores,tune=tune,discard_tuned_samples=True) #tune=tune cores=cores chains=num_chains
            trace_df = pm.backends.tracetab.trace_to_dataframe(self.trace,include_transformed=True)

            excfreq_median = np.median(self.crackheat_table["ExcFreq (Hz)"].values)
            
            return (trace_df,
                    # Scalars
                    self.crack_model_shear_factor_prior_mu,
                    self.crack_model_shear_factor_prior_sigma,
                    self.crackheat_scalefactor,
                    excfreq_median,
                    self.predicted_crackheating_lower_bound)
        pass



    
    def plot_and_estimate_shear(self,
                                trace_df,
                                mu_prior_mu,
                                mu_prior_sigma,
                                msqrtR_prior_mu,
                                msqrtR_prior_sigma,
                                crack_model_shear_factor_prior_mu,
                                crack_model_shear_factor_prior_sigma,
                                sigma_additive_prior_mu_unscaled,
                                sigma_additive_prior_sigma_unscaled,
                                sigma_multiplicative_prior_mu,
                                sigma_multiplicative_prior_sigma,
                                crackheat_scalefactor,
                                excfreq_median,
                                predicted_crackheating_lower_bound,
                                marginal_bins=50,joint_bins=(230,200)):
        """ Create diagnostic histograms. Also return coordinates of 
        joint histogram peak as estimates of mu and msqrtR"""

        #trace_df["crackheat_scalefactor"]=self.crackheat_scalefactor
        #trace_df["Median ExcFreq (Hz)"] = np.median(self.crackheat_table["ExcFreq (Hz)"].values)

        from matplotlib import pyplot as pl
        import cycler

        ## Scalars
        #mu_prior_mu=self.mu_prior_mu
        #mu_prior_sigma=self.mu_prior_sigma
        #msqrtR_prior_mu=self.msqrtR_prior_mu
        #msqrtR_prior_sigma=self.msqrtR_prior_sigma
        ##sigma_additive_prior_mu=self.sigma_additive_prior_mu
        #sigma_additive_prior_sigma=self.sigma_additive_prior_sigma
        #sigma_multiplicative_prior_mu=self.sigma_multiplicative_prior_mu
        #sigma_multiplicative_prior_sigma=self.sigma_multiplicative_prior_sigma

        
        #crackheat_scalefactor=self.crackheat_scalefactor
        #excfreq_median = np.median(self.crackheat_table["ExcFreq (Hz)"].values)

        mu_vals=trace_df["mu"].values # self.trace.get_values("mu")
        msqrtR_vals = trace_df["msqrtR"].values # self.trace.get_values("msqrtR")
        crack_model_shear_factor_vals = trace_df["crack_model_shear_factor"].values # self.trace.get_values("msqrtR")
        sigma_additive_vals=trace_df["sigma_additive"].values/crackheat_scalefactor # self.trace.get_values("sigma_additive")/self.crackheat_scalefactor
        sigma_multiplicative_vals=trace_df["sigma_multiplicative"].values #self.trace.get_values("sigma_multiplicative")
        
        # # read in predicted_crackheating
        # predicted_crackheating=np.zeros((mu_vals.shape[0],self.crackheat_table["ThermalPower (W)"].values.shape[0]),dtype='d')
        # for datapt_idx in range(self.crackheat_table["ThermalPower (W)"].values.shape[0]):
        #     predicted_crackheating[:,datapt_idx] = trace_df["predicted_crackheating__%d" % (datapt_idx)]
        #     pass

        # prediction_factors = predicted_crackheating/(self.crackheat_table["ThermalPower (W)"].values/self.crackheat_table["ExcFreq (Hz)"].values)[np.newaxis,:]
        
        # Results
        mu_estimate = np.median(mu_vals)
        mu_sd=np.std(mu_vals)
        log_mu_estimate = np.median(np.log(mu_vals))
        log_mu_sd=np.std(np.log(mu_vals))
        msqrtR_estimate = np.median(msqrtR_vals)
        msqrtR_sd=np.std(msqrtR_vals)
        log_msqrtR_estimate = np.median(np.log(msqrtR_vals))
        log_msqrtR_sd = np.std(np.log(msqrtR_vals))
        crack_model_shear_factor_estimate = np.median(crack_model_shear_factor_vals)
        crack_model_shear_factor_sd = np.std(crack_model_shear_factor_vals)
        log_crack_model_shear_factor_estimate = np.median(np.log(crack_model_shear_factor_vals))
        log_crack_model_shear_factor_sd = np.std(np.log(crack_model_shear_factor_vals))
        #self.log_msqrtR_estimate = np.median(np.log(msqrtR_vals))
        #self.log_msqrtR_sd=np.std(np.log(msqrtR_vals))
        sigma_additive_estimate = np.median(sigma_additive_vals)
        sigma_additive_sd=np.std(sigma_additive_vals)
        sigma_multiplicative_estimate = np.median(sigma_multiplicative_vals)
        sigma_multiplicative_sd=np.std(sigma_multiplicative_vals)
        log_sigma_multiplicative_estimate = np.median(np.log(sigma_multiplicative_vals))
        log_sigma_multiplicative_sd=np.std(np.log(sigma_multiplicative_vals))
        

        results = (mu_estimate,
                   mu_sd,
                   log_mu_estimate,
                   log_mu_sd,
                   msqrtR_estimate,
                   msqrtR_sd,
                   log_msqrtR_estimate,
                   log_msqrtR_sd,
                   crack_model_shear_factor_estimate,
                   crack_model_shear_factor_sd,
                   log_crack_model_shear_factor_estimate,
                   log_crack_model_shear_factor_sd,
                   sigma_additive_estimate,
                   sigma_additive_sd,
                   sigma_multiplicative_estimate,
                   sigma_multiplicative_sd,
                   log_sigma_multiplicative_estimate,
                   log_sigma_multiplicative_sd)
        
        
        #traceplots=pl.figure()

        # Current arviz traceplots do not work because of attempt to access 'observed' member 
        #     https://discourse.pymc.io/t/pm-traceplot-error/3524/8
        #     https://github.com/arviz-devs/arviz/pull/731
        # ... Attempted to apply patch to installed copy, but it did not help
        #traceplot_axes = pm.traceplot(self.trace)
        #traceplots = traceplot_axes[0,0].figure
        
        traceplots = pl.figure()
        pl.subplot(3,2,1)
        pl.plot(mu_vals)
        pl.title('mu')
        pl.subplot(3,2,2)
        pl.plot(msqrtR_vals)
        pl.title('msqrtR')
        pl.subplot(3,2,3)
        pl.plot(crack_model_shear_factor_vals)
        pl.title('crack_model_shear_factor')
        pl.subplot(3,2,4)
        pl.plot(sigma_additive_vals)
        pl.title('sigma_additive')
        pl.subplot(3,2,5)
        pl.plot(sigma_multiplicative_vals)
        pl.title('sigma_multiplicative')
        

        
        gaussian = lambda x,mu,sigma : (1.0/(sigma*np.sqrt(2.0*np.pi)))*np.exp(-((x-mu)**2.0)/(2.0*sigma**2.0))
        lognormal = lambda x,mu,sigma : (1.0/(x*sigma*np.sqrt(2.0*np.pi)))*np.exp(-((np.log(x)-mu)**2.0)/(2.0*sigma**2.0))
        halfnormal = lambda x,sigma: (np.sqrt(2)/(sigma*np.sqrt(np.pi)))*np.exp(-(x**2.0)/(2.0*sigma**2.0))        

        mu_hist = pl.figure()
        pl.clf()
        pl.hist(mu_vals,bins=marginal_bins,density=True)
        mu_range=np.linspace(0,pl.axis()[1],100)
        pl.plot(mu_range,lognormal(mu_range,mu_prior_mu,mu_prior_sigma),'-')
        pl.plot(mu_range,gaussian(mu_range,mu_estimate,mu_sd),'-')
        pl.plot(mu_range,lognormal(mu_range,log_mu_estimate,log_mu_sd),'-')
        pl.xlabel('mu')
        pl.title('mu lognormal mu = %g; mu lognormal sd = %g' % (log_mu_estimate,log_mu_sd))
        pl.legend(('Prior','Posterior approx. (normal)','Posteror approx. (lognormal)','MCMC Histogram'),loc="best")
        pl.grid()
        
        
        msqrtR_hist = pl.figure()
        pl.clf()
        pl.hist(msqrtR_vals,bins=marginal_bins,density=True)
        msqrtR_range=np.linspace(0,pl.axis()[1],100)
        pl.plot(msqrtR_range,lognormal(msqrtR_range,msqrtR_prior_mu,msqrtR_prior_sigma),'-')
        pl.plot(msqrtR_range,gaussian(msqrtR_range,msqrtR_estimate,msqrtR_sd),'-')
        pl.plot(msqrtR_range,lognormal(msqrtR_range,log_msqrtR_estimate,log_msqrtR_sd),'-')
        pl.xlabel('m*sqrtR')
        pl.title('m*sqrtR lognormal mu = %g; m*sqrtR lognormal sd = %g' % (log_msqrtR_estimate,log_msqrtR_sd))
        pl.legend(('Prior','Posterior approx. (normal)','Posteror approx. (lognormal)','MCMC Histogram'),loc="best")
        pl.grid()

        crack_model_shear_factor_hist = pl.figure()
        pl.clf()
        pl.hist(crack_model_shear_factor_vals,bins=marginal_bins,density=True)
        crack_model_shear_factor_range=np.linspace(0,pl.axis()[1],100)
        pl.plot(crack_model_shear_factor_range,lognormal(crack_model_shear_factor_range,crack_model_shear_factor_prior_mu,crack_model_shear_factor_prior_sigma),'-')
        pl.plot(crack_model_shear_factor_range,gaussian(crack_model_shear_factor_range,crack_model_shear_factor_estimate,crack_model_shear_factor_sd),'-')
        pl.plot(crack_model_shear_factor_range,lognormal(crack_model_shear_factor_range,log_crack_model_shear_factor_estimate,log_crack_model_shear_factor_sd),'-')
        pl.xlabel('crack_model_shear_factor')
        pl.title('shearfact lognormal mu = %g; shearfact lognormal sd = %g' % (log_crack_model_shear_factor_estimate,log_crack_model_shear_factor_sd))
        pl.legend(('Prior','Posterior approx. (normal)','Posteror approx. (lognormal)','MCMC Histogram'),loc="best")
        pl.grid()


        #log_msqrtR_hist = pl.figure()
        #pl.clf()
        #pl.hist(np.log(msqrtR_vals),bins=marginal_bins,density=True)
        #log_msqrtR_range=np.linspace(0,pl.axis()[1],100)
        #pl.plot(log_msqrtR_range,np.exp(self.msqrtR_prior.logp(np.exp(log_msqrtR_range))).eval(),'-')
        #pl.plot(log_msqrtR_range,gaussian(log_msqrtR_range,self.log_msqrtR_estimate,self.log_msqrtR_sd),'-')
        #pl.xlabel('log(m*sqrtR)')
        #pl.legend(('Prior','Posterior approx.','MCMC Histogram'),loc="best")
        #pl.grid()

        sigma_additive_hist = pl.figure()
        pl.clf()
        pl.hist(sigma_additive_vals,bins=marginal_bins,density=True)
        sa_range=np.linspace(0,pl.axis()[1],100)
        pl.plot(sa_range,gaussian(sa_range,sigma_additive_prior_mu_unscaled,sigma_additive_prior_sigma_unscaled),'-')
        pl.plot(sa_range,gaussian(sa_range,sigma_additive_estimate,sigma_additive_sd),'-')
        pl.xlabel('sigma_additive (J/cy)')
        pl.legend(('Prior','Posterior approx.','MCMC Histogram'),loc="best")
        pl.title('sigma_additive normal mu = %g; sigma_additive normal sd = %g' % (sigma_additive_estimate,sigma_additive_sd))
        pl.grid()

        sigma_additive_power_hist = pl.figure()
        pl.clf()
        pl.hist(sigma_additive_vals*1e3*excfreq_median,bins=marginal_bins,density=True)
        sap_range=np.linspace(0,pl.axis()[1],100) # note: sap_range is in mW
        pl.plot(sap_range,gaussian(sap_range/1e3/excfreq_median,sigma_additive_prior_mu_unscaled,sigma_additive_prior_sigma_unscaled)/1.e3/excfreq_median,'-')
        pl.plot(sap_range,gaussian(sap_range/1e3/excfreq_median,sigma_additive_estimate,sigma_additive_sd)/1.e3/excfreq_median,'-')
        pl.xlabel('sigma_additive (mW) assuming typ freq of %f kHz' % (excfreq_median/1e3))
        pl.grid()


        sigma_multiplicative_hist = pl.figure()
        pl.clf()
        pl.hist(sigma_multiplicative_vals,bins=marginal_bins,density=True)
        sm_range=np.linspace(0,pl.axis()[1],100)
        pl.plot(sm_range,lognormal(sm_range,sigma_multiplicative_prior_mu,sigma_multiplicative_prior_sigma),'-')
        pl.plot(sm_range,gaussian(sm_range,sigma_multiplicative_estimate,sigma_multiplicative_sd),'-')
        pl.plot(sm_range,lognormal(sm_range,log_sigma_multiplicative_estimate,log_sigma_multiplicative_sd),'-')
        pl.xlabel('sigma_multiplicative')
        pl.title('sigma_mul lognormal mu = %g; sigma_mul lognormal sd = %g' % (log_sigma_multiplicative_estimate,log_sigma_multiplicative_sd))
        pl.legend(('Prior','Posterior approx. (normal)','Posteror approx. (lognormal)','MCMC Histogram'),loc="best")
        pl.grid()

        
        probit = lambda x: np.sqrt(2)*scipy.special.erfinv(2.0*x-1.0)

        sigma_multiplicative_pdfs = pl.figure()
        # heating is multiplied by multiplicative noise which is presumed to be lognormally distributed with a mu of ln(predicted heating) and variance of sigma_multiplicative_estimate
        # For this plot we represent it relative ot a predicted heating of 1.0 so ln(predicted heating) = 0 
        pl.clf()
        smp_range=np.linspace(0.001,4.0,200)
        smp_function = lambda x,sigma : (1.0/(x*sigma*np.sqrt(2.0*np.pi)))*np.exp(-((np.log(x))**2.0)/(2.0*sigma**2.0))
        pl.plot(smp_range,smp_function(smp_range,np.exp(log_sigma_multiplicative_estimate)),'-',
                smp_range,smp_function(smp_range,np.exp(log_sigma_multiplicative_estimate-probit(.975)*sigma_multiplicative_sd)),'--',
                smp_range,smp_function(smp_range,np.exp(log_sigma_multiplicative_estimate+probit(.975)*sigma_multiplicative_sd)),'--')
        #log_smp_range=np.linspace(-7,1.4,200)
        ## sigma_multiplicative is log-normally distributed so log_sigma_multiplicative should be normally distributed
        #log_smp_function = lambda x,mu,sigma : (1.0/(sigma*np.sqrt(2.0*np.pi)))*np.exp(-((x-mu)**2.0)/(2.0*sigma**2.0))
        #pl.plot(np.exp(log_smp_range),log_smp_function(log_smp_range,0,np.exp(log_sigma_multiplicative_estimate)),'-',
        #        np.exp(log_smp_range),log_smp_function(log_smp_range,0,np.exp(log_sigma_multiplicative_estimate-probit(.975)*log_sigma_multiplicative_sd)),'--',
        #np.exp(log_smp_range),log_smp_function(log_smp_range,0,np.exp(log_sigma_multiplicative_estimate+probit(.975)*log_sigma_multiplicative_sd)),'--')
        pl.xlabel('multiplier')
        pl.title('Probability density for multiplicative error') 
        pl.legend(('Based on best estimate (median)','97.5% lower bound for 95% conf interval','97.5% upper bound for 95% conf interval'))
        pl.grid()

        
    
        joint_hist = pl.figure()
        pl.clf()
        (hist,hist_mu_edges,hist_msqrtR_edges,hist_image)=pl.hist2d(mu_vals,msqrtR_vals,
                                                                    range=((mu_estimate-3.0*mu_sd,
                                                                            mu_estimate+3.0*mu_sd),
                                                                           (msqrtR_estimate-3.0*msqrtR_sd,
                                                                            msqrtR_estimate+3.0*msqrtR_sd)),
                                                                    bins=joint_bins)
        pl.grid()
        pl.colorbar()
        pl.xlabel('mu')
        pl.ylabel('m*sqrt(R) (sqrt(m)/m^2)')
        
        #histpeakpos = np.unravel_index(np.argmax(hist,axis=None),hist.shape)
        #self.mu_estimate = (hist_mu_edges[histpeakpos[0]]+hist_mu_edges[histpeakpos[0]+1])/2.0
        #self.msqrtR_estimate = (hist_msqrtR_edges[histpeakpos[1]]+hist_msqrtR_edges[histpeakpos[1]+1])/2.0
    
        # Compare
        predicted = self.predict_crackheating(mu_estimate,np.log(msqrtR_estimate),log_crack_model_shear_factor_estimate)*self.crackheat_table["ExcFreq (Hz)"].values
        
        # add to crackheat_table
        
        self.crackheat_table["predicted"]=predicted

        ## with:
        #self.actual = self.crackheat_table["ThermalPower (W)"].values

        # Group predicted and actual heating by specimen
        specimen_grouping = self.crackheat_table.groupby("Specimen")

        specimen_groups = [ specimen_grouping.get_group(specimen) for specimen in specimen_grouping.groups ]
        predicted_by_specimen = [ specimen_group["predicted"].values for specimen_group in specimen_groups ]
        actual_by_specimen = [ specimen_group["ThermalPower (W)"].values for specimen_group in specimen_groups ]
        specimen_group_specimens = list(specimen_grouping.groups) 

        markerstyle_cycler=cycler.cycler(marker=['o','v','^','<','>','s','p','+','x'])()
        prediction_plot = pl.figure()
        pl.clf()
        #pl.plot(predicted,self.actual,'x',
        #        (0,np.max(predicted)),(0,np.max(predicted)),'-')
        [ pl.plot(predicted_by_specimen[idx]*1e3,actual_by_specimen[idx]*1e3,linestyle='',**next(markerstyle_cycler)) for idx in range(len(specimen_group_specimens)) ] 
        pl.plot((0,np.max(predicted)*1.6*1e3),(0,np.max(predicted)*1.6*1e3),'-')
        pl.legend(specimen_group_specimens,loc='lower right')
        pl.xlabel('Predicted heating from model (mW)')
        pl.ylabel('Actual heating from experiment (mW)')
        pl.title('mu_estimate=%g; msqrtR_estimate=%g' % (mu_estimate,msqrtR_estimate))
        pl.grid()

        markerstyle_cyclerz=cycler.cycler(marker=['o','v','^','<','>','s','p','+','x'])()
        prediction_zoom_plot = pl.figure()
        pl.clf()
        #pl.plot(predicted,self.actual,'x',
        #        (0,np.max(predicted)),(0,np.max(predicted)),'-')
        [ pl.plot(predicted_by_specimen[idx]*1e3,actual_by_specimen[idx]*1e3,linestyle='',**next(markerstyle_cyclerz)) for idx in range(len(specimen_group_specimens)) ] 
        pl.plot((0,np.max(predicted)*1.6*1e3),(0,np.max(predicted)*1.6*1e3),'-')
        pl.legend(specimen_group_specimens,loc='lower right')
        pl.axis((0,np.max(predicted)*0.25*1e3,0,np.max(predicted)*0.2*1e3))
        pl.xlabel('Predicted heating from model (mW)')
        pl.ylabel('Actual heating from experiment (mW)')
        pl.title('mu_estimate=%g; msqrtR_estimate=%g' % (mu_estimate,msqrtR_estimate))
        pl.grid()


        plots = (traceplots,
                 mu_hist,
                 msqrtR_hist,
                 crack_model_shear_factor_hist,
                 sigma_additive_hist,
                 sigma_additive_power_hist,
                 sigma_multiplicative_hist,
                 sigma_multiplicative_pdfs,
                 joint_hist,
                 prediction_plot,
                 prediction_zoom_plot)
        
        # self.MixedNoiseOp(theano.shared(self.trace.get_values("sigma_additive")[0]),theano.shared(self.trace.get_values("sigma_multiplicative")[0]),theano.shared(self.trace.get_values("predicted_crackheating")[0,:])).eval() ... gives log(p) = 10
        # self.MixedNoiseOp.evaluate_p_from_cache(self.trace.get_values("sigma_additive")[0],self.trace.get_values("sigma_multiplicative")[0],self.trace.get_values("predicted_crackheating")[0,0],self.MixedNoiseOp.observed[0])
        
        #pdf_integral1 = scipy.integrate.quad(lambda obs: self.MixedNoiseOp.integrate_kernel(self.MixedNoiseOp.lognormal_normal_convolution_integral_y_zero_to_eps,self.MixedNoiseOp.lognormal_normal_convolution_kernel,self.trace.get_values("sigma_additive")[0],self.trace.get_values("sigma_multiplicative")[0],self.trace.get_values("predicted_crackheating")[0,1]*crackheat_scalefactor,obs),150.0,np.inf)[0]

        #pdf_integral2 = scipy.integrate.quad(lambda obs: self.MixedNoiseOp.integrate_kernel(self.MixedNoiseOp.lognormal_normal_convolution_integral_y_zero_to_eps,self.MixedNoiseOp.lognormal_normal_convolution_kernel,self.trace.get_values("sigma_additive")[0],self.trace.get_values("sigma_multiplicative")[0],self.trace.get_values("predicted_crackheating")[0,1]*crackheat_scalefactor,obs),.0000001,150.0)[0]
        
        #import theano.tests.unittest_tools
        #theano.config.compute_test_value = "off"

        #theano.tests.unittest_tools.verify_grad(self.MixedNoiseOp,[ self.trace.get_values("sigma_additive")[0], self.trace.get_values("sigma_multiplicative")[0],self.trace.get_values("predicted_crackheating")[0,:]*self.crackheat_scalefactor ],abs_tol=1e-12,rel_tol=1e-5,eps=1e-7)

        #raise ValueError("Foo!")

        #return (self.mu_estimate,self.msqrtR_estimate,traceplots,mu_hist,msqrtR_hist,joint_hist,prediction_plot)
        

        return (results,plots)




        

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
            #self.msqrtR = np.exp(logmsqrtR)
            self.log_msqrtR = logmsqrtR
            
            #
            # specify the predicted heating from the parameters Lambda (now extracted into mu and msqrtR)... Should be 1 column by N rows. 
            #
            # Create pymc3 predicted_crackheating expression
            self.predict_crackheating_per_specimen_op_instance = predict_crackheating_per_specimen_op(self)
            self.predicted_crackheating = self.predict_crackheating_per_specimen_op_instance(self.mu,self.log_msqrtR)


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
                    print("grad_mu_values = %s" % (str(self.predict_crackheating_per_specimen_grad_mu(np.array([0.3]*self.M), np.array([np.log(5e6)]*self.M)))))
                    print("grad_log_msqrtR_values = %s" % (str(self.predict_crackheating_per_specimen_grad_log_msqrtR(np.array([0.3]*self.M), np.array([np.log(5e6)]*self.M)))))

                    # Test gradient with respect to mu
                    theano.tests.unittest_tools.verify_grad(lambda mu_val: self.predict_crackheating_per_specimen_op_instance(mu_val, theano.shared(np.array([np.log(5e6)]*self.M))) ,[ np.array([0.3]*self.M),],abs_tol=1e-12,rel_tol=1e-5) # mu=0.3, msqrtR=5e6
                    # Test gradient with respect to msqrtR
                    theano.tests.unittest_tools.verify_grad(lambda log_msqrtR_val: self.predict_crackheating_per_specimen_op_instance(theano.shared(np.array([0.3]*self.M)),log_msqrtR_val) ,[ np.array([np.log(5e6)]*self.M)],abs_tol=1e-11,rel_tol=1e-3) # mu=0.3, msqrtR=5e6  NOTE: rel_tol is very tight here because Theano gradient.py/abs_rel_err() lower bounds the relative divisor to 1.e-8 and if we are not tight, we don't actually diagnose errors. 

                    print("\n\n\nVerify_grad() completed!!!\n\n\n")
                    pass
                finally:
                    theano.config.compute_test_value = orig_ctv
                    pass
                pass
            
            pass

        # set up for sampling
        
        with self.model:
        
            self.step = pm.Metropolis()
            #self.step=pm.NUTS(target_accept=0.80)
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

        markerstyle_cycler=cycler.cycler(marker=['o','v','^','<','>','s','p','+','x'])()  #'D'])()
        

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
        
        return (self.mu_estimate,self.msqrtR_estimate,Theta0_median,Theta1_median,packed_L_median,trace_frame,traceplots,theta_L_sigmaerror,lambdaplots,histograms,lambda_scatterplot,mu_msqrtR_scatterplot,mu_hist,msqrtR_hist,joint_hist,prediction_plot)
    
    pass
