import sys
import os
import os.path
import glob
import re
import numpy as np
import theano.tensor as tt
from theano import gof
#from theano import pp
import theano.tests.unittest_tools
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
        
        return [ (self.predict_crackheating_grad_mu_op(*inputs)*output_grads[0]).sum(), (self.predict_crackheating_grad_msqrtR_op(*inputs)*output_grads[0]).sum() ]   # I believe the sum()s here are because output_grads has a single element (corresponding to our single output) but that output is a vector, so output_grads[0] is a vector. We are supposed to return the tensor product dC/df * df_i/dx where dC/df is output_gradients[0], df_1/dx is predict_crackheating_grad_mu_op, and df_2/dx is predict_crackheating_grad_msqrtR_op. Since f_1 and f_2 are vectors, dC_df is also a vector and returning the tensor product means summing over the elements. 

    
    def infer_shape(self,node,input_shapes):
        return [ (self.estparam_obj.crackheat_table.shape[0],) ]

    

    def __init__(self,estparam_obj):
        self.itypes = [tt.dscalar,tt.dscalar]
        self.otypes = [tt.dvector]
        self.estparam_obj = estparam_obj
        
        self.predict_crackheating_grad_mu_op = as_op(itypes=[tt.dscalar,tt.dscalar], otypes = [tt.dvector])(self.estparam_obj.predict_crackheating_grad_mu)
        self.predict_crackheating_grad_msqrtR_op = as_op(itypes=[tt.dscalar,tt.dscalar], otypes = [tt.dvector])(self.estparam_obj.predict_crackheating_grad_msqrtR)

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
    crackheating=None

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
            
        
        pass
    

    def predict_crackheating_grad_mu(self,mu,msqrtR):
        """Predict derivative of crackheating with respect to mu for each row in self.crackheat_table,
        given hypothesized values for mu and msqrtR"""
        
        retval = np.zeros(self.crackheat_table.shape[0],dtype='d')
        
        # Could parallelize this loop!
        for index in range(self.crackheat_table.shape[0]):
            datagrid=np.array(((mu,self.crackheat_table["BendingStress (Pa)"].values[index],self.crackheat_table["DynamicStressAmpl (Pa)"].values[index],msqrtR),),dtype='d')
            retval[index]=self.crack_surrogates[self.crackheat_table["specimen_nums"].values[index]].evaluate_derivative(datagrid,0,accel_trisolve_devs=self.accel_trisolve_devs)[0]
            pass
        return retval


    def predict_crackheating_grad_msqrtR(self,mu,msqrtR):
        """Predict derivative of crackheating with respect to mu for each row in self.crackheat_table,
        given hypothesized values for mu and msqrtR"""
        
        retval = np.zeros(self.crackheat_table.shape[0],dtype='d')
        
        # Could parallelize this loop!
        for index in range(self.crackheat_table.shape[0]):
            datagrid=np.array(((mu,self.crackheat_table["BendingStress (Pa)"].values[index],self.crackheat_table["DynamicStressAmpl (Pa)"].values[index],msqrtR),),dtype='d')
            retval[index]=self.crack_surrogates[self.crackheat_table["specimen_nums"].values[index]].evaluate_derivative(datagrid,3,accel_trisolve_devs=self.accel_trisolve_devs)[0]
            pass
        return retval
    
    
    
    def predict_crackheating(self,mu,msqrtR):
        """Predict crackheating for each row in self.crackheat_table,
        given hypothesized values for mu and msqrtR"""
        
        retval = np.zeros(self.crackheat_table.shape[0],dtype='d')
        
        # Could parallelize this loop!
        for index in range(self.crackheat_table.shape[0]):
            datagrid=np.array(((mu,self.crackheat_table["BendingStress (Pa)"].values[index],self.crackheat_table["DynamicStressAmpl (Pa)"].values[index],msqrtR),),dtype='d')
            retval[index]=self.crack_surrogates[self.crackheat_table["specimen_nums"].values[index]].evaluate(datagrid,meanonly=True,accel_trisolve_devs=self.accel_trisolve_devs)["mean"][0]
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
            #self.dynamic_stress = pm.Normal('dynamic_stress',mu=20e6, sigma=5e6,observed=self.crackheat_table["DynamicStressAmpl (Pa)"].values)
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
    pass
