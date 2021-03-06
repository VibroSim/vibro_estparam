import sys
import os
import os.path
import glob
import re
import numpy as np
import theano.tensor as tt
import pymc3 as pm
import pandas as pd
from matplotlib import pyplot as pl
from theano.compile.ops import as_op

from crackheat_surrogate2.load_surrogate import load_denorm_surrogates_from_jsonfile


from vibro_estparam.estparam import estparam

from function_as_script import scriptify

if __name__=="__main__":
    datapath='/tmp/data2/'


    np.random.seed(14264602)

    surrogateext = "_surrogate.json"
    crackheatfiles = glob.glob(os.path.join(datapath,"*_crackheat_table.csv"))
    surrogatefiles = glob.glob(os.path.join(datapath,"*"+surrogateext))


    #import pyopencl as cl
    
    #context = cl.create_some_context()
    #accel_trisolve_devs=(os.getpid(),tuple([ (dev.platform.name,dev.name) for dev in context.devices]))
    accel_trisolve_devs=(os.getpid(), (('NVIDIA CUDA', 'Quadro GP100'),)) 
    #accel_trisolve_devs = (os.getpid(),(('Intel(R) OpenCL HD Graphics', 'Intel(R) Gen9 HD Graphics NEO'),))
    #accel_trisolve_devs = None


    crack_specimens = [ os.path.split(filename)[1][:(-len(surrogateext))] for filename in surrogatefiles ]  # we number cracks according to their index in this list

    #estimator = estparam.fromfilelists(crack_specimens,crackheatfiles,surrogatefiles,accel_trisolve_devs)
    estimator = scriptify(estparam.fromfilelists)(crack_specimens,crackheatfiles,surrogatefiles,accel_trisolve_devs)

    scriptify(estimator.load_data)(filter_outside_closure_domain=True)
    
    #estimator.posterior_estimation(1000,4,cores=4)
    scriptify(estimator.posterior_estimation_partial_pooling)(500,4,cores=4,tune=50)
    scriptify(estimator.plot_and_estimate_partial_pooling)()
    pl.show()
    pass
