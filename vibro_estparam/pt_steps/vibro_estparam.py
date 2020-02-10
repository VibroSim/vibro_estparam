import sys
import os
import os.path
import csv
import ast
import copy
import posixpath
import subprocess
import numpy as np

from limatix.dc_value import hrefvalue as hrefv
from limatix.dc_value import numericunitsvalue as numericunitsv
from limatix.dc_value import arrayvalue as arrayv
from limatix.xmldoc import xmldoc
from limatix.canonicalize_path import string_to_etxpath_expression


from vibro_estparam.estparam import estparam

import matplotlib

from matplotlib import pyplot as pl


def run(_xmldoc,_element,
        material_str,
        steps_per_chain_int = 500,
        num_chains_int=4,
        cores_int=4,
        tune_int=250,
        partial_pooling_bool=False,
        filter_outside_closure_domain_bool=False):
    
    outputfiles = _xmldoc.xpathcontext(_element,"/prx:inputfiles/prx:inputfile/prx:outputfile")

    #context = cl.create_some_context()
    #accel_trisolve_devs=(os.getpid(),tuple([ (dev.platform.name,dev.name) for dev in context.devices]))
    #accel_trisolve_devs=(os.getpid(), (('NVIDIA CUDA', 'Quadro GP100'),)) 
    #accel_trisolve_devs = (os.getpid(),(('Intel(R) OpenCL HD Graphics', 'Intel(R) Gen9 HD Graphics NEO'),))
    accel_trisolve_devs = None


    crack_specimens = []
    crackheatfiles = []
    surrogatefiles = []

    for outputfile in outputfiles:
        outputdoc = xmldoc.loadhref(hrefv.fromxml(_xmldoc,outputfile))
        cracks = outputdoc.xpath("dc:crack[count(@dc:ignore) < 1 and dc:spcmaterial=%s]" % (string_to_etxpath_expression(material_str)))
        for crack in cracks: 
            specimen=outputdoc.xpathsinglecontextstr(crack,"dc:specimen",default="UNKNOWN")
            material = outputdoc.xpathsinglecontextstr(crack,"dc:spcmaterial",default="UNKNOWN")
            assert(material == material_str)
            crackheat_table_el = outputdoc.xpathsinglecontext(crack,"dc:crackheat_table",default=None)
            surrogate_el = outputdoc.xpathsinglecontext(crack,"dc:surrogate",default=None)
            if crackheat_table_el is not None and surrogate_el is not None:
                crackheat_table_href = hrefv.fromxml(outputdoc,crackheat_table_el)
                surrogate_href = hrefv.fromxml(outputdoc,surrogate_el)
                
                crackheatfiles.append(crackheat_table_href.getpath())
                surrogatefiles.append(surrogate_href.getpath())
                crack_specimens.append(specimen)
                pass
            if crackheat_table_el is None:
                print("WARNING: No crack heating table found for specimen %s!" % (specimen))
                pass
            if surrogate_el is None:
                print("WARNING: No surrogate found for specimen %s!" % (specimen))
                pass
            
            pass
        pass



    #estimator = estparam.fromfilelists(crack_specimens,crackheatfiles,surrogatefiles,accel_trisolve_devs)
    estimator = estparam.fromfilelists(crack_specimens,crackheatfiles,surrogatefiles,accel_trisolve_devs)

    estimator.load_data(filter_outside_closure_domain=filter_outside_closure_domain_bool)


    if partial_pooling_bool:        
        posterior_estimation=estimator.posterior_estimation_partial_pooling
        plot_and_estimate=estimator.plot_and_estimate_partial_pooling
        pass
    else:
        posterior_estimation=estimator.posterior_estimation
        plot_and_estimate=estimator.plot_and_estimate
        pass
    
    (trace_df,
     mu_prior_mu,
     mu_prior_sigma,
     msqrtR_prior_mu,
     msqrtR_prior_sigma,
     sigma_additive_prior_sigma_unscaled,
     sigma_multiplicative_prior_mu,
     sigma_multiplicative_prior_sigma,
     crackheat_scalefactor,
     excfreq_median,
     predicted_crackheating_lower_bound) = posterior_estimation(steps_per_chain_int,num_chains_int,cores=cores_int,tune=tune_int)
    #posterior_estimation(10,4,cores=4,tune=20)
    
    
    trace_frame_href = hrefv("%s_trace_frame.csv" % (material_str.replace(" ","_")),_xmldoc.getcontexthref().leafless())
    trace_df.to_csv(trace_frame_href.getpath())
    
    
    
    ret = [
        (("dc:trace_frame", {"material": material_str}), trace_frame_href),
        (("dc:mu_prior_mu", {"material": material_str}), numericunitsv(mu_prior_mu,"Unitless")),
        (("dc:mu_prior_sigma", {"material": material_str}), numericunitsv(mu_prior_sigma,"Unitless")),
        #(("dc:msqrtR_prior_mu",{"material": material_str}), numericunitsv(msqrtR_prior_mu,"m^-1.5")),
        (("dc:msqrtR_prior_mu",{"material": material_str}), numericunitsv(msqrtR_prior_mu,"ln_meters*-1.5")),
        (("dc:msqrtR_prior_sigma",{"material": material_str}), numericunitsv(msqrtR_prior_sigma,"ln_meters*-1.5")),
        (("dc:sigma_additive_prior_sigma_unscaled",{"material": material_str}), numericunitsv(sigma_additive_prior_sigma_unscaled,"W/Hz")),
        (("dc:sigma_multiplicative_prior_mu",{"material": material_str}), numericunitsv(sigma_multiplicative_prior_mu,"Unitless")),
        (("dc:sigma_multiplicative_prior_sigma",{"material": material_str}), numericunitsv(sigma_multiplicative_prior_sigma,"Unitless")),
        (("dc:crackheat_scalefactor",{"material": material_str}), numericunitsv(crackheat_scalefactor,"Unitless")),
        (("dc:excfreq_median",{"material": material_str}), numericunitsv(excfreq_median,"Hz")),
        (("dc:predicted_crackheating_lower_bound",{"material": material_str}), numericunitsv(predicted_crackheating_lower_bound,"W/Hz")),
        (("dc:filter_outside_closure_domain",{"material": material_str}), filter_outside_closure_domain_bool),
    ]
    
    return ret
