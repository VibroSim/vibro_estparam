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


    mu_prior_mu_el = _xmldoc.xpathsinglecontext(_element,"dc:shear_mu_prior_mu[@material=%s]" % (string_to_etxpath_expression(material_str)))
    mu_prior_mu = numericunitsv.fromxml(_xmldoc,mu_prior_mu_el).value("Unitless")

    mu_prior_sigma_el = _xmldoc.xpathsinglecontext(_element,"dc:shear_mu_prior_sigma[@material=%s]" % (string_to_etxpath_expression(material_str)))
    mu_prior_sigma = numericunitsv.fromxml(_xmldoc,mu_prior_sigma_el).value("Unitless")

    msqrtR_prior_mu_el = _xmldoc.xpathsinglecontext(_element,"dc:shear_msqrtR_prior_mu[@material=%s]" % (string_to_etxpath_expression(material_str)))
    msqrtR_prior_mu = numericunitsv.fromxml(_xmldoc,msqrtR_prior_mu_el).value("ln_meters*-1.5")

    msqrtR_prior_sigma_el = _xmldoc.xpathsinglecontext(_element,"dc:shear_msqrtR_prior_sigma[@material=%s]" % (string_to_etxpath_expression(material_str)))
    msqrtR_prior_sigma = numericunitsv.fromxml(_xmldoc,msqrtR_prior_sigma_el).value("ln_meters*-1.5")

    sigma_additive_prior_mu_unscaled_el = _xmldoc.xpathsinglecontext(_element,"dc:shear_sigma_additive_prior_mu_unscaled[@material=%s]" % (string_to_etxpath_expression(material_str)))
    sigma_additive_prior_mu_unscaled = numericunitsv.fromxml(_xmldoc,sigma_additive_prior_mu_unscaled_el).value("W/Hz")

    sigma_additive_prior_sigma_unscaled_el = _xmldoc.xpathsinglecontext(_element,"dc:shear_sigma_additive_prior_sigma_unscaled[@material=%s]" % (string_to_etxpath_expression(material_str)))
    sigma_additive_prior_sigma_unscaled = numericunitsv.fromxml(_xmldoc,sigma_additive_prior_sigma_unscaled_el).value("W/Hz")

    sigma_multiplicative_prior_mu_el = _xmldoc.xpathsinglecontext(_element,"dc:shear_sigma_multiplicative_prior_mu[@material=%s]" % (string_to_etxpath_expression(material_str)))
    sigma_multiplicative_prior_mu = numericunitsv.fromxml(_xmldoc,sigma_multiplicative_prior_mu_el).value("Unitless")

    sigma_multiplicative_prior_sigma_el = _xmldoc.xpathsinglecontext(_element,"dc:shear_sigma_multiplicative_prior_sigma[@material=%s]" % (string_to_etxpath_expression(material_str)))
    sigma_multiplicative_prior_sigma = numericunitsv.fromxml(_xmldoc,sigma_multiplicative_prior_sigma_el).value("Unitless")
    

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
            surrogate_el = outputdoc.xpathsinglecontext(crack,"dc:shear_surrogate",default=None)
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

    estimator.load_data(filter_outside_closure_domain=filter_outside_closure_domain_bool,shear=True)


    
    (trace_df,
     crack_model_shear_factor_prior_mu,
     crack_model_shear_factor_prior_sigma,
     crackheat_scalefactor,
     excfreq_median,
     predicted_crackheating_lower_bound) = estimator.posterior_estimation_shear(mu_prior_mu,mu_prior_sigma,msqrtR_prior_mu,msqrtR_prior_sigma,sigma_additive_prior_mu_unscaled,sigma_additive_prior_sigma_unscaled,sigma_multiplicative_prior_mu,sigma_multiplicative_prior_sigma,steps_per_chain_int,num_chains_int,cores=cores_int,tune=tune_int)
    #posterior_estimation(10,4,cores=4,tune=20)
    
    
    trace_frame_href = hrefv("%s_shear_trace_frame.csv" % (material_str.replace(" ","_")),_xmldoc.getcontexthref().leafless())
    trace_df.to_csv(trace_frame_href.getpath())
    
    
    
    ret = [
        (("dc:shear_trace_frame", {"material": material_str}), trace_frame_href),
        (("dc:shear_crack_model_shear_factor_prior_mu",{"material": material_str}), numericunitsv(crack_model_shear_factor_prior_mu,"Unitless")),
        (("dc:shear_crack_model_shear_factor_prior_sigma",{"material": material_str}), numericunitsv(crack_model_shear_factor_prior_sigma,"Unitless")),
        (("dc:shear_crackheat_scalefactor",{"material": material_str}), numericunitsv(crackheat_scalefactor,"Unitless")),
        (("dc:shear_excfreq_median",{"material": material_str}), numericunitsv(excfreq_median,"Hz")),
        (("dc:shear_predicted_crackheating_lower_bound",{"material": material_str}), numericunitsv(predicted_crackheating_lower_bound,"W/Hz")),
        (("dc:shear_filter_outside_closure_domain",{"material": material_str}), filter_outside_closure_domain_bool),
    ]
    
    return ret
