import sys
import os
import os.path
import csv
import ast
import copy
import posixpath
import subprocess
import numpy as np

import pandas as pd

from limatix.dc_value import hrefvalue as hrefv
from limatix.dc_value import numericunitsvalue as numericunitsv
from limatix.dc_value import arrayvalue as arrayv
from limatix.xmldoc import xmldoc
from limatix.canonicalize_path import string_to_etxpath_expression


from vibro_estparam.estparam import estparam

import matplotlib

from matplotlib import pyplot as pl


def run(_xmldoc,_element,
        material_str):

    
    
    outputfiles = _xmldoc.xpathcontext(_element,"/prx:inputfiles/prx:inputfile/prx:outputfile")

    #context = cl.create_some_context()
    #accel_trisolve_devs=(os.getpid(),tuple([ (dev.platform.name,dev.name) for dev in context.devices]))
    accel_trisolve_devs=(os.getpid(), (('NVIDIA CUDA', 'Quadro GP100'),)) 
    #accel_trisolve_devs = (os.getpid(),(('Intel(R) OpenCL HD Graphics', 'Intel(R) Gen9 HD Graphics NEO'),))
    #accel_trisolve_devs = None


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


    trace_frame_el = _xmldoc.xpathsinglecontext(_element,"dc:shear_trace_frame[@material='%s']" % (material_str))
    trace_frame_href = hrefv.fromxml(_xmldoc,trace_frame_el)

    trace_df = pd.read_csv(trace_frame_href.getpath())

    mu_prior_mu = _xmldoc.xpathsinglecontextfloat(_element,"dc:shear_mu_prior_mu[@material='%s']"  % (material_str))
    mu_prior_sigma = _xmldoc.xpathsinglecontextfloat(_element,"dc:shear_mu_prior_sigma[@material='%s']" % (material_str))

    msqrtR_prior_mu_el = _xmldoc.xpathsinglecontext(_element,"dc:shear_msqrtR_prior_mu[@material='%s']" % (material_str))
    msqrtR_prior_sigma_el = _xmldoc.xpathsinglecontext(_element,"dc:shear_msqrtR_prior_sigma[@material='%s']" % (material_str))

    msqrtR_prior_mu = numericunitsv.fromxml(_xmldoc,msqrtR_prior_mu_el).value("ln_meters*-1.5")
    msqrtR_prior_sigma = numericunitsv.fromxml(_xmldoc,msqrtR_prior_sigma_el).value("ln_meters*-1.5")


    crack_model_shear_factor_prior_mu_el = _xmldoc.xpathsinglecontext(_element,"dc:shear_crack_model_shear_factor_prior_mu[@material='%s']" % (material_str))
    crack_model_shear_factor_prior_sigma_el = _xmldoc.xpathsinglecontext(_element,"dc:shear_crack_model_shear_factor_prior_sigma[@material='%s']" % (material_str))

    crack_model_shear_factor_prior_mu = numericunitsv.fromxml(_xmldoc,crack_model_shear_factor_prior_mu_el).value("Unitless")
    crack_model_shear_factor_prior_sigma = numericunitsv.fromxml(_xmldoc,crack_model_shear_factor_prior_sigma_el).value("Unitless")
    

    sigma_additive_prior_mu_unscaled_el = _xmldoc.xpathsinglecontext(_element,"dc:shear_sigma_additive_prior_mu_unscaled[@material='%s']" % (material_str))
    sigma_additive_prior_mu_unscaled = numericunitsv.fromxml(_xmldoc,sigma_additive_prior_mu_unscaled_el).value("W/Hz")

    sigma_additive_prior_sigma_unscaled_el = _xmldoc.xpathsinglecontext(_element,"dc:shear_sigma_additive_prior_sigma_unscaled[@material='%s']" % (material_str))
    sigma_additive_prior_sigma_unscaled = numericunitsv.fromxml(_xmldoc,sigma_additive_prior_sigma_unscaled_el).value("W/Hz")
    
    sigma_multiplicative_prior_mu = _xmldoc.xpathsinglecontextfloat(_element,"dc:shear_sigma_multiplicative_prior_mu[@material='%s']"  % (material_str))
    sigma_multiplicative_prior_sigma = _xmldoc.xpathsinglecontextfloat(_element,"dc:shear_sigma_multiplicative_prior_sigma[@material='%s']" % (material_str))
    
    crackheat_scalefactor = _xmldoc.xpathsinglecontextfloat(_element,"dc:shear_crackheat_scalefactor[@material='%s']" % (material_str))

    excfreq_median_el = _xmldoc.xpathsinglecontext(_element,"dc:shear_excfreq_median[@material='%s']" % (material_str))
    excfreq_median = numericunitsv.fromxml(_xmldoc,excfreq_median_el).value("Hz")
    
    predicted_crackheating_lower_bound_el = _xmldoc.xpathsinglecontext(_element,"dc:shear_predicted_crackheating_lower_bound[@material='%s']" % (material_str))
    predicted_crackheating_lower_bound = numericunitsv.fromxml(_xmldoc,predicted_crackheating_lower_bound_el).value("W/Hz")
    
    filter_outside_closure_domain = bool(ast.literal_eval(_xmldoc.xpathsinglecontextstr(_element,"dc:shear_filter_outside_closure_domain[@material='%s']" % (material_str))))

    #estimator = estparam.fromfilelists(crack_specimens,crackheatfiles,surrogatefiles,accel_trisolve_devs)
    estimator = estparam.fromfilelists(crack_specimens,crackheatfiles,surrogatefiles,accel_trisolve_devs)

    estimator.load_data(filter_outside_closure_domain=filter_outside_closure_domain,shear=True)
    
    (results,plots) = estimator.plot_and_estimate_shear(trace_df,
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
                                                        predicted_crackheating_lower_bound)
    

    #if partial_pooling_bool:        
    #    posterior_estimation=estimator.posterior_estimation_partial_pooling
    #    plot_and_estimate=estimator.plot_and_estimate_partial_pooling
    #    pass
    #else:
    #    posterior_estimation=estimator.posterior_estimation
    #    plot_and_estimate=estimator.plot_and_estimate
    #    pass

    (mu_estimate,
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
     log_sigma_multiplicative_sd) = results
        
    (traceplots,
     mu_hist,
     msqrtR_hist,
     crack_model_shear_factor_hist,
     sigma_additive_hist,
     sigma_additive_power_hist,
     sigma_multiplicative_hist,
     sigma_multiplicative_pdfs,
     joint_hist,
     prediction_plot,
     prediction_zoom_plot) = plots

    pl.figure(traceplots.number)
    traceplots_href = hrefv("shear_%s_traceplots.png" % (material_str.replace(" ","_")),_xmldoc.getcontexthref().leafless())
    pl.savefig(traceplots_href.getpath(),dpi=900)

    pl.figure(mu_hist.number)
    mu_hist_href = hrefv("shear_%s_mu_hist.png" % (material_str.replace(" ","_")),_xmldoc.getcontexthref().leafless())
    pl.savefig(mu_hist_href.getpath(),dpi=300)

    pl.figure(msqrtR_hist.number)
    msqrtR_hist_href = hrefv("shear_%s_msqrtR_hist.png" % (material_str.replace(" ","_")),_xmldoc.getcontexthref().leafless())
    pl.savefig(msqrtR_hist_href.getpath(),dpi=300)

    pl.figure(crack_model_shear_factor_hist.number)
    crack_model_shear_factor_hist_href = hrefv("shear_%s_crack_model_shear_factor_hist.png" % (material_str.replace(" ","_")),_xmldoc.getcontexthref().leafless())
    pl.savefig(crack_model_shear_factor_hist_href.getpath(),dpi=300)
    
    pl.figure(sigma_additive_hist.number)
    sigma_additive_hist_href = hrefv("shear_%s_sigma_additive_hist.png" % (material_str.replace(" ","_")),_xmldoc.getcontexthref().leafless())
    pl.savefig(sigma_additive_hist_href.getpath(),dpi=300)

    pl.figure(sigma_additive_power_hist.number)
    sigma_additive_power_hist_href = hrefv("shear_%s_sigma_additive_power_hist.png" % (material_str.replace(" ","_")),_xmldoc.getcontexthref().leafless())
    pl.savefig(sigma_additive_power_hist_href.getpath(),dpi=300)


    pl.figure(sigma_multiplicative_hist.number)
    sigma_multiplicative_hist_href = hrefv("shear_%s_sigma_multiplicative_hist.png" % (material_str.replace(" ","_")),_xmldoc.getcontexthref().leafless())
    pl.savefig(sigma_multiplicative_hist_href.getpath(),dpi=300)

    pl.figure(sigma_multiplicative_pdfs.number)
    sigma_multiplicative_pdfs_href = hrefv("shear_%s_sigma_multiplicative_pdfs.png" % (material_str.replace(" ","_")),_xmldoc.getcontexthref().leafless())
    pl.savefig(sigma_multiplicative_pdfs_href.getpath(),dpi=300)

    
    pl.figure(joint_hist.number)
    joint_hist_href = hrefv("shear_%s_joint_hist.png" % (material_str.replace(" ","_")),_xmldoc.getcontexthref().leafless())
    pl.savefig(joint_hist_href.getpath(),dpi=300)
    
    
    pl.figure(prediction_plot.number)
    prediction_plot_href = hrefv("shear_%s_prediction_plot.png" % (material_str.replace(" ","_")),_xmldoc.getcontexthref().leafless())
    pl.savefig(prediction_plot_href.getpath(),dpi=300)

    pl.figure(prediction_zoom_plot.number)
    prediction_zoom_plot_href = hrefv("shear_%s_prediction_zoom_plot.png" % (material_str.replace(" ","_")),_xmldoc.getcontexthref().leafless())
    pl.savefig(prediction_zoom_plot_href.getpath(),dpi=300)
    

    ret = [
        (("dc:shear_mu_estimate",{ "material": material_str}), numericunitsv(mu_estimate,"Unitless")),
        (("dc:shear_mu_sd",{ "material": material_str}), numericunitsv(mu_sd,"Unitless")),
        (("dc:shear_log_mu_estimate",{ "material": material_str}), numericunitsv(log_mu_estimate,"Unitless")),
        (("dc:shear_log_mu_sd",{ "material": material_str}), numericunitsv(log_mu_sd,"Unitless")),
        (("dc:shear_msqrtR_estimate",{"material": material_str}), numericunitsv(msqrtR_estimate,"m^-1.5")),
        (("dc:shear_msqrtR_sd",{"material": material_str}), numericunitsv(msqrtR_sd,"m^-1.5")),
        (("dc:shear_log_msqrtR_estimate",{"material": material_str}), numericunitsv(log_msqrtR_estimate,"ln_meters*-1.5")),
        (("dc:shear_log_msqrtR_sd",{"material": material_str}), numericunitsv(log_msqrtR_sd,"ln_meters*-1.5")),
        (("dc:shear_crack_model_shear_factor_estimate",{"material": material_str}), numericunitsv(crack_model_shear_factor_estimate,"Unitless")),
        (("dc:shear_crack_model_shear_factor_sd",{"material": material_str}), numericunitsv(crack_model_shear_factor_sd,"Unitless")),
        (("dc:shear_log_crack_model_shear_factor_estimate",{"material": material_str}), numericunitsv(log_crack_model_shear_factor_estimate,"Unitless")),
        (("dc:shear_log_crack_model_shear_factor_sd",{"material": material_str}), numericunitsv(log_crack_model_shear_factor_sd,"Unitless")),
        (("dc:shear_sigma_additive_estimate",{ "material": material_str}), numericunitsv(sigma_additive_estimate,"W/Hz")),
        (("dc:shear_sigma_additive_sd",{ "material": material_str}), numericunitsv(sigma_additive_sd,"W/Hz")),
        (("dc:shear_sigma_multiplicative_estimate",{ "material": material_str}), numericunitsv(sigma_multiplicative_estimate,"Unitless")),
        (("dc:shear_sigma_multiplicative_sd",{ "material": material_str}), numericunitsv(sigma_multiplicative_sd,"Unitless")),
        (("dc:shear_log_sigma_multiplicative_estimate",{ "material": material_str}), numericunitsv(log_sigma_multiplicative_estimate,"Unitless")),
        (("dc:shear_log_sigma_multiplicative_sd",{ "material": material_str}), numericunitsv(log_sigma_multiplicative_sd,"Unitless")),
        (("dc:shear_traceplots",{ "material": material_str}), traceplots_href),
        (("dc:shear_mu_histogram",{ "material": material_str}), mu_hist_href),
        (("dc:shear_msqrtR_histogram",{ "material": material_str}), msqrtR_hist_href),
        (("dc:shear_crack_model_shear_factor_histogram",{ "material": material_str}), crack_model_shear_factor_hist_href),
        (("dc:shear_sigma_additive_histogram",{ "material": material_str}), sigma_additive_hist_href),
        (("dc:shear_sigma_additive_power_histogram",{ "material": material_str}), sigma_additive_power_hist_href),
        (("dc:shear_sigma_multiplicative_histogram",{ "material": material_str}), sigma_multiplicative_hist_href),
        (("dc:shear_sigma_multiplicative_pdfs",{ "material": material_str}), sigma_multiplicative_pdfs_href),
        (("dc:shear_joint_histogram",{"material": material_str}), joint_hist_href),
        (("dc:shear_prediction_plot",{"material": material_str}), prediction_plot_href),
        (("dc:shear_prediction_zoom_plot",{"material": material_str}), prediction_zoom_plot_href),
    ]    

    return ret
