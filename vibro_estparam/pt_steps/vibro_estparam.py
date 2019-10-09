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
from limatix.xmldoc import xmldoc


from vibro_estparam.estparam import estparam

import matplotlib

from matplotlib import pyplot as pl


def run(_xmldoc,_element,
        material_str,
        steps_per_chain_int = 500,
        num_chains_int=4,
        cores_int=4,
        tune_int=250,
        partial_pooling_bool=False
        filter_outside_closure_domain_bool=False):
    
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
        cracks = outputdoc.xpath("dc:crack")
        for crack in cracks: 
            specimen=outputdoc.xpathsinglecontextstr(crack,"dc:specimen",default="UNKNOWN")
            material = outputdoc.xpathsinglecontextstr(crack,"dc:spcmaterial",default="UNKNOWN")
            if material == material_str:
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
    
    posterior_estimation(steps_per_chain_int,num_chains_int,cores=cores_int,tune=tune_int)
    #posterior_estimation(10,4,cores=4,tune=20)
    (mu_estimate,msqrtR_estimate,traceplots_fig,mu_hist_fig,msqrtR_hist_fig,joint_hist_fig,prediction_plot_fig) = plot_and_estimate()

    pl.figure(traceplots_fig.number)
    traceplots_href = hrefv("%s_traceplots.png" % (material_str.replace(" ","_")),_xmldoc.getcontexthref().leafless())
    pl.savefig(traceplots_href.getpath(),dpi=300)

    pl.figure(mu_hist_fig.number)
    mu_hist_href = hrefv("%s_mu_histogram.png" % (material_str.replace(" ","_")),_xmldoc.getcontexthref().leafless())
    pl.savefig(mu_hist_href.getpath(),dpi=300)

    pl.figure(msqrtR_hist_fig.number)
    msqrtR_hist_href = hrefv("%s_msqrtR_histogram.png" % (material_str.replace(" ","_")),_xmldoc.getcontexthref().leafless())
    pl.savefig(msqrtR_hist_href.getpath(),dpi=300)

    pl.figure(joint_hist_fig.number)
    joint_hist_href = hrefv("%s_joint_histogram.png" % (material_str.replace(" ","_")),_xmldoc.getcontexthref().leafless())
    pl.savefig(joint_hist_href.getpath(),dpi=300)

    pl.figure(prediction_plot_fig.number)
    prediction_plot_href = hrefv("%s_prediction_plot.png" % (material_str.replace(" ","_")),_xmldoc.getcontexthref().leafless())
    pl.savefig(prediction_plot_href.getpath(),dpi=300)
    
    return (
        (("dc:traceplots",{ "material": material_str}), traceplots_href),
        (("dc:mu_estimate",{ "material": material_str}), numericunitsv(mu_estimate,"Unitless")),
        (("dc:msqrtR_estimate",{"material": material_str}), numericunitsv(msqrtR_estimate,"m^-1.5")),
        (("dc:mu_histogram",{"material": material_str}), mu_hist_href),
        (("dc:msqrtR_histogram",{"material": material_str}), msqrtR_hist_href),
        (("dc:joint_histogram",{"material": material_str}), joint_hist_href),
        (("dc:prediction_plot",{"material": material_str}), prediction_plot_href),
    )

    
