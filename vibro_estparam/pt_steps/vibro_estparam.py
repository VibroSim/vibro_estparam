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
        partial_pooling_bool=False,
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
    (mu_estimate,msqrtR_estimate,trace_frame,traceplots_fig,theta_L_sigmaerror_fig,lambdaplots,histograms,lambda_scatterplot_fig,mu_msqrtR_scatterplot_fig,mu_hist_fig,msqrtR_hist_fig,joint_hist_fig,prediction_plot_fig) = plot_and_estimate()

    trace_frame_href = hrefv("%s_trace_frame.csv" % (material_str.replace(" ","_")),_xmldoc.getcontexthref().leafless())
    trace_frame.to_csv(trace_frame_href.getpath())

    pl.figure(traceplots_fig.number)
    traceplots_href = hrefv("%s_traceplots.png" % (material_str.replace(" ","_")),_xmldoc.getcontexthref().leafless())
    pl.savefig(traceplots_href.getpath(),dpi=900)

    pl.figure(theta_L_sigmaerror_fig.number)
    theta_L_sigmaerror_href = hrefv("%s_theta_L_sigmaerror.png" % (material_str.replace(" ","_")),_xmldoc.getcontexthref().leafless())
    pl.savefig(theta_L_sigmaerror_href.getpath(),dpi=300)

    lambdaplot_hrefs=[]
    for lambdaidx in range(len(lambdaplots)):
        pl.figure(lambdaplots[lambdaidx].number)
        lambdaplot_href = hrefv("%s_lambda_%.2d.png" % (material_str.replace(" ","_"),lambdaidx),_xmldoc.getcontexthref().leafless())
        pl.savefig(lambdaplot_href.getpath(),dpi=300)
        lambdaplot_hrefs.append(lambdaplot_href)
        pass


    histogram_hrefs=[]
    histidx=0
    for histkey in histograms:
        pl.figure(histograms[histkey].number)
        histogram_href = hrefv("%s_histogram_%.2d.png" % (material_str.replace(" ","_"),histidx),_xmldoc.getcontexthref().leafless())
        pl.savefig(histogram_href.getpath(),dpi=300)
        histogram_hrefs.append(histogram_href)
        histidx+=1
        pass

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

    # !!!*** come up with better directory for ouput
    #pm.save_trace(estimator.trace,directory="pymc3_trace",overwrite=True)    

    ret = [
        (("dc:trace_frame", {"material": material_str}), trace_frame_href),
        (("dc:traceplots",{ "material": material_str}), traceplots_href),
        (("dc:theta_L_sigmaerror",{ "material": material_str}), theta_L_sigmaerror_href),
        (("dc:mu_estimate",{ "material": material_str}), numericunitsv(mu_estimate,"Unitless")),
        (("dc:msqrtR_estimate",{"material": material_str}), numericunitsv(msqrtR_estimate,"m^-1.5")),
        (("dc:mu_histogram",{"material": material_str}), mu_hist_href),
        (("dc:msqrtR_histogram",{"material": material_str}), msqrtR_hist_href),
        (("dc:joint_histogram",{"material": material_str}), joint_hist_href),
        (("dc:prediction_plot",{"material": material_str}), prediction_plot_href),
    ]
    ret.extend([ (("dc:lambdaplot",{"material": material_str, "lambdaidx" : str(lambdaidx)}), lambdaplot_hrefs[lambdaidx]) for lambdaidx in range(len(lambdaplot_hrefs))])

    ret.extend([ (("dc:histogram",{"material": material_str, "histidx" : str(histidx)}), histogram_hrefs[histidx]) for histidx in range(len(histogram_hrefs))])
    

    return ret
