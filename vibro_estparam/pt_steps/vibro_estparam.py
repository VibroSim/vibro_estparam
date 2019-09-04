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

from crackheat_surrogate import get_rscripts_path

def run(_xmldoc,_element,
        material_str,
        filter_outside_closure_domain_bool=False):
    
    outputfiles = _xmldoc.xpathcontext(_element,"/prx:inputfiles/prx:inputfile/prx:outputfile")

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


    crack_specimens = [ os.path.split(filename)[1][:(-len(surrogateext))] for filename in surrogatefiles ]  # we number cracks according to their index in this list

    #estimator = estparam.fromfilelists(crack_specimens,crackheatfiles,surrogatefiles,accel_trisolve_devs)
    estimator = scriptify(estparam.fromfilelists)(crack_specimens,crackheatfiles,surrogatefiles,accel_trisolve_devs)

    estimator.load_data(filter_outside_closure_domain=filter_outside_closure_domain_bool)
    
    estimator.posterior_estimation(1000,4,cores=4)
    (mu_estimate,msqrtR_estimate,mu_hist_fig,msqrtR_hist_fig,joint_hist_fig,prediction_plot_fig) = estimator.plot_and_estimate()

    pl.figure(mu_hist_fig.fignum)
    mu_hist_href = hrefv("%s_mu_histogram.png" % (material_str.replace(" ","_")),_xmldoc.getcontexthref().leafless())
    pl.savefig(mu_hist_href.getpath(),dpi=300)

    pl.figure(msqrtR_hist_fig.fignum)
    msqrtR_hist_href = hrefv("%s_msqrtR_histogram.png" % (material_str.replace(" ","_")),_xmldoc.getcontexthref().leafless())
    pl.savefig(msqrtR_hist_href.getpath(),dpi=300)

    pl.figure(joint_hist_fig.fignum)
    joint_hist_href = hrefv("%s_joint_histogram.png" % (material_str.replace(" ","_")),_xmldoc.getcontexthref().leafless())
    pl.savefig(joint_hist_href.getpath(),dpi=300)

    pl.figure(prediction_plot_fig.fignum)
    prediction_plot_href = hrefv("%s_prediction_plot.png" % (material_str.replace(" ","_")),_xmldoc.getcontexthref().leafless())
    pl.savefig(prediction_plot_href.getpath(),dpi=300)
    
    return {
        "dc:mu_estimate": numericunitsv(mu_estimate,"Unitless"),
        "dc:msqrtR_estimate": numericunitsv(msqrtR_estimate,"m^(-3/2)"),
        "dc:mu_histogram": mu_hist_href,
        "dc:msqrtR_histogram": msqrtR_hist_href,
        "dc:joint_histogram": joint_hist_href,
        "dc:prediction_plot": prediction_plot_href,
    }

    
