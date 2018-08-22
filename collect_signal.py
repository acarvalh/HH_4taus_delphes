#!/usr/bin/env python
import os, sys, time,math
import os, subprocess, sys
import ROOT
import fileinput
import glob
import pandas
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import numpy as np
from root_numpy import root2array, rec2array, array2root, tree2array
from collections import OrderedDict
from optparse import OptionParser
execfile("functions.py") # read_channel

parser = OptionParser()
parser.add_option("--input ", type="string", dest="input", help="A valid file in Delphes format. If you give somthing without .root extension it will interpret as a folder and try to process all the root files on that folder in series", default="/eos/cms/store/user/acarvalh/delphes_HH_4tau/")
parser.add_option("--onlyCount", action="store_true", dest="onlyCount", help="Only reports the number of events on the sample", default=False)
(options, args) = parser.parse_args()

#if "HHTo4T" in options.input : channel = "HHTo4T"
#if "HHTo4V" in options.input : channel = "HHTo4V"
#if "HHTo2T2V" in options.input : channel = "HHTo2T2V"

channels = ["HHTo4T" , "HHTo4V", "HHTo2T2V"]
## read ALLCount
totalEvt = []
for cc, channel in enumerate(channels) :
    countRead = options.input+"/Folder_"+channel+"/AllCount_"+channel+".txt"
    file = open(countRead,"r")
    for line in file : totalEvt +=  [int(line.split(" ")[1])]
    print ("Number of processed events on", countRead, "for channel",channel, "is", totalEvt[cc])

Categories=[
    'tree_lge4_0tauh',
    'tree_3l_1tauh',
    'tree_2l_2tauh',
    'tree_1l_3tauh',
    'tree_1l_0tauh',
    'tree_0l_0tauh',
    'tree_0l_4tauh',          #### just added
    'tree_0l_3tauh',          #### just added
    'tree_0l_5getauh',        #### just added
    'tree_2los_0tauh',
    'tree_2lss_0tauh',
    'tree_3l_0tauh',
    'tree_2l_1tauh',
    'tree_1l_2tauh',
    'tree_5l_1tauh',
    'tree_4l_2tauh',
    'tree_0l_2tauh',
    'tree_1l_1tauh',       ### missed ones
    'tree_0l_1tauh',       ### missed ones
    'tree_3l_2tauh',       ### missed ones
    'tree_1l_4tauh',       ### missed ones
    'tree_2l_3tauh',       ### missed ones
    'tree_4l_3tauh',       ### missed ones
    'tree_4l_1tauh',       ### missed ones
    'tree_3l_3tauh',       ### missed ones
    'tree_2l_4tauh',       ### missed ones
    'tree_1l_5tauh',       ### missed ones
    #'tree_total'
    ]
print("number of categories___",len(Categories))


#branches_double_names = ["MissingET", "MissingET_eta", "MissingET_phi", "scalarHT"]
#branches_int_names = ["evtcounter", "njets", "nBs" , "nelectrons",  "nmuons", "ntaus", "sum_lep_charge", "sum_tauh_charge"]
### doing less plots
branches_double_names = ["MissingET", "scalarHT"]
branches_int_names = [ "njets", "sum_lep_charge", "sum_tauh_charge"]
featuresToPlot = branches_int_names+branches_double_names
### the bellow is not possible to read on pandas -- it is easier to implement variables on the analysis code
## #branches_double_arrays_names = ["pt", "eta", "phi", "mass", "charge"]

#####################################
### Options to histograms with matplotlib
color1='g'
printmin=True
####################################


## I am converting the tree to pandas (python format), it is an eazy format to manipulate and do plots
dict_data_4T = read_channel(Categories, channels[0], totalEvt[0])
dict_data_4V = read_channel(Categories, channels[1], totalEvt[1])
dict_data_2T2V = read_channel(Categories, channels[2], totalEvt[2])
colorCat = ['g','r','b']

base_folder = os.getcwd()+'/Folder_plots'
if not os.path.exists(base_folder) : os.mkdir( base_folder )
for category in Categories :
        plotname = base_folder+"/plots_"+category+".pdf"
        hist_params = {'histtype': 'bar', 'fill': True , 'lw':3}
        sizeArray=int(math.sqrt(len(featuresToPlot))) if math.sqrt(len(featuresToPlot)) % int(math.sqrt(len(featuresToPlot))) == 0 else int(math.sqrt(len(featuresToPlot)))+1
        drawStatErr=True
        plt.figure(figsize=(4*sizeArray, 4*sizeArray))
        for n, feature in enumerate(featuresToPlot):
            # add sub plot on our figure
            plt.subplot(sizeArray, sizeArray, n+1)
            for dd, dict_data in enumerate([dict_data_4T, dict_data_4V, dict_data_2T2V]) :
                minArray = []
                maxArray = []
                if len(dict_data[category][0]["evtcounter"]) > 0 :
                    if feature in branches_int_names :
                        min_value, max_value = np.percentile(dict_data[category][0][feature], [0.0, 100])
                        if feature not in ["evtcounter"] : # "sum_lep_charge", "sum_tauh_charge"
                            min_value=min(min_value, 0.0)-0.5
                            max_value=max_value+0.5
                    else : min_value, max_value = np.percentile(dict_data[category][0][feature], [0.0, 99])
                    minArray += [min_value]
                    maxArray += [max_value]
                else :
                    print ("category ",category,"of channel", channels[dd], "had no events")
                    minArray += [0]
                    maxArray += [0]
            maxY = []
            for dd, dict_data in enumerate([dict_data_4T, dict_data_4V, dict_data_2T2V]) :
                if feature in branches_int_names :
                    if abs(dict_data[category][0][feature].max()) > 0 :
                        if not int(dict_data[category][0][feature].min()) == 0 :
                            nbin = abs(int(dict_data[category][0][feature].max()))+abs(int(dict_data[category][0][feature].min()))+1
                        else : nbin = abs(int(dict_data[category][0][feature].max()))+1
                    else : nbin = 1
                    #if feature not in ["evtcounter"] :
                else : nbin = 10
                # define range for histograms by cutting 1% of data from both ends
                min_value = min(minArray)
                max_value = max(maxArray)
                if max_value == 0 or max_value == -0.5 : max_value = min_value + 1
                ## to avoid the plotter to crash if there are no events
                if printmin and dd == 0  : print (feature, "min/max value and number of bins:", min_value, max_value, nbin)
                values1, bins, _ = plt.hist(dict_data[category][0][feature].values,
                       range=(min_value,  max_value),
                       bins=nbin, edgecolor=colorCat[dd], color=colorCat[dd], alpha = 0.4,
                       label=channels[dd], normed=True, **hist_params )
                if drawStatErr:
                    normed = sum(dict_data[category][0][feature].values)
                    mid = 0.5*(bins[1:] + bins[:-1])
                    err=np.sqrt(values1*normed)/normed # denominator is because plot is normalized
                    plt.errorbar(mid, values1, yerr=err, fmt='none', color= colorCat[dd], ecolor= colorCat[dd], edgecolor=colorCat[dd], lw=2)
                #areaSig = sum(np.diff(bins)*values)
                #print areaSig
                if n == len(featuresToPlot)-1 : plt.legend(loc='best')
                plt.xlabel(feature)
                maxY += [values1.max()]
                #plt.xscale('log')
                #plt.yscale('log')
        plt.ylim(ymin=0, ymax=max(maxY)*1.3)
        plt.savefig(plotname)
        plt.close()
        print ("saved", plotname)
