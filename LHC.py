# For now run this ncar python env in the command line, not sure how to execute within script:
#source /glade/p/work/kdagon/ncar_pylib_clone/bin/activate
# this doesn't work:
#import subprocess
#subprocess.call(["source","/glade/p/work/kdagon/ncar_pylib_clone/bin/activate"])

import sys
import os

import json

import math
import numpy

# module for lhs function
from pyDOE import *
#help(lhs)



def gen_params():
    """
    Generates sets of values for specified parameters.

    :return: numpy.ndarray
    """
    # select parameters
    params = ['medlynslope','dleaf','kmax','fff','dint','baseflow_scalar']

    n = len(params) # number of parameters
    #samp = 100 # sample size per parameter
    samp = 10 # start small to test

    # construct the latin hypercube
    lhd = lhs(n, samples=samp) # default sampling criterion = random

    # specify PFTs for relevant parameters
    #medlynslope_PFTs = ['NET/NDT','BET','BDT','SHR','C3ag','C3g','C4','C3c']
    #dleaf_PFTs = ['NET','NDT','BET','BDT','BES','BDSt','BDSb','grass','crop']

    # specify min/max values for PFT-dependent parameters
    medlynslope_mins = numpy.array([1.29,1.63,3.19,2.25,2.00,3.05,0.53,3.46])
    medlynslope_maxs = numpy.array([4.70,4.59,5.11,9.27,2.44,9.45,4.03,7.70])

    dleaf_mins = numpy.array([0.000216,0.00072,0.0081,0.0081,0.0081,0.000405,0.000162,0.000144,0.000162])
    dleaf_maxs = numpy.array([0.00108,0.0036,0.0567,0.243,0.081,0.1215,0.0486,0.018,0.1215])

    # build min/max arrays for all parameters
    min_values = numpy.array([medlynslope_mins, dleaf_mins, 2*10**-9, 0.02, 0.5, 0.0005])
    max_values = numpy.array([medlynslope_maxs, dleaf_maxs, 3.8*10**-8, 5, 1, 0.1])

    # generate parameter sets
    param_sets = (max_values - min_values)*lhd + min_values
    #print(type(param_sets))

    return param_sets

def write_params(param_sets, param_file=None):
    """
    Writes params from gen_params() to disk.
    """
    if param_file is None:
        param_file = os.path.dirname(os.path.abspath(__file__))
        param_file = os.path.join(param_file, "parameters")
    
    #with open(param_file, "w") as f:
    numpy.save(param_file, param_sets)
                
def read_params(param_file=None):
    """
    Reads params from disk.
    """
    data = None
    if param_file is None:
        param_file = os.path.dirname(os.path.abspath(__file__))
        param_file = os.path.join(param_file, "parameters.npy")

    #with open(param_file, "r") as f:
    data = numpy.load(file=param_file)
    #print (data)
    return data

def print_params(mode="pft", case_number=None):
    """
    Generates string output for use by bash script.

    :return: string
    """
    
    # Check to see if we have a case number.
    if case_number is None:
        print ("you gotsta gimme a case number!")
        return None

    # Instantiate the output string
    output = ""

    # Handle modes.
    # TODO: Instead of an if statement, these modes should be put in to
    # different functions.
    
    # Handle PFT mode.
    if mode is "pft":
        print ("pft mode not supported yet")
    
    # Handle namelist mode.
    elif mode is "namelist":
        # these are the params for the namelist format
        nl_param_names = ['fff','dint','baseflow_scalar']
        
        # get all the parameters from gen_params()
        sets = read_params()
        #print ("case_number: %s" % case_number)
        #print ("type: %s" % type(case_number))
        case = sets[int(case_number)]
        # grab the last 3 parameters
        nl_params = case[3:6]
        # add case number to output string
        #output += "\n----case: %s\n" % c_num

        # param counter
        p_count = 0

        # loop over the paramter names 
        for name in nl_param_names:
            # for each param name, put the corresponding value, and format
            # correctly.
            output += "%s=%s\n" % (name, nl_params[p_count])
            p_count += 1
            
    else:
        print ("Mode must be either \'pft\' or \'namelist\'")

    return output

#write_params(gen_params())
#read_params()

print (print_params("namelist", sys.argv[1]))


# write out parameter values
# temporary fix until this can be done in one script
#import csv
#with open('LHC.csv', 'w') as output:
#    writer = csv.writer(output, lineterminator='\n')
#    writer.writerows(param_sets)


