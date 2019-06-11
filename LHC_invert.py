# For now run this ncar python env in the command line (or bash script)
# Not sure how to execute within python script:
#source /glade/work/kdagon/ncar_pylib_clone/bin/activate

# this doesn't work:
#import subprocess
#subprocess.call(["source","/glade/work/kdagon/ncar_pylib_clone/bin/activate"])

import sys
import os

import math
import numpy

# module for lhs function
#from pyDOE import *
#help(lhs)

def read_params(param_file=None):
        """
        Reads params from disk.
        """
        data = None
        if param_file is None:
            param_file = os.path.dirname(os.path.abspath(__file__))
            param_file = os.path.join(param_file,"parameters.npy")

        data = numpy.load(file=param_file, allow_pickle=True)
        return data

def gen_LHC():
    """
    Inverts the parameter array for original LHC sampling.

    :return: numpy.ndarray
    """
    #sets = read_params()
    #sets = read_params("parameters_LHC_100.npy")
    sets = read_params("parameters.npy")
    #print(sets)

    # condense down sets
    i=0
    while i < sets.shape[0]:
        sets[int(i)][0] = sets[int(i)][0][0]
        sets[int(i)][1] = sets[int(i)][1][0]
        i += 1

    #print(sets)

    # specify min/max values for PFT-dependent parameters
    medlynslope_mins = numpy.array([1.29,1.63,3.19,2.25,2.00,3.05,0.53,3.46])
    medlynslope_maxs = numpy.array([4.70,4.59,5.11,9.27,2.44,9.45,4.03,7.70])

    dleaf_mins = numpy.array([0.000216,0.00072,0.0081,0.0081,0.0081,0.000405,0.000162,0.000144,0.000162])
    dleaf_maxs = numpy.array([0.00108,0.0036,0.0567,0.243,0.081,0.1215,0.0486,0.018,0.1215])

    # build min/max arrays for all parameters
    # give it single PFT value to avoid nested arrays
    min_values = numpy.array([medlynslope_mins[0], dleaf_mins[0], 2*10**-9, 0.02, 0.5, 0.0005])
    max_values = numpy.array([medlynslope_maxs[0], dleaf_maxs[0], 3.8*10**-8, 5, 1, 0.1])

    # generate parameter sets
    lhd = (sets - min_values)/(max_values - min_values)
    #print(lhd)

    return lhd

def write_LHC(lhd, param_file=None):
    """
    Writes LHC values from gen_LHC() to disk.
    """
    if param_file is None:
        param_file = os.path.dirname(os.path.abspath(__file__))
        param_file = os.path.join(param_file, "lhc")

    numpy.save(param_file, lhd)

# run read_params, gen_LHC and write_LHC functions to save LHC values
# command line input specifies number of simulations/samples
write_LHC(gen_LHC())
