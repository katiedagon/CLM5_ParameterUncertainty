source /glade/p/work/kdagon/ncar_pylib_clone/bin/activate

import math
import numpy

# module for lhs function
from pyDOE import *
#help(lhs)

# select parameters
params = ['medlynslope','kmax','fff','dint','dleaf','baseflow_scalar']

n = len(params) # number of parameters
#samp = 100 # sample size per parameter
samp = 10 # start small to test

# construct the latin hypercube
lhd = lhs(n, samples=samp) # default sampling criterion = random

# specify PFTs for relevant parameters
medlynslope_PFTs = ['NET/NDT','BET','BDT','SHR','C3ag','C3g','C4','C3c']
dleaf_PFTs = ['NET','NDT','BET','BDT','BES','BDSt','BDSb','grass','crop']

# specify min/max values for PFT-dependent parameters
medlynslope_mins = numpy.array([1.29,1.63,3.19,2.25,2.00,3.05,0.53,3.46])
medlynslope_maxs = numpy.array([4.70,4.59,5.11,9.27,2.44,9.45,4.03,7.70])

dleaf_mins = numpy.array([0.000216,0.00072,0.0081,0.0081,0.0081,0.000405,0.000162,0.000144,0.000162])
dleaf_maxs = numpy.array([0.00108,0.0036,0.0567,0.243,0.081,0.1215,0.0486,0.018,0.1215])

# build min/max arrays for all parameters
min_values = numpy.array([medlynslope_mins, 2*10**-9, 0.02, 0.5, dleaf_mins, 0.0005])
max_values = numpy.array([medlynslope_maxs, 3.8*10**-8, 5, 1, dleaf_maxs, 0.1])

# generate parameter sets
param_sets = (max_values - min_values)*lhd + min_values



