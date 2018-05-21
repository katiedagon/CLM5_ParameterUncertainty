import math
import numpy

from pyDOE import *
#help(lhs)

params = ['medlynslope','kmax','fff','dint','dleaf','baseflow_scalar']

n = len(params) # number of parameters
samp = 100 # sample size per parameter

lhd = lhs(n, samples=samp) # default sampling criterion = random

medlynslope_PFTs = ['NET/NDT','BET','BDT','SHR','C3ag','C3g','C4','C3c']

medlynslope_mins = [1.29,1.63,3.19,2.25,2.00,3.05,0.53,3.46]
medlynslope_maxs = [4.70,4.59,5.11,9.27,2.44,9.45,4.03,7.70]

dleaf_PFTs = ['NET','NDT','BET','BDT','BES','BDSt','BDSb','grass','crop']

dleaf_mins = [0.000216,0.00072,0.0081,0.0081,0.0081,0.000405,0.000162,0.000144,0.000162]
dleaf_maxs = [0.00108,0.0036,0.0567,0.243,0.081,0.1215,0.0486,0.018,0.1215]

min_values = [medlynslope_mins, 2*10**-9, 0.02, 0.5, dleaf_mins, 0.0005]
max_values = [medlynslope_maxs, 3.8*10**-8, 5, 1, dleaf_maxs, 0.1]



