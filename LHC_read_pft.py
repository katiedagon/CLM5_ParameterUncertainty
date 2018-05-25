# For now run this ncar python env in the command line (or bash script)
# Not sure how to execute within python script:
#source /glade/p/work/kdagon/ncar_pylib_clone/bin/activate

# this doesn't work:
#import subprocess
#subprocess.call(["source","/glade/p/work/kdagon/ncar_pylib_clone/bin/activate"])

import sys
import os

import math
import numpy

from scipy.io import netcdf as nc

def read_params(param_file=None):
    """
    Reads params from disk.
    """
    data = None
    if param_file is None:
        param_file = os.path.dirname(os.path.abspath(__file__))
        param_file = os.path.join(param_file, "parameters.npy")

    data = numpy.load(file=param_file)
    return data

def generate_params_pft(case_number=None,inputfile=None):
    """
    Generates param files for use by bash script.
    PFT parameters only.

    :return: modified netcdf file
    """
    
    # Check to see if we have a case number.
    if case_number is None:
        print ("Missing case number")
        return None

    # Check for input file.
    if inputfile is None:
        print("Missing input file")
        return None

    # Read unmodified params file
    f = nc.netcdf_file(inputfile, 'a')

    # Specify PFT parameter names
    pft_param_names = ['medlynslope','dleaf','kmax']

    # Specify PFTs
    #medlynslope_PFTs = ['NET/NDT','BET','BDT','SHR','C3ag','C3g','C4','C3c']
    #dleaf_PFTs = ['NET','NDT','BET','BDT','BES','BDSt','BDSb','grass','crop']

    # get all the parameters values from read_params()
    sets = read_params()
    # case_number must be an integer
    case = sets[int(case_number)]
    # grab the PFT parameter values (first 3)                                                                                      
    pft_params = case[0:3]

    # param counter
    p_count = 0

    # this is ad-hoc for now
    for pnames in pft_param_names:
        var = f.variables[pnames]
        if len(var.dimensions) > 1:
            # deal with segment for kmax
            var[:,1:] = pft_params[2]
        elif pnames == "medlynslope":
            # medlynslope
            var[1:4] = pft_params[0][0]
            var[4:6] = pft_params[0][1]
            var[6:9] = pft_params[0][2]
            var[9:12] = pft_params[0][3]
            var[12] = pft_params[0][4]
            var[13] = pft_params[0][5]
            var[[14,17,18,67,68,75,76]] = pft_params[0][6]
            var[15:17] = pft_params[0][7]
            var[19:67] = pft_params[0][7]
            var[69:75] = pft_params[0][7]
            var[77:79] = pft_params[0][7]
        else:
            # dleaf
            var[1:3] = pft_params[1][0]
            var[3] = pft_params[1][1]
            var[4:6] = pft_params[1][2]
            var[6:9] = pft_params[1][3]
            var[9] = pft_params[1][4]
            var[10] = pft_params[1][5] 
            var[11] = pft_params[1][6]
            var[12:15] = pft_params[1][7] 
            var[15:] = pft_params[1][8]
        p_count += 1

    # close file (save modifications)
    f.close()
    
# Run read and generate params functions
# Must give a case number and inputfile (both specified in bash script)
# sys.argv knows to take the numbers entered after python command
generate_params_pft(sys.argv[1],sys.argv[2])

