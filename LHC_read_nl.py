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

def print_params_nl(case_number=None):
    """
    Generates string output for use by bash script.
    Namelist parameters only.

    :return: string
    """
    
    # Check to see if we have a case number.
    if case_number is None:
        print ("Missing case number")
        return None

    # Instantiate the output string
    output = ""

    # Specify namelist parameter names
    nl_param_names = ['fff','dint','baseflow_scalar']
        
    # get all the parameters values from read_params()
    sets = read_params()
    # case_number must be an integer
    case = sets[int(case_number)]
    # grab the namelist parameter values (last 3)
    nl_params = case[3:6]
    # add case number to output string
    #output += "\n----case: %s\n" % c_num

    # param counter
    p_count = 0

    # loop over the paramter names 
    for name in nl_param_names:
    # for each param name, put the corresponding value, and format correctly.
        output += "%s=%s\n" % (name, nl_params[p_count])
        p_count += 1
            
    return output

# Run read and print functions to output
# Must give a case number (specified in bash script)
# sys.argv knows to take the number entered after python command
print (print_params_nl(sys.argv[1]))


# write out parameter values as csv file
# temporary fix until this can be done in one script
#import csv
#with open('LHC.csv', 'w') as output:
#    writer = csv.writer(output, lineterminator='\n')
#    writer.writerows(param_sets)


