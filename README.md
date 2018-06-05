# CLM5 Biogeophysical Parameter Sensitivity and Optimization Project

This repository provides code to run a CLM5 ensemble to investigate parameter sensitivity and optimization using neural networks.

# Requirements

Create a clone of the [NCAR package library](https://www2.cisl.ucar.edu/resources/computational-systems/cheyenne/software/python)

Install [pyDOE package](https://pythonhosted.org/pyDOE/randomized.html#latin-hypercube) in order to use lhs function for Latin Hypercube sampling

```bash
# Create clone
ncar_pylib -c 20180129 /glade/p/work/kdagon/ncar_pylib_clone
cd /glade/p/work/kdagon/ncar_pylib_clone
# Edit /bin/activate to change VIRTUAL_ENV variable to be /glade/p/work/kdagon/ncar_pylib_clone
# Source the virtual environment
source /bin/activate
# Install package with specified install dir
pip install --upgrade -t /glade/p/work/kdagon/ncar_pylib_clone/lib/python3.6/site-packages pyDOE
# Import pyDOE in python
python
import pyDOE
help(pyDOE.lhs)
```

# Primary Files

The master script is ensemble_script_clm5 (bash). This sets up ensemble cases, configuration, etc. Within this you set the ensemble size.

This also calls 3 python scripts to set parameter values:

1) LHC_write.py

* Generate latin hypercube of parameter values based on specifies parameters and ranges
* Write out parameter values to a file
* Do this only once for the ensemble

2) LHC_read_pft.py

* Read PFT-dependent parameter values from file
* Generate params files (netcdf) based on PFT-dependent parameter values
* Do this for each ensemble simulation

3) LHC_read_nl.py

* Read namelist parameter values from file
* Put them in the namelist in the proper format
* Do this for each ensemble simulation

# Supplemental Files

* LHC_invert.py inverts the existing parameter array back to the original LHC random sampling
* pft_var.ncl provides NCL script for generating PFT-dependent param files
* clm5_params.c171117.nc is the current CLM5 default parameter file
