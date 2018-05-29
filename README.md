README

1/30/18

Attempting to install pyDOE package in order to use lhs function

https://pythonhosted.org/pyDOE/randomized.html#latin-hypercube

First create a clone of NCAR package library

https://www2.cisl.ucar.edu/resources/computational-systems/cheyenne/software/python
>ncar_pylib -c 20180129 /glade/p/work/kdagon/ncar_pylib_clone

>cd /glade/p/work/kdagon/ncar_pylib_clone

Edit /bin/activate to change VIRTUAL_ENV variable to be /glade/p/work/kdagon/ncar_pylib_clone

Then source the virtual environment
>source /bin/activate

Install package with specified install dir
>pip install --upgrade -t /glade/p/work/kdagon/ncar_pylib_clone/lib/python3.6/site-packages pyDOE

Import pyDOE in python
>python

>import pyDOE

>help(pyDOE.lhs)

5/22/18

Python script (LHC.py) is now updated to generate parameter sets.

ensemble_script_clm5 provides bash script for generating ensemble cases (create, setup, submit)

pft_var.ncl provides NCL script for generating PFT-dependent param files

5/29/18

Master script is ensemble_script_clm5 (bash)
Sets up ensemble cases, configuration, etc.
Within this you set the ensemble size

This also calls 3 python scripts to set parameter values:

1) LHC_write.py
Generate latin hypercube of parameter values based on specifies parameters and ranges
Write out parameter values to a file
Do this only once for the ensemble

2) LHC_read_pft.py
Read PFT-dependent parameter values from file
Generate params files (netcdf) based on PFT-dependent parameter values
Do this for each ensemble simulation

3) LHC_read_nl.py
Read namelist parameter values from file
Put them in the namelist in the proper format
Do this for each ensemble simulation
