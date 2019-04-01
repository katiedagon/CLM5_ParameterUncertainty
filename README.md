# CLM5 Biogeophysical Parameter Sensitivity and Optimization Project

This repository provides code to run a CLM5 ensemble to investigate parameter sensitivity and optimization using neural networks.

# Requirements

Create a clone of the [NCAR package library](https://www2.cisl.ucar.edu/resources/computational-systems/cheyenne/software/python).

Install [pyDOE package](https://pythonhosted.org/pyDOE/randomized.html#latin-hypercube) in order to use lhs function for Latin Hypercube sampling.

Install [tensorflow](https://www.tensorflow.org/) and [keras](https://keras.io/) packages for machine learning (neural networks).

(Optional) Install [eofs](https://ajdawson.github.io/eofs/index.html) for SVD analysis.

(Optional) Install [PyMC3](https://docs.pymc.io/) or [emcee](http://dfm.io/emcee/current/) for MCMC analysis.

```bash
# Create clone
ncar_pylib -c 20180129 /glade/work/kdagon/ncar_pylib_clone
cd /glade/work/kdagon/ncar_pylib_clone
# Source the virtual environment
source /bin/activate
# Install packages with specified install dir
pip install --upgrade -t /glade/work/kdagon/ncar_pylib_clone/lib/python3.6/site-packages pyDOE
pip install --upgrade -t /glade/work/kdagon/ncar_pylib_clone/lib/python3.6/site-packages tensorflow
pip install --upgrade -t /glade/work/kdagon/ncar_pylib_clone/lib/python3.6/site-packages keras
```

# Ensemble Generating Files

The master ensemble script is ensemble_script_clm5 (bash). This sets up ensemble cases, configuration, etc. Within this you set the ensemble size.

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

There is also a script for testing a single parameter set in CLM5: run_paramset_clm5 (bash). This does not rely on the above python scripts and parameter values are set manually in this script and by providing a modified params file.

# Data Processing Files

* outputdata/process_outputdata.ncl and associated scripts process CLM output for training the neural network
* obs/process_obs.ncl and associated scripts process the observational data  
* SVD.py and associated scripts perform a singular value decomposition on observations and model output   
* compare_obs_GM.py compares observational data with distributions of global mean model output

# Machine Learning Files

* NN_create.py: Create and test out simple neural networks in Python with Keras.
* NN_develop.py: Further refine and train neural networks using parameter values and CLM model output.
* NN_multi-dim.py: Test out multidimensional output.
* NN_test.py: Test out different NN configurations (# of layers, # of nodes).
* NN_resample.py: Use resampling of the training data to better refine candidate NN models.
* NN_finalize.py: Finalize best NN model (single and multidimensional versions).

# Supplemental Files

* LHC_invert.py inverts the existing parameter array (parameters.npy) back to the original LHC random sampling and writes out these values (lhc.npy)
* pft_var.ncl provides an NCL script for generating PFT-dependent parameter files
* clm5_params.c171117.nc is the current CLM5 default parameter file for reference
* simple_model.py tests simpler models between LHC values and CLM output (e.g., correlation coefficients, scatterplots, multi-linear regression)
