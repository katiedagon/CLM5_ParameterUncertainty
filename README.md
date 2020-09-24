# CLM5 Biophysical Parameter Uncertainty Project

This repository provides code to run a CLM5 perturbed parameter ensemble to investigate parameter uncertainty using neural networks.

## Requirements

Install [pyDOE package](https://pythonhosted.org/pyDOE/randomized.html#latin-hypercube) in order to use lhs function for Latin Hypercube sampling.

Install [tensorflow](https://www.tensorflow.org/) and [keras](https://keras.io/) packages for machine learning (neural networks).

<!---
Install [emcee](https://emcee.readthedocs.io/en/latest/) for MCMC sampling and [corner](https://corner.readthedocs.io/en/latest/) for visualization.
-->
## Ensemble Generating Files

The primary ensemble script is `ensemble_script_clm5` (bash). This sets up, configures, and submits ensemble cases. Within this you set the ensemble size.

This also calls 3 python scripts to set parameter values:

1) `LHC_write.py`

* Generate Latin Hypercube of parameter values based on specified parameters and ranges
* Write out parameter values to a file
* Do this only once for the ensemble

2) `LHC_read_pft.py`

* Read PFT-dependent parameter values from file
* Generate params files (netcdf) based on PFT-dependent parameter values
* Do this for each ensemble simulation

3) `LHC_read_nl.py`

* Read namelist parameter values from file
* Put them in the namelist in the proper format
* Do this for each ensemble simulation

There is a version specific for running an idealized future climate ensemble (`ensemble_script_clm5_warming`), where the surface temperature is increased by 2K everywhere (see also `perturb_TBOT` for how the forcing data is modified).

There is also a script for testing a single parameter set in CLM5: `run_paramset_clm5`. This does not rely on the above python scripts and parameter values are set manually in this script and by providing a modified params file.

There is also a script for running an [ILAMB](https://www.ilamb.org/) compatible test case: `run_ILAMB_case_clm5`.

## Data Processing Files

The `outputdata` folder includes scripts to process CLM output for training the neural network.

The `obs` folder includes scripts to process the observational data for optimizing the neural network. This folder also includes scripts for sampling, regridding, and plotting differences between the obs and model output. 

The `SVD` folder includes scripts to perform a singular value decomposition on model output and observations.  

## Machine Learning Files

The `NN` folder includes scripts used to construct, train, and optimize the neural networks.

* `NN_multi-dim.py`: Test out multidimensional output for the neural network.
* `NN_tune_*.py`: Hyperparameter tuning (e.g., learning rate, batch size).
* `NN_test.py`: Test out different NN configurations (# of layers, # of nodes).
* `NN_resample.py`: Use resampling of the training data to better refine candidate NN models.
* `NN_finalize*.py`: Finalize best NN model (single and multidimensional versions).
* `NN_finalize_plot.ipynb`: Finalize and plot NN emulator performance. 
* `NN_interp.py`: Interpret the NN results.
* `NN_interp_plot.ipynb`: Visualize the interpretation results.
* `NN_opt*.py`: Optimize the emulator predictions (weighted and unweighted versions).
* `NN_opt_plot.ipynb`: Plot the results of the NN optimization.
* `NN_parameter_response_maps.ipynb`: Explore optimal parameter space.
* `NN_reconstruct.ipynb`: Evaluating emulator performance based on EOF reconstruction.

## Supplemental Files

* `LHC_invert.py`: Invert the existing parameter array <!---(parameters.npy) -->back to the original LHC random sampling and write out these values<!--- (lhc.npy)-->.
* `LHC_visualize.ipynb`: Visualize the LHC-generated input values. 
* `simple_model.py`: Test simpler models between LHC values and CLM output (e.g., correlation coefficients, scatterplots, multi-linear regression).
* `time_test.py`: Calculate compute time for emulation.

## Parameter Sensitivity

The `Parameter_Sensitivity` folder include scripts used to generate and analyze one-at-a-time parameter sensitivity simulations for CLM5 biophysical parameters, the results of which inform parameter selection for the PPE.
