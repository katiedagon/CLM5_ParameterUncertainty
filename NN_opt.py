# For now run this ncar python env in the command line (or bash script)
# Not sure how to execute within python script:
#source /glade/work/kdagon/ncar_pylib_clone/bin/activate

# Optimize the 2-layer multiple output Neural Network
# Connecting obs, variance, predictions through likelihood function
# 4/9/19

from scipy import stats

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax

# Fix random seed for reproducibility
#np.random.seed(7)

# Read in input array
inputdata = np.load(file="lhc_100.npy")

# List of input variables
in_vars = ['medlynslope','dleaf','kmax','fff','dint','baseflow_scalar']

# Read in output array
# First 3 modes account for over 98% of variance
# Calculated in SVD.py
# After processing in outputdata/process_outputdata_SVD.ncl
outputdata = np.load(file="outputdata/outputdata_GPP_SVD_3modes.npy")
nmodes = outputdata.shape[1]

# Read in emulator predictions
# Generated in NN_finalize_multi-dim.py
model_preds = np.load(file="outputdata/emulated_GPP_SVD_3modes.npy")

# Read in observational targets
# Calculated in SVD.py
# After processing in obs/process_obs_SVD.ncl
obs = np.load(file="obs/obs_GPP_SVD_3modes.npy")

# Read in calculated variance
# Standard deviation of U_obs across full observational dataset
# 27 years, for the first 3 modes
# Calculated in SVD_obs.py
# After processing in obs/process_obs_SVD.ncl
sd = np.load(file="obs/obs_GPP_SVD_3modes_allyrs_sd.npy")

# Define likelihood function using emulator predictions
L = np.sum(((model_preds-obs)/sd)**2, axis=1)
#print(L.shape)
#plt.plot(L)
#plt.show()

# Isolate "best match" parameter set
# Based on simple minimum
Lmin = np.argmin(L)
print(Lmin)
print(L[Lmin])

# Print best match (LHC scaling values)
print(inputdata[Lmin,:])

# Read in actual parameter values
parameters = np.load(file="parameter_files/parameters_LHC_100.npy")
# Print best match (actual parameter values)
print(parameters[Lmin,:])


# Define likelihood function using actual CLM output
L_alt = np.sum(((outputdata-obs)/sd)**2, axis=1)
#print(L_alt.shape)
plt.plot(L_alt, label='CLM PPE')
plt.plot(L, label='NN Preds')
plt.legend()
plt.show()
Lmin_alt = np.argmin(L_alt)
print(Lmin_alt)
print(L_alt[Lmin_alt])
print(inputdata[Lmin_alt,:])
print(parameters[Lmin_alt,:])
