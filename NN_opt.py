# For now run this ncar python env in the command line (or bash script)
# Not sure how to execute within python script:
#source /glade/work/kdagon/ncar_pylib_clone/bin/activate

# Optimize the 2-layer multiple output Neural Network
# Connecting obs, variance, predictions through likelihood function
# 4/9/19

from scipy import stats
from scipy.optimize import minimize

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax

# Fix random seed for reproducibility
#np.random.seed(7)

# Read in input array
inputdata = np.load(file="lhc_100.npy")

# List of input variables
in_vars = ['medlynslope','dleaf','kmax','fff','dint','baseflow_scalar']
npar = len(in_vars)

# Read in output array
# First 3 modes account for over 98% of variance
# Calculated in SVD.py
# After processing in outputdata/process_outputdata_SVD.ncl
outputdata = np.load(file="outputdata/outputdata_GPP_SVD_3modes.npy")
nmodes = outputdata.shape[1]

# Read in emulator predictions
# Generated in NN_finalize_multi-dim.py
#model_preds = np.load(file="outputdata/emulated_GPP_SVD_3modes.npy")
# Instead, use trained NN to define cost function
# Build, compile, and fit here (once) before optimization
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.regularizers import l2
import keras.backend as K

model = Sequential()
model.add(Dense(7, input_dim=inputdata.shape[1], activation='relu',
    kernel_regularizer=l2(.001)))
model.add(Dense(9, activation='tanh', kernel_regularizer=l2(.001)))
model.add(Dense(nmodes))
opt_dense = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
def mean_sq_err(y_true,y_pred):
        return K.mean((y_true-y_pred)**2)
model.compile(opt_dense, "mse", metrics=[mean_sq_err])
results = model.fit(inputdata, outputdata, epochs=500, batch_size=30, verbose=0)

# test predictive capability
#test = np.ones((1,npar))*0.5
#print(test)
#print(test.shape)
#test_preds = model.predict(test)
#print(test_preds)
#print(test_preds.shape)

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
def normerr(x):
    print(x)
    #print(x.shape)
    xt = x.reshape(1,-1)
    #print(xt.shape)
    model_preds = model.predict(xt)
    #print(model_preds)
    L = np.sum(((model_preds-obs)/sd)**2, axis=1)
    return L

# Define initial condition parameter values (LHC scalings)
# Start in the middle of the uncertainty range (arbitrary)
#x0 = np.ones((1,npar))*0.5 # Nelder-Mead format
#print(x0.shape)
#x0 = np.ones(npar)*0.5 # Other formats

# Algorithms may be getting stuck on initial values (unclear why)
# Try generating LHC initial values
#from pyDOE import *
#lhd = lhs(npar,samples=1) # default sampling criterion = random
#x0 = lhd[0,:]

## Trying out different optimization algorithms ##

# Nelder-Mead Simplex algorithm (unconstrained)
#res = minimize(normerr, x0, method='nelder-mead', options={'disp':True})
#print(res.x)
#print(res.fun)

# Trust-Region Constrained algorithm
#from scipy.optimize import Bounds
#bounds = Bounds(0,1)
#res = minimize(normerr, x0, method='trust-constr', bounds=bounds, 
#        options={'disp':True})
#print(res.x)
#print(res.fun)

# SLSQP algorithm
#res = minimize(normerr, x0, method='SLSQP', bounds=bounds,
#        options={'disp':True})
#print(res.x)
#print(res.fun)

# L-BFGS-B algorithm
#res = minimize(normerr, x0, method='L-BFGS-B', bounds=bounds,
#        options={'disp':True})
#print(res.x)
#print(res.fun)

# TNC algorithm
#res = minimize(normerr, x0, method='TNC', bounds=bounds, options={'disp':True})
#print(res.x)
#print(res.fun)

# Differential Evolution (global)
#from scipy.optimize import differential_evolution
#bounds = [(0,1), (0,1), (0,1), (0,1), (0,1), (0,1)]
#res = differential_evolution(normerr, bounds, init='LHC', disp=True)
#res = differential_evolution(normerr, bounds, init='random', disp=True)
#print(res.x)
#print(res.fun)
#print(res.nit)
#print(res.nfev)
#print(res.success)
#print(res.message)

# Brute Force (global)
#from scipy import optimize
#rranges = (slice(0, 1, 0.25), slice(0, 1, 0.25), slice(0, 1, 0.25), slice(0, 1,
#    0.25), slice(0, 1, 0.25), slice(0, 1, 0.25))
#resb = optimize.brute(normerr, bounds, Ns=2, full_output=True, finish=optimize.fmin, disp=True)
#resb = optimize.brute(normerr, rranges, full_output=True, finish=optimize.fmin, disp=True)
#resb[0]
#resb[1]

# SHGO (global)
from scipy.optimize import shgo
bounds = [(0,1), (0,1), (0,1), (0,1), (0,1), (0,1)]
res = shgo(normerr, bounds, options={'disp':True})
print(res.x)
print(res.fun)

# Older code from manually calculating L
#L = np.sum(((model_preds-obs)/sd)**2, axis=1)
#print(L.shape)
#plt.plot(L)
#plt.show()

# Isolate "best match" parameter set
# Based on simple minimum
#Lmin = np.argmin(L)
#print(Lmin)
#print(L[Lmin])

# Print best match (LHC scaling values)
#print(inputdata[Lmin,:])

# Read in actual parameter values
#parameters = np.load(file="parameter_files/parameters_LHC_100.npy")
# Print best match (actual parameter values)
#print(parameters[Lmin,:])

# Define likelihood function using actual CLM output
#L_alt = np.sum(((outputdata-obs)/sd)**2, axis=1)
#print(L_alt.shape)
#plt.plot(L_alt, label='CLM PPE')
#plt.plot(L, label='NN Preds')
#plt.legend()
#plt.show()
#Lmin_alt = np.argmin(L_alt)
#print(Lmin_alt)
#print(L_alt[Lmin_alt])
#print(inputdata[Lmin_alt,:])
#print(parameters[Lmin_alt,:])
