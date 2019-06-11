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
#np.random.seed(9)

# Read in input array
inputdata = np.load(file="lhc_100.npy", allow_pickle=True)

# List of input variables
in_vars = ['medlynslope','dleaf','kmax','fff','dint','baseflow_scalar']
npar = len(in_vars)

# Read in output array
# First 3 modes account for over 98% of variance
# Calculated in SVD.py
# After processing in outputdata/process_outputdata_SVD.ncl
outputdata_GPP = np.load(file="outputdata/outputdata_GPP_SVD_3modes.npy",
        allow_pickle=True)
outputdata_LHF = np.load(file="outputdata/outputdata_LHF_SVD_3modes.npy",
        allow_pickle=True)
nmodes = outputdata_GPP.shape[1]

# Read in emulator predictions
# Generated in NN_finalize_multi-dim.py
#model_preds = np.load(file="outputdata/emulated_GPP_SVD_3modes.npy")
# Instead, use trained NN to define cost function
# Build, compile, and fit here (once) before optimization
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import RMSprop
#from keras.regularizers import l2
import keras.backend as K

#model = Sequential()
#model.add(Dense(9, input_dim=inputdata.shape[1], activation='relu',
#    kernel_regularizer=l2(.001)))
#model.add(Dense(9, activation='tanh', kernel_regularizer=l2(.001)))
#model.add(Dense(nmodes))
#opt_dense = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
def mean_sq_err(y_true,y_pred):
    return K.mean((y_true-y_pred)**2)
#model.compile(opt_dense, "mse", metrics=[mean_sq_err])
#results = model.fit(inputdata, outputdata, epochs=500, batch_size=30, verbose=0)

# Load previously trained model
from keras.models import load_model
model_GPP = load_model('NN_GPP_finalize_multi-dim.h5', custom_objects={'mean_sq_err':
    mean_sq_err})
model_LHF = load_model('NN_LHF_finalize_multi-dim.h5',
    custom_objects={'mean_sq_err': mean_sq_err})

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
obs_GPP = np.load(file="obs/obs_GPP_SVD_3modes.npy", allow_pickle=True)
obs_LHF = np.load(file="obs/obs_LHF_SVD_3modes.npy", allow_pickle=True)
#print(obs)

# Read in calculated variance
# Standard deviation of U_obs across full observational dataset
# 27 years, for the first 3 modes
# Calculated in SVD_obs.py
# After processing in obs/process_obs_SVD.ncl
sd_GPP = np.load(file="obs/obs_GPP_SVD_3modes_allyrs_sd.npy", allow_pickle=True)
sd_LHF = np.load(file="obs/obs_LHF_SVD_3modes_allyrs_sd.npy", allow_pickle=True)

# Set weighting factor for likelihood function
#B = 5220728/4714979.5 #Lmin_GPP / Lmin_LHF (CLM PPE)
#B = 6071773/3619373 #Lmin_GPP / Lmin_LHF (SHGO opt)
#B = 5220728/3619373 #Lmin_GPP / Lmin_LHF (min across all evals)
#B = 1.3
B = 1

# Define likelihood function using emulator predictions
def normerr(x):
    #print(x)
    #print(x.shape)
    xt = x.reshape(1,-1)
    #print(xt.shape)
    model_preds_GPP = model_GPP.predict(xt)
    model_preds_LHF = model_LHF.predict(xt)
    #print(model_preds)
    #L = np.sum(((model_preds-obs)/sd)**2, axis=1)
    #L = np.sum(((model_preds_GPP-obs_GPP)/sd_GPP)**2, axis=1) + np.sum(((model_preds_LHF-obs_LHF)/sd_LHF)**2, axis=1)
    L = np.sum(((model_preds_GPP-obs_GPP)/sd_GPP)**2, axis=1) + B*np.sum(((model_preds_LHF-obs_LHF)/sd_LHF)**2, axis=1)
    #print(L)
    return L

# Report predicts and normerr of specific param set
#xtest = np.array([0.0649900246,0.999909575,0.00000923707104,0.999953316,
#    0.0000373489882,0.0000148023339])
#xtest = np.array([1,1,0,1,0,0])
#print(normerr(xtest))
#print(model.predict(xtest.reshape(1,-1)))

# Define initial condition parameter values (LHC scalings)
# Start in the middle of the uncertainty range (arbitrary)
#x0 = np.ones((1,npar))*0.5 # Nelder-Mead format
#print(x0.shape)
#x0 = np.ones(npar)*0.5 # Other formats
#x0 = np.array([0,1,0,1,0,0])
#x0 = np.array([1,1,0,1,0,0])
#x0 = np.array([0.99987907,1,0,0.99982818,0,0])
#print(normerr(x0))

# Algorithms may be getting stuck on initial values (unclear why)
# Try generating LHC initial values
#from pyDOE import *
#lhd = lhs(npar,samples=1) # default sampling criterion = random
#x0 = lhd[0,:]
#print(x0)
#print(normerr(x0))

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
#print(res)
#print(res.x)
#print(res.fun)

# SLSQP algorithm
#res = minimize(normerr, x0, method='SLSQP', bounds=bounds,
#        options={'disp':True})
#print(res)
#print(res.x)
#print(res.fun)

# L-BFGS-B algorithm
#res = minimize(normerr, x0, method='L-BFGS-B', bounds=bounds,
#        options={'disp':True})
#print(res.x)
#print(res.fun)

# TNC algorithm
#res = minimize(normerr, x0, method='TNC', bounds=bounds, options={'disp':True})
#print(res)
#print(res.x)
#print(res.fun)

# Differential Evolution (global)
#from scipy.optimize import differential_evolution
#bounds = [(0,1), (0,1), (0,1), (0,1), (0,1), (0,1)]
#res = differential_evolution(normerr, bounds, init='latinhypercube', disp=True)
#res = differential_evolution(normerr, bounds, init='random', disp=True)
#print(res)
#print(res.x)
#print(res.fun)
#print(res.nit)
#print(res.nfev)
#print(res.success)
#print(res.message)
#opt_preds = model.predict(res.x.reshape(1,-1))
#print(opt_preds)

# Brute Force (global)
#from scipy import optimize
#rranges = (slice(0, 1, 0.25), slice(0, 1, 0.25), slice(0, 1, 0.25), slice(0, 1,
#    0.25), slice(0, 1, 0.25), slice(0, 1, 0.25))
#rranges = ((0,1), (0,1), (0,1), (0,1), (0,1), (0,1))
#lb = np.zeros(npar, dtype=float)
#ub = np.ones(npar, dtype=float)
#rranges = zip(lb, ub)
#resb = optimize.brute(normerr, rranges, Ns=2, full_output=True, finish=optimize.fmin, disp=True)
#resb = optimize.brute(normerr, rranges, full_output=True, finish=optimize.fmin, disp=True)
#print(resb[0])
#print(resb[1])

# SHGO (global)
from scipy.optimize import shgo
bounds = [(0,1), (0,1), (0,1), (0,1), (0,1), (0,1)]
res = shgo(normerr, bounds)
#res = shgo(normerr, bounds, options={'disp':True})
#res = shgo(normerr, bounds, options={'disp':True}, sampling_method='sobol')
print(res)
#print(res.x)
#print(res.fun)
#opt_preds = model.predict(res.x.reshape(1,-1))
#print(opt_preds)
#print(opt_preds.shape)
opt_preds_GPP = model_GPP.predict(res.x.reshape(1,-1))
opt_preds_LHF = model_LHF.predict(res.x.reshape(1,-1))
print(opt_preds_GPP)
print(opt_preds_LHF)

# Dual Annealing (global)
#from scipy.optimize import dual_annealing
#bounds = [(0,1), (0,1), (0,1), (0,1), (0,1), (0,1)]
#res = dual_annealing(normerr, bounds=bounds, x0=x0)
#res = dual_annealing(normerr, bounds, maxiter=10000, x0=x0)
#print(res)
#print(normerr(res.x))
#opt_preds = model.predict(res.x.reshape(1,-1))
#print(opt_preds)
#opt_preds_GPP = model_GPP.predict(res.x.reshape(1,-1))
#opt_preds_LHF = model_LHF.predict(res.x.reshape(1,-1))
#print(opt_preds_GPP)
#print(opt_preds_LHF)

# Nonlinear Least Squares
#from scipy.optimize import least_squares
#bounds = ([0,1])
#res = least_squares(normerr, x0, bounds=bounds)
#print(res)

# pyOpt - issues with python 2 syntax
#import pyOpt
#opt_prob = pyOpt.Optimization('Optimizing the Emulator',normerr)
#opt_prob.addObj('L')
#opt_prob.addVarGroup('x',6,'c',lower=0.0,upper=1.0,value=0.5)
#print(opt_prob)
#slsqp = pyOpt.pySLSQP()
#[fstr, xstr, inform] = slsqp(opt_prob,sens_type='FD')
#print(opt_prob.solution)

# Manually calculating L from NN prediction (n=100)
#model_preds = model.predict(inputdata)
model_preds_GPP = model_GPP.predict(inputdata)
model_preds_LHF = model_LHF.predict(inputdata)
#print(model_preds.shape)
#L = np.sum(((model_preds-obs)/sd)**2, axis=1)
#L = np.sqrt((1/3)*(np.sum(((model_preds-obs)/sd)**2, axis=1)))
#L = np.sum(((model_preds_GPP-obs_GPP)/sd_GPP)**2, axis=1) + np.sum(((model_preds_LHF-obs_LHF)/sd_LHF)**2, axis=1)
L = np.sum(((model_preds_GPP-obs_GPP)/sd_GPP)**2, axis=1) + B*np.sum(((model_preds_LHF-obs_LHF)/sd_LHF)**2, axis=1)
#print(L.shape)
#plt.plot(L)
#plt.show()

# Isolate "best match" parameter set
# Based on simple minimum
Lmin = np.argmin(L)
print(Lmin)
print(L[Lmin])
#print(model_preds[Lmin,:])
# Print best match (LHC scaling values)
#print(inputdata[Lmin,:])

# Read in actual parameter values
#parameters = np.load(file="parameter_files/parameters_LHC_100.npy",allow_pickle=True)
# Print best match (actual parameter values)
#print(parameters[Lmin,:])

# Plot comparison between distributions
# Mode 1
fig=plt.figure()
ax=plt.subplot(111)
#ax.hist(outputdata_GPP[:,0], label='CLM PPE')
#ax.hist(model_preds_GPP[:,0], label='NN Preds')
#plt.xlabel('EOF1 GPP')
ax.hist(outputdata_LHF[:,0], label='CLM PPE')
ax.hist(model_preds_LHF[:,0], label='NN Preds')
plt.xlabel('EOF1 LHF')
plt.ylabel('Counts')
# this number is taken from SVD on hydro_ensemble_LHC_86
# which is the paramset from the original LHC that produces the min L
#ax.axvline(x=0.38246822, color='#1f77b4', linestyle='dashed', linewidth=2,
#        label='CLM PPE with min normerr')
# this number is the min L from the predictions
#ax.axvline(x=model_preds[Lmin,0], color='#ff7f0e', linestyle='dashed',
#                linewidth=2, label='NN Preds with min normerr')
# also show obs target
#ax.axvline(x=obs_GPP[:,0], color='red', linestyle='dashed', linewidth=2,
#        label='obs')
ax.axvline(x=obs_LHF[:,0], color='red', linestyle='dashed', linewidth=2, label='obs')
# and optimization result
#ax.axvline(x=opt_preds_GPP[:,0], color='green', linestyle='dashed', linewidth=2,
#        label='Optimized NN Preds')
ax.axvline(x=opt_preds_LHF[:,0], color='green', linestyle='dashed', linewidth=2,
        label='Optimized NN Preds')
# and model run with optimized params
# this number is taken from SVD on test_paramset_SVD_006
#ax.axvline(x=0.4248498, color='blue', linestyle='dashed', linewidth=2,
#        label='CLM with optimized params')
#ax.axvline(x=-0.18539695, color='blue', linestyle='dashed', linewidth=2,
#        label='CLM with optimized params')
# this number is taken from SVD on test_paramset_LHF_SVD_001
#ax.axvline(x=-0.36178064, color='lightblue', linestyle='dashed', linewidth=2,
#        label='CLM with optimized params V1')
# this number is taken from SVD on test_paramset_GPP_LHF_SVD_001
#ax.axvline(x=0.41596237, color='blue', linestyle='dashed', linewidth=2,
#        label='CLM with optimized params')
#ax.axvline(x=-0.27669725, color='blue', linestyle='dashed', linewidth=2,
#        label='CLM with optimized params')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.savefig("dist_outputdata_NNv005_GPP_SVD_md_mode1.pdf")
#plt.savefig("dist_outputdata_NNv006_GPP_SVD_md_mode1.pdf")
#plt.savefig("dist_outputdata_NNv001_LHF_SVD_md_mode1.pdf")
#plt.savefig("dist_outputdata_NNv002_LHF_SVD_md_mode1.pdf")
#plt.savefig("dist_outputdata_NNv001_GPP_LHF_SVD_md_mode1.pdf")
plt.show()

# Mode 2
fig=plt.figure()
ax=plt.subplot(111)
#ax.hist(outputdata_GPP[:,1], label='CLM PPE')
#ax.hist(model_preds_GPP[:,1], label='NN Preds')
#plt.xlabel('EOF2 GPP')
ax.hist(outputdata_LHF[:,1], label='CLM PPE')
ax.hist(model_preds_LHF[:,1], label='NN Preds')
plt.xlabel('EOF2 LHF')
plt.ylabel('Counts')
#ax.axvline(x=obs_GPP[:,1], color='red', linestyle='dashed', linewidth=2,
#        label='obs')
ax.axvline(x=obs_LHF[:,1], color='red', linestyle='dashed', linewidth=2,
        label='obs')
#ax.axvline(x=opt_preds_GPP[:,1], color='green', linestyle='dashed',
#        linewidth=2, label='Optimized NN Preds')
ax.axvline(x=opt_preds_LHF[:,1], color='green', linestyle='dashed', linewidth=2,
        label='Optimized NN Preds')
#ax.axvline(x=-0.5293461, color='blue', linestyle='dashed', linewidth=2,
#        label='CLM with optimized params')
#ax.axvline(x=0.0494138, color='blue', linestyle='dashed', linewidth=2,
#        label='CLM with optimized params')
#ax.axvline(x=-0.17363021, color='lightblue', linestyle='dashed', linewidth=2,
#        label='CLM with optimized params V1')
#ax.axvline(x=-0.4265692, color='blue', linestyle='dashed', linewidth=2,
#        label='CLM with optimized params')
#ax.axvline(x=-0.08327985, color='blue', linestyle='dashed', linewidth=2,                                                                         
#        label='CLM with optimized params')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.savefig("dist_outputdata_NNv006_GPP_SVD_md_mode2.pdf")
#plt.savefig("dist_outputdata_NNv001_LHF_SVD_md_mode2.pdf")
#plt.savefig("dist_outputdata_NNv002_LHF_SVD_md_mode2.pdf")  
#plt.savefig("dist_outputdata_NNv001_GPP_LHF_SVD_md_mode2.pdf") 
plt.show()

# Mode 3
fig=plt.figure()
ax=plt.subplot(111)
#ax.hist(outputdata_GPP[:,2], label='CLM PPE')
#ax.hist(model_preds_GPP[:,2], label='NN Preds')                                                                                                 
#plt.xlabel('EOF3 GPP')
ax.hist(outputdata_LHF[:,2], label='CLM PPE')
ax.hist(model_preds_LHF[:,2], label='NN Preds')
plt.xlabel('EOF3 LHF')
plt.ylabel('Counts')
#ax.axvline(x=obs_GPP[:,2], color='red', linestyle='dashed', linewidth=2,
#        label='obs')
ax.axvline(x=obs_LHF[:,2], color='red', linestyle='dashed', linewidth=2,
        label='obs')
#ax.axvline(x=opt_preds_GPP[:,2], color='green', linestyle='dashed', linewidth=2,
#        label='Optimized NN Preds')
ax.axvline(x=opt_preds_LHF[:,2], color='green', linestyle='dashed', linewidth=2,
        label='Optimized NN Preds')
#ax.axvline(x=0.5337539, color='blue', linestyle='dashed', linewidth=2, 
#        label='CLM with optimized params') 
#ax.axvline(x=-0.49806377, color='blue', linestyle='dashed', linewidth=2,
#        label='CLM with optimized params')
#ax.axvline(x=0.35999912, color='blue', linestyle='dashed', linewidth=2,
#        label='CLM with optimized params V1') 
#ax.axvline(x=-0.32231316, color='blue', linestyle='dashed', linewidth=2,
#        label='CLM with optimized params')
#ax.axvline(x=0.37923342, color='blue', linestyle='dashed', linewidth=2,                                                                        
#        label='CLM with optimized params')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.savefig("dist_outputdata_NNv006_GPP_SVD_md_mode3.pdf")
#plt.savefig("dist_outputdata_NNv001_LHF_SVD_md_mode3.pdf")
#plt.savefig("dist_outputdata_NNv002_LHF_SVD_md_mode3.pdf")    
#plt.savefig("dist_outputdata_NNv001_GPP_LHF_SVD_md_mode3.pdf")
plt.show()

# Define likelihood function using actual CLM output
#L_alt = np.sum(((outputdata_LHF-obs_LHF)/sd_LHF)**2, axis=1)
#L_alt = np.sqrt((1/3)*(np.sum(((outputdata-obs)/sd)**2, axis=1)))
#L_alt = np.sum(((outputdata_GPP-obs_GPP)/sd_GPP)**2, axis=1) + np.sum(((outputdata_LHF-obs_LHF)/sd_LHF)**2, axis=1) 
L_alt = np.sum(((outputdata_GPP-obs_GPP)/sd_GPP)**2, axis=1) + B*np.sum(((outputdata_LHF-obs_LHF)/sd_LHF)**2, axis=1)
#print(L_alt.shape)
#plt.plot(L_alt, label='CLM PPE')
#plt.hist(L_alt, label='CLM PPE')
#plt.plot(L, label='NN Preds')
#plt.hist(L, label='NN Preds')
#plt.legend()
#plt.xlabel('Parameter Set')
#plt.xlabel('Normalized Error')
#plt.ylabel('Normalized Error')
#plt.show()
Lmin_alt = np.argmin(L_alt)
print(Lmin_alt)
print(L_alt[Lmin_alt])
#print(inputdata[Lmin_alt,:])
#print(outputdata_LHF[Lmin_alt,:])
#print(parameters[Lmin_alt,:])
#print(np.min(L_alt))
#print(np.min(L))
#plt.plot(np.argmin(L_alt), np.min(L_alt), color='#1f77b4', marker='o',
#        markersize=12)
#plt.plot(np.argmin(L), np.min(L), color='#ff7f0e', marker='o', markersize=12)
#plt.savefig("normerr_outputdata_NNv005.pdf")
#plt.show()
