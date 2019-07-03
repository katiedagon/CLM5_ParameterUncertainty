# Explore different weights for GPP and LHF
# for the combined optimization of the 2-layer multiple output Neural Network
# 7/2/19

#from scipy import stats
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.axes as ax

# Read in input array
#inputdata = np.load(file="lhc_100.npy", allow_pickle=True)

# List of input variables
#in_vars = ['medlynslope','dleaf','kmax','fff','dint','baseflow_scalar']
#npar = len(in_vars)

# Read in output array
# Calculated in SVD.py
# After processing in outputdata/process_outputdata_SVD.ncl
outputdata_GPP = np.load(file="outputdata/outputdata_GPP_SVD_3modes.npy",
        allow_pickle=True)
outputdata_LHF = np.load(file="outputdata/outputdata_LHF_SVD_3modes.npy",
        allow_pickle=True)
#nmodes = outputdata_GPP.shape[1]

# Load previously trained model
import keras.backend as K
def mean_sq_err(y_true,y_pred):
    return K.mean((y_true-y_pred)**2)
from keras.models import load_model
model_GPP = load_model('NN_GPP_finalize_multi-dim.h5', custom_objects={'mean_sq_err':
    mean_sq_err})
model_LHF = load_model('NN_LHF_finalize_multi-dim.h5',
    custom_objects={'mean_sq_err': mean_sq_err})

# Read in observational targets
# Calculated in SVD.py
# After processing in obs/process_obs_SVD.ncl
obs_GPP = np.load(file="obs/obs_GPP_SVD_3modes.npy", allow_pickle=True)
obs_LHF = np.load(file="obs/obs_LHF_SVD_3modes.npy", allow_pickle=True)

# Read in calculated variance
# Standard deviation of U_obs across full observational dataset
# 27 years, for the first 3 modes
# Calculated in SVD_obs.py
# After processing in obs/process_obs_SVD.ncl
sd_GPP = np.load(file="obs/obs_GPP_SVD_3modes_allyrs_sd.npy", allow_pickle=True)
sd_LHF = np.load(file="obs/obs_LHF_SVD_3modes_allyrs_sd.npy", allow_pickle=True)

# Set weighting factor sampling range for likelihood function
Brange = np.arange(0,3.01,0.01)

# Define likelihood function using emulator predictions
def normerr(x):
    xt = x.reshape(1,-1)
    model_preds_GPP = model_GPP.predict(xt)
    model_preds_LHF = model_LHF.predict(xt)
    L = np.sum(((model_preds_GPP-obs_GPP)/sd_GPP)**2, axis=1) + b*np.sum(((model_preds_LHF-obs_LHF)/sd_LHF)**2, axis=1)
    return L

# Set up SHGO (global optimization)
from scipy.optimize import shgo
bounds = [(0,1), (0,1), (0,1), (0,1), (0,1), (0,1)]

# Set up Lmin arrays
Lmin_GPP = np.zeros(len(Brange))
Lmin_LHF = np.zeros(len(Brange))
Lmin_LHF_wgt = np.zeros(len(Brange))
Lmin = np.zeros(len(Brange))

# Run the optimization over range of B
for ind, b in np.ndenumerate(Brange):
    res = shgo(normerr, bounds)
    opt_preds_GPP = model_GPP.predict(res.x.reshape(1,-1))
    opt_preds_LHF = model_LHF.predict(res.x.reshape(1,-1))
    Lmin_GPP[ind] = np.sum(((opt_preds_GPP-obs_GPP)/sd_GPP)**2, axis=1)
    Lmin_LHF[ind] = np.sum(((opt_preds_LHF-obs_LHF)/sd_LHF)**2, axis=1)
    Lmin[ind] = res.fun
    Lmin_LHF_wgt[ind] = b*Lmin_LHF[ind]

#print(Lmin_GPP, Lmin_LHF, Lmin)

# Plot Lmin for GPP and LHF vs B
plt.plot(Brange, Lmin_GPP, label="Lmin_GPP")
#plt.plot(Brange, Lmin_LHF, label="Lmin_LHF")
plt.plot(Brange, Lmin_LHF_wgt, label="Lmin_LHF_wgt")
#plt.plot(Brange, Lmin, label="Lmin_total")
plt.legend()
plt.xlabel("Optimization Weighting Factor")
plt.ylabel("Lmin")
#plt.savefig("choice_of_wgt_B_Lmin.pdf")
plt.savefig("choice_of_wgt_B_Lmin_LHF_wgt.pdf")
plt.show()

# Find intersection
#Lmin_inter = np.min(abs(Lmin_GPP-Lmin_LHF))
#print(Lmin_inter)
#Brange_min = Brange[np.argmin(abs(Lmin_GPP-Lmin_LHF))]
#print(Brange_min)

# Plot sum/avg
#plt.plot(Brange, Lmin_GPP+Lmin_LHF, label="unweighted")
#plt.plot(Brange, Lmin_GPP+Lmin_LHF_wgt, label="weighted")
#plt.plot(Brange, (Lmin_GPP+Lmin_LHF)/2)
#plt.legend()
#plt.show()

