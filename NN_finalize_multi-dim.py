# For now run this ncar python env in the command line (or bash script)
# Not sure how to execute within python script:
#source /glade/work/kdagon/ncar_pylib_clone/bin/activate

# Finalize the 2-layer multiple output Neural Network
# 10/11/18

#import time
#start = time.time()

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

import keras.backend as K

from scipy import stats

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax

# Fix random seed for reproducibility
np.random.seed(9)

# Read in input array
inputdata = np.load(file="lhc_100.npy")

# List of input variables
in_vars = ['medlynslope','dleaf','kmax','fff','dint','baseflow_scalar']

# Read in output array
#outputdata_raw = np.load(file="outputdata/outputdata_GPP_SVD.npy")
#outputdata = outputdata_raw[:,:3]
# First 3 modes account for over 98% of variance
outputdata = np.load(file="outputdata/outputdata_GPP_SVD_3modes.npy")
nmodes = outputdata.shape[1]

# Percent of variance (for weighted avg R^2)
# Calculated in SVD.py
#svd_var = [0.771, 0.1914, 0.0128]
svd_var = [0.8341, 0.1349, 0.0119]

# Create 2-layer simple model
model = Sequential()
# specify input_dim as number of parameters, not number of simulations
# l2 norm regularizer
model.add(Dense(14, input_dim=inputdata.shape[1], activation='relu',
    kernel_regularizer=l2(.001)))
# second layer with hyperbolic tangent activation
model.add(Dense(9, activation='tanh', kernel_regularizer=l2(.001)))
# output layer with linear activation
model.add(Dense(nmodes))

# Define model metrics
def mean_sq_err(y_true,y_pred):
    return K.mean((y_true-y_pred)**2)

# Compile model
opt_dense = RMSprop(lr=0.005, rho=0.9, epsilon=None, decay=0.0)
model.compile(opt_dense, "mse", metrics=[mean_sq_err])
#model.summary()

# Fit the model using ALL data
results = model.fit(inputdata, outputdata, epochs=500, batch_size=30, verbose=0)

# Save out model
model.save('NN_finalize_multi-dim.h5')

# Make predictions
model_preds = model.predict(inputdata)
#print(model_preds.shape)

# Save out predictions
#np.save("outputdata/emulated_GPP_SVD_3modes.npy", model_preds)

# model metric for predictions
def mse_preds(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)

# calculate model mean error with predictions
model_me = mse_preds(outputdata, model_preds)

# scatterplot actual versus predicted
plt.scatter(outputdata[:,0], model_preds[:,0])
plt.xlabel('CLM Model Output')
plt.ylabel('NN Predictions')
plt.title('EOF1 GPP')
plt.xlim(np.amin([outputdata[:,0],model_preds[:,0]])-0.1,np.amax([outputdata[:,0],model_preds[:,0]])+0.1)
plt.ylim(np.amin([outputdata[:,0],model_preds[:,0]])-0.1,np.amax([outputdata[:,0],model_preds[:,0]])+0.1)
plt.savefig("validation_scatter_finalize_GPP_SVD_md_mode1.pdf")
plt.show()
plt.scatter(outputdata[:,1], model_preds[:,1])
plt.xlabel('CLM Model Output')
plt.ylabel('NN Predictions')
plt.title('EOF2 GPP')
plt.xlim(np.amin([outputdata[:,1],model_preds[:,1]])-0.1,np.amax([outputdata[:,1],model_preds[:,1]])+0.1)
plt.ylim(np.amin([outputdata[:,1],model_preds[:,1]])-0.1,np.amax([outputdata[:,1],model_preds[:,1]])+0.1)
plt.savefig("validation_scatter_finalize_GPP_SVD_md_mode2.pdf")
plt.show()
plt.scatter(outputdata[:,2], model_preds[:,2])
plt.xlabel('CLM Model Output')
plt.ylabel('NN Predictions')
plt.title('EOF3 GPP')
plt.xlim(np.amin([outputdata[:,2],model_preds[:,2]])-0.1,np.amax([outputdata[:,2],model_preds[:,2]])+0.1)
plt.ylim(np.amin([outputdata[:,2],model_preds[:,2]])-0.1,np.amax([outputdata[:,2],model_preds[:,2]])+0.1)
plt.savefig("validation_scatter_finalize_GPP_SVD_md_mode3.pdf")
plt.show()

print("Model Mean Error: %.2g" % results.history['mean_sq_err'][-1])
print("Prediction Mean Error: %.2g" % model_me)

r_array = []
# linear regression of actual vs predicted
slope, intercept, r_value, p_value, std_err = stats.linregress(outputdata[:,0],
        model_preds[:,0])
print("Mode 1 r-squared: %.2g" % r_value**2)
r_array.append(r_value**2)
slope, intercept, r_value, p_value, std_err = stats.linregress(outputdata[:,1],
        model_preds[:,1])
print("Mode 2 r-squared: %.2g" % r_value**2)
r_array.append(r_value**2)
slope, intercept, r_value, p_value, std_err = stats.linregress(outputdata[:,2],
        model_preds[:,2])
print("Mode 3 r-squared: %.2g" % r_value**2)
r_array.append(r_value**2)

print("avg. r-squared: %.2g" % np.mean(r_array))
print("wgt avg. r-squared: %.2g" % np.average(r_array,weights=svd_var))

# Predictions with inflated ensemble
inputdata_inflate = np.load(file="lhc_1000.npy")
model_preds_inflate = model.predict(inputdata_inflate)
#emulate_time = time.time()
#print(emulate_time - start)

# Read in observational targets
# Calculated in SVD.py
# After processing in obs/process_obs_SVD.ncl
obs = np.load(file="obs/obs_GPP_SVD_3modes.npy")

# Read in model default
# Calculated in SVD.py
# After processing in outputdata/process_model_SVD.ncl
model_default = np.load(file="outputdata/modeldefault_GPP_SVD_3modes.npy")

#plt.hist(model_preds_inflate[:,0], bins=20)
#plt.xlabel('NN Predicted EOF1 GPP')
#plt.ylabel('Counts')
#plt.savefig("dist_outputdata_GPP_SVD_md_mode1_inflate1000.pdf")
#plt.show()

plt.hist(model_preds_inflate[:,0], bins=20)
plt.xlabel('NN Predicted EOF1 GPP')
plt.ylabel('Counts')
plt.axvline(x=obs[:,0], color='r', linestyle='dashed', linewidth=2)
plt.axvline(x=model_default[:,0], color='k', linestyle='dashed', linewidth=2)
plt.savefig("dist_outputdata_GPP_SVD_md_mode1_withobs_anddefault_inflate1000.pdf")
plt.show()

# with optimal value - found in NN_opt.py
#plt.hist(model_preds_inflate[:,0], bins=20)
#plt.xlabel('NN Predicted EOF1 GPP')
#plt.ylabel('Counts')
#plt.axvline(x=obs[:,0], color='r', linestyle='dashed', linewidth=2)
#plt.axvline(x=model_default[:,0], color='k', linestyle='dashed', linewidth=2)
#plt.axvline(x=0.35518807, color='b', linestyle='dashed', linewidth=2)
#plt.savefig("dist_outputdata_GPP_SVD_md_mode1_withobs_anddefault_andopt_inflate1000.pdf")
#plt.show()

plt.hist(model_preds_inflate[:,1], bins=20)
plt.xlabel('NN Predicted EOF2 GPP')                                                                                                             
plt.ylabel('Counts')
plt.axvline(x=obs[:,1], color='r', linestyle='dashed', linewidth=2)
plt.axvline(x=model_default[:,1], color='k', linestyle='dashed', linewidth=2)
plt.savefig("dist_outputdata_GPP_SVD_md_mode2_withobs_anddefault_inflate1000.pdf")
plt.show()

plt.hist(model_preds_inflate[:,2], bins=20)
plt.xlabel('NN Predicted EOF3 GPP')                                                                                                             
plt.ylabel('Counts')
plt.axvline(x=obs[:,2], color='r', linestyle='dashed', linewidth=2)
plt.axvline(x=model_default[:,2], color='k', linestyle='dashed', linewidth=2)
plt.savefig("dist_outputdata_GPP_SVD_md_mode3_withobs_anddefault_inflate1000.pdf")
plt.show()
