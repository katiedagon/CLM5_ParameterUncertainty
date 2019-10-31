# For now run this ncar python env in the command line (or bash script)
# Not sure how to execute within python script:
#source /glade/work/kdagon/ncar_pylib_clone/bin/activate

# Finalize the 2-layer single output Neural Network
# 8/16/18

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

import keras.backend as K

from scipy import stats

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from scipy.io import netcdf as nc

# Fix random seed for reproducibility
#np.random.seed(7) #GM 002
#np.random.seed(5) #GM 006
#np.random.seed(3) # SVD 001
np.random.seed(9)

# Read in input array
inputdata = np.load(file="lhc_100.npy", allow_pickle=True)

# List of input variables
in_vars = ['medlynslope','dleaf','kmax','fff','dint','baseflow_scalar']

# Read in output array
#outputdata = np.loadtxt("outputdata/outputdata_GPP.csv")
#outputdata_all = np.load("outputdata/outputdata_GPP_SVD.npy")

# Training to predict global mean GPP, LHF
#var = "GPP"
var = "LHF"
f=nc.netcdf_file("outputdata/outputdata_"+var+"_GM_100_diff.nc",'r', mmap=False)
X = f.variables[var]
outputdata = X[:]

# Plot histogram of PPE diffs
#plt.hist(outputdata, bins=20)
#plt.xlabel('GM dGPP (gC/m2/yr)')
#plt.ylabel('Counts')
#plt.title('Difference in GM GPP (future-control)')
#plt.show()

# Specify mode (SVD only)
#mode = 1
#outputdata = outputdata_all[:,mode-1]
#plt.hist(outputdata, bins=20)
#plt.xlabel('Mode 1 of GPP SVD (U-vector)')
#plt.ylabel('Counts')
#plt.savefig("dist_outputdata_GPP_SVD_mode1.pdf")
#plt.show()

# Create 2-layer simple model
model = Sequential()
# specify input_dim as number of parameters, not number of simulations
# l2 norm regularizer
model.add(Dense(14, input_dim=inputdata.shape[1], activation='relu',
    kernel_regularizer=l2(.001)))
# second layer with hyperbolic tangent activation
model.add(Dense(6, activation='tanh', kernel_regularizer=l2(.001)))
# output layer with linear activation
model.add(Dense(1))

# Define model metrics
def mean_sq_err(y_true,y_pred):
    return K.mean((y_true-y_pred)**2)

# Compile model
opt_dense = RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0) # defaults except lr (GPP GM diff)
#opt_dense = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0) # defaults except lr (LHF GM diff)
#opt_dense = SGD(lr=0.005, momentum=0.99, decay=1e-4, nesterov=True)
#opt_dense = SGD(lr=0.01, momentum=0, nesterov=False) # defaut settings
#opt_dense = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False) # defaults except lr
#opt_dense = Adagrad(lr=0.01) # default settings
#opt_dense = Adadelta(lr=1, rho=0.95) # default settings
model.compile(opt_dense, "mse", metrics=[mean_sq_err])
model.summary()

# Separate training/test/val data: 60/20/20 split
#x_train = inputdata[0:60]
#x_test = inputdata[60:80]
#x_val = inputdata[80:]
#y_train = outputdata[0:60]
#y_test = outputdata[60:80]
#y_val = outputdata[80:]

# Fit model using train/test
#eps = 500
#results = model.fit(x_train, y_train, epochs=eps, batch_size=30,
#        verbose=0, validation_data=(x_test,y_test))

# Fit with stopping criteria when val_loss starts increasing
#es = EarlyStopping(monitor='val_loss', min_delta=1, 
#        patience=50, verbose=1, mode='min')
#results = model.fit(x_train, y_train, epochs=eps, batch_size=20,
#        callbacks=[es], validation_data=(x_test,y_test)) 

# Plot training history by epoch plt.plot(results.epoch, results.history['val_mean_sq_err'], label='test')
#plt.plot(results.epoch, results.history['val_mean_sq_err'], label='test')
#plt.plot(results.epoch, results.history['mean_sq_err'], label='train')
#plt.legend()                                                                                                                                  
#plt.ylabel('Mean Squared Error')                                                                                                              
#plt.xlabel('Epoch')                                                                                                                           
#plt.title('Neural Network Training History')                                                                                                  
#plt.savefig("train_history_"+var+"_GM_diff.pdf")                                                                                                      
#plt.show()

#print('Total epochs = '+str(max(results.epoch)+1))

# Plot training history after epoch X
#epcut = 100 # GPP GM diff
#epcut = 10 # LHF GM diff
#plt.plot(results.epoch[epcut:], results.history['val_mean_sq_err'][epcut:],
#        label='test')
#plt.plot(results.epoch[epcut:], results.history['mean_sq_err'][epcut:],
#        label='train')
#plt.legend()
#plt.ylabel('Mean Squared Error')
#plt.xlabel('Epoch')
#plt.title('Neural Network Training History after '+str(epcut)+' epochs')
#plt.savefig("train_history_lasteps_"+var+"_GM_diff.pdf")
#plt.show()

# Make predictions - using validation set (single dim)
#model_preds = model.predict(x_val)[:,0]
#model_test = model.predict(x_test)[:,0]
#model_train = model.predict(x_train)[:,0]

# Distribution of predictions vs. actual
#plt.hist(y_val, label="CLM Model Output")
#plt.hist(model_preds, label="NN Predictions")
#plt.legend()
#plt.ylabel("Counts")
#plt.xlabel("Difference in GM "+var+", future-control (W/m2)")
#plt.show()

# model metric for predictions
#def mse_preds(y_true,y_pred):
#    return np.mean((y_true-y_pred)**2)

# calculate model mean error with predictions
#model_me = mse_preds(y_val, model_preds)
#print("Validation MSE:", model_me)

# plot validation errors
#plt.hist((y_val-model_preds)**2, bins=20)
#plt.xlabel('Validation squared errors')
#plt.ylabel('Counts')
#plt.title('Distribution of validation errors')
#plt.savefig("dist_val_err_"+var+"_GM_diff.pdf")
#plt.show()

# scatterplot actual versus predicted
#plt.scatter(y_val, model_preds, label='validation')
#plt.scatter(y_train, model_train, label='training')
#plt.scatter(y_test, model_test, label= 'testing')
#plt.legend()
#plt.xlabel('CLM Model Output')
#plt.ylabel('NN Predictions')
#plt.title('Difference in GM '+var+', future-control (gC/m2/yr)')
#plt.title('Difference in GM '+var+', future-control (W/m2)')
#axbuff = 5 # GPP GM diff
#axbuff = 0.1 # LHF GM diff
#plt.xlim(np.amin(outputdata)-axbuff,np.amax(outputdata)+axbuff)
#plt.ylim(np.amin(outputdata)-axbuff,np.amax(outputdata)+axbuff)
#plt.savefig("validation_scatter_GPP-SVD-mode1.pdf")
#plt.savefig("validation_scatter_GPP_ET_SVD-mode1GPP.pdf")
#plt.savefig("validation_scatter_training_GPP_SVD_md_mode1.pdf")
#plt.savefig("validation_scatter_training_"+var+"_GM_diff.pdf")
#plt.show()

# linear regression actual vs. predicted
#slope, intercept, r_value, p_value, std_err = stats.linregress(y_val,model_preds)
#print("Prediction r-squared:", r_value**2)
#slope, intercept, r_value, p_value, std_err = stats.linregress(y_train,model_train)
#print("Training r-squared:", r_value**2)
#slope, intercept, r_value, p_value, std_err = stats.linregress(y_test,model_test)
#print("Testing r-squared:", r_value**2)


# Fit the model using ALL data (finalize step)
#eps = 300 # GPP GM diff
eps = 53 # LHF GM diff
results = model.fit(inputdata, outputdata, epochs=eps, batch_size=20, verbose=0)
#print(results.history)

# Check out the "training" history
#plt.plot(results.epoch, results.history['mean_sq_err'])
#plt.legend()                                                                                                                                  
#plt.ylabel('Mean Squared Error')                                                                                                              
#plt.xlabel('Epoch')                                                                                                                           
#plt.title('Neural Network Training History')                                                                                                  
#plt.show()

# Save finalized model
model.save('emulators/NN_'+var+'_finalize_GM_diff.h5')

# Make predictions - using ALL data
model_preds = model.predict(inputdata)[:,0]

# Distribution of predictions vs. actual
plt.hist(outputdata, label="CLM Model Output")
plt.hist(model_preds, label="NN Predictions")
plt.legend()
plt.ylabel("Counts")
plt.xlabel("Difference in GM "+var+", future-control (W/m2)")
plt.savefig("dist_outputdata_modelpreds_"+var+"_GM_diff.pdf")
plt.show()

# model metric for predictions
def mse_preds(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)

# calculate model mean error with predictions
model_me = mse_preds(outputdata, model_preds)

# scatterplot actual versus predicted
plt.scatter(outputdata, model_preds)
plt.xlabel('CLM Model Output')
plt.ylabel('NN Predictions')
#plt.title('Difference in GM '+var+', future-control (gC/m2/yr)')
plt.title('Difference in GM '+var+', future-control (W/m2)')
#axbuff = 5
axbuff = 0.2
plt.xlim(np.amin(outputdata)-axbuff,np.amax(outputdata)+axbuff)                                                                              
plt.ylim(np.amin(outputdata)-axbuff,np.amax(outputdata)+axbuff)
#plt.savefig("validation_scatter_finalize_SVD_mode1.pdf")
#plt.savefig("validation_scatter_finalize_SVD_mode2.pdf")
#plt.savefig("validation_scatter_finalize_SVD_mode3.pdf")
#plt.savefig("validation_scatter_finalize_GM_GPP_002.pdf")
plt.savefig("validation_scatter_finalize_"+var+"_GM_diff.pdf")
plt.show()

# linear regression of actual vs predicted
slope, intercept, r_value, p_value, std_err = stats.linregress(outputdata,model_preds)

print("Model Mean Error:", results.history['mean_sq_err'][-1])
print("Prediction Mean Error: ", model_me)
print("r-squared:", r_value**2)

##

# Predictions with inflated ensemble
#inputdata_inflate = np.load(file="lhc_1000.npy")
#model_preds_inflate = model.predict(inputdata_inflate)[:,0]

# GM GPP from process_obs_GM.ncl
#GPP_obs_GM = 2.356544
# SVD GPP Mode 1 from SVD.py
#GPP_obs_SVD_mode1 = 0.73506486

# without obs line
#plt.hist(model_preds_inflate,bins=20)
#plt.xlabel('NN Predicted GM GPP')
#plt.ylabel('Counts')
#plt.savefig("dist_outputdata_GM_GPP_inflate1000_002.pdf")
#plt.show()
# with obs line
#plt.hist(model_preds_inflate,bins=20)
#plt.xlabel('NN Predicted GM GPP')
#plt.xlabel('NN Predicted Mode 1 of GPP SVD')
#plt.ylabel('Counts')
#plt.axvline(x=GPP_obs_GM, color='r', linestyle='dashed', linewidth=2)
#plt.axvline(x=GPP_obs_SVD_mode1, color='r', linestyle='dashed', linewidth=2) 
#plt.savefig("dist_outputdata_GM_GPP_withobs_inflate1000_002.pdf")
#plt.savefig("dist_outputdata_GM_GPP_withobs_inflate1000_006.pdf")
#plt.savefig("dist_outputdata_GPP_SVD_mode1_withobs_inflate1000_001.pdf")
#plt.show()

# Read in actual parameter values
#parameters = np.load(file="parameter_files/parameters_LHC_1000.npy")

# Isolate "best match" parameter set
#diff = abs(model_preds_inflate - GPP_obs_GM)
#diff = abs(model_preds_inflate - GPP_obs_SVD_mode1)
#print(diff)
#pset = np.argmin(diff)
#print(pset)
#print(model_preds_inflate[pset])

# Print best match (scaling values)
#print(inputdata_inflate[pset,:])
# Print best match (parameter values)
#print(parameters[pset,:])

# Next: run CLM with the above parameter values
# Calculate resulting GM GPP and plot on histogram as above
#GPP_GM_002 = 2.444088
#GPP_GM_005 = 2.327848
#GPP_GM_006 = 2.34726
# Calculate resulting SVD mode 1 GPP and plot on histogram
#GPP_SVD_mode1_001 = 0.37745455
#plt.hist(model_preds_inflate,bins=20)
#plt.xlabel('NN Predicted GM GPP')
#plt.xlabel('NN Predicted Mode 1 of GPP SVD')
#plt.ylabel('Counts')
#plt.axvline(x=GPP_obs_GM, color='r', linestyle='dashed', linewidth=2)
#plt.axvline(x=GPP_obs_SVD_mode1, color='r', linestyle='dashed', linewidth=2)
#plt.axvline(x=GPP_GM_002, color='b', linestyle='dashed', linewidth=2)
#plt.axvline(x=GPP_GM_006, color='g', linestyle='dashed', linewidth=2)
# Also plot best match from inflated ensemble
#plt.axvline(x=model_preds_inflate[pset], color='c', linestyle='dashed',
#        linewidth=2) 
#plt.axvline(x=GPP_SVD_mode1_001, color='b', linestyle='dashed', linewidth=2)
#plt.savefig("dist_outputdata_GM_GPP_withobs_andmodel_inflate1000_002.pdf")
#plt.savefig("dist_outputdata_GPP_SVD_mode1_withobs_andmodel_inflate1000_001.pdf")
#plt.show()

# Add NN prediction vs. model output to scatter plot
#plt.scatter(outputdata, model_preds, c='silver')
#plt.plot(GPP_GM_006, model_preds_inflate[pset], 'b*', markersize=12)
#plt.xlabel('CLM Model Output')
#plt.ylabel('NN Predictions')
#plt.xlim(np.amin([outputdata,model_preds])-0.1,np.amax([outputdata,model_preds])+0.1)
#plt.ylim(np.amin([outputdata,model_preds])-0.1,np.amax([outputdata,model_preds])+0.1)
#plt.savefig("validation_scatter_finalize_GM_GPP_002_OOSP.pdf")
#plt.show()

# How do default model params compare?
#GPP_GM_default = 2.614403
#GPP_SVD_mode1_default = -0.06415711
#plt.hist(model_preds_inflate,bins=20)
#plt.xlabel('NN Predicted GM GPP')
#plt.xlabel('NN Predicted Mode 1 of GPP SVD')
#plt.ylabel('Counts')
#plt.axvline(x=GPP_obs_GM, color='r', linestyle='dashed', linewidth=2)
#plt.axvline(x=GPP_GM_002, color='b', linestyle='dashed', linewidth=2)
#plt.axvline(x=GPP_GM_006, color='g', linestyle='dashed', linewidth=2)
#plt.axvline(x=GPP_GM_default, color='k', linestyle='dashed', linewidth=2) 
#plt.axvline(x=GPP_obs_SVD_mode1, color='r', linestyle='dashed', linewidth=2)
#plt.axvline(x=model_preds_inflate[pset], color='c', linestyle='dashed',
#                linewidth=2)
#plt.axvline(x=GPP_SVD_mode1_001, color='b', linestyle='dashed', linewidth=2)
#plt.axvline(x=GPP_SVD_mode1_default, color='k', linestyle='dashed', linewidth=2)
#plt.savefig("dist_outputdata_GM_GPP_withobs_andmodel_anddefault_inflate1000_002.pdf")
#plt.savefig("dist_outputdata_GM_GPP_withobs_andmodel_anddefault_inflate1000_006.pdf")
#plt.savefig("dist_outputdata_GPP_SVD_mode1_withobs_andmodel_anddefault_inflate1000_001.pdf")
#plt.show()
