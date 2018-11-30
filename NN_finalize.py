# For now run this ncar python env in the command line (or bash script)
# Not sure how to execute within python script:
#source /glade/work/kdagon/ncar_pylib_clone/bin/activate

# Finalize the 2-layer single output Neural Network
# 8/16/18

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
np.random.seed(7)

# Read in input array
inputdata = np.load(file="lhc_100.npy")

# List of input variables
in_vars = ['medlynslope','dleaf','kmax','fff','dint','baseflow_scalar']

# Read in output array
outputdata = np.loadtxt("outputdata/outputdata_GPP.csv")
#outputdata_all = np.load("outputdata/outputdata_GPP_SVD.npy")

# Specify mode (SVD only)
#mode = 3
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
model.add(Dense(8, input_dim=inputdata.shape[1], activation='linear',
    kernel_regularizer=l2(.001)))
# second layer with hyperbolic tangent activation
model.add(Dense(2, activation='tanh', kernel_regularizer=l2(.001)))
# output layer with linear activation
model.add(Dense(1))

# Define model metrics
def mean_sq_err(y_true,y_pred):
    return K.mean((y_true-y_pred)**2)

# Compile model
opt_dense = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(opt_dense, "mse", metrics=[mean_sq_err])
#model.summary()

# Separate training/test/val data: 60/20/20 split
#x_train = inputdata[0:60,:]
#x_test = inputdata[60:80,:]
#x_val = inputdata[80:,:]
#y_train = outputdata[0:60]
#y_test = outputdata[60:80]
#y_val = outputdata[80:]

# Fit the model using ALL data
results = model.fit(inputdata, outputdata, epochs=500, batch_size=30, verbose=0)
#print(results.history)

# Plot training history by epoch - are these lines the same?
#plt.plot(results.epoch, results.history['loss'], label='test')
#plt.plot(results.epoch, results.history['mean_sq_err'], label='train')
#plt.xticks(results.epoch)
#plt.legend()
#plt.hlines(y=0,xmin=0,xmax=15)
#plt.hlines(y=0,xmin=0,xmax=40)
#plt.ylabel('Mean Squared Error')
#plt.xlabel('Epoch')
#plt.title('Neural Network Training History')                                                                                                  
#plt.savefig("train_history_RMSprop.eps")
#plt.show()

# Make predictions
model_preds = model.predict(inputdata)[:,0]

# model metric for predictions
def mse_preds(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)

# calculate model mean error with predictions
model_me = mse_preds(outputdata, model_preds)

# scatterplot actual versus predicted
plt.scatter(outputdata, model_preds)
plt.xlabel('CLM Model Output')
plt.ylabel('NN Predictions')
plt.xlim(np.amin([outputdata,model_preds])-0.1,np.amax([outputdata,model_preds])+0.1)
plt.ylim(np.amin([outputdata,model_preds])-0.1,np.amax([outputdata,model_preds])+0.1)
#plt.savefig("validation_scatter_finalize_SVD_mode1.pdf")
#plt.savefig("validation_scatter_finalize_SVD_mode2.pdf")
#plt.savefig("validation_scatter_finalize_SVD_mode3.pdf")
plt.savefig("validation_scatter_finalize_GM_GPP_002.pdf")
plt.show()

# linear regression of actual vs predicted
slope, intercept, r_value, p_value, std_err = stats.linregress(outputdata,
                model_preds)

print("Model Mean Error:", results.history['mean_sq_err'][-1])
print("Prediction Mean Error: ", model_me)
print("r-squared:", r_value**2)

##

# Predictions with inflated ensemble
inputdata_inflate = np.load(file="lhc_1000.npy")
model_preds_inflate = model.predict(inputdata_inflate)[:,0]

# GM GPP from process_obs_GM.ncl
GPP_obs_GM = 2.356544

# without obs line
plt.hist(model_preds_inflate,bins=20)
plt.xlabel('NN Predicted GM GPP')
plt.ylabel('Counts')
plt.savefig("dist_outputdata_GM_GPP_inflate1000_002.pdf")
plt.show()
# with obs line
plt.hist(model_preds_inflate,bins=20)
plt.xlabel('NN Predicted GM GPP')
plt.ylabel('Counts')
plt.axvline(x=GPP_obs_GM, color='r', linestyle='dashed', linewidth=2)
plt.savefig("dist_outputdata_GM_GPP_withobs_inflate1000_002.pdf")
plt.show()

# Read in actual parameter values
parameters = np.load(file="parameter_files/parameters_LHC_1000.npy")

# Isolate "best match" parameter set
diff = abs(model_preds_inflate - GPP_obs_GM)
#print(diff)
pset = np.argmin(diff)
print(pset)
print(model_preds_inflate[pset])

# Print best match (scaling values)
print(inputdata_inflate[pset,:])
# Print best match (parameter values)
print(parameters[pset,:])


# Next: run CLM with the above parameter values
# Calculate resulting GM GPP and plot on histogram as above
