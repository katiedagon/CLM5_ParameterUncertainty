# For now run this ncar python env in the command line (or bash script)
# Not sure how to execute within python script:
#source /glade/work/kdagon/ncar_pylib_clone/bin/activate

# Finalize the 2-layer multiple output Neural Network
# 10/11/18

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
#np.random.seed(7)

# Read in input array
inputdata = np.load(file="lhc_100.npy")

# List of input variables
in_vars = ['medlynslope','dleaf','kmax','fff','dint','baseflow_scalar']

# Read in output array
outputdata_raw = np.load(file="outputdata/outputdata_GPP_SVD.npy")
# First 3 modes account for over 97% of variance
outputdata = outputdata_raw[:,:3]
nmodes = outputdata.shape[1]

# Percent of variance (for weighted avg R^2)
svd_var = [0.771, 0.1914, 0.0128]

# Create 2-layer simple model
model = Sequential()
# specify input_dim as number of parameters, not number of simulations
# l2 norm regularizer
model.add(Dense(8, input_dim=inputdata.shape[1], activation='relu',
    kernel_regularizer=l2(.001)))
# second layer with hyperbolic tangent activation
model.add(Dense(8, activation='tanh', kernel_regularizer=l2(.001)))
# output layer with linear activation
model.add(Dense(nmodes))

# Define model metrics
def mean_sq_err(y_true,y_pred):
    return K.mean((y_true-y_pred)**2)

# Compile model
opt_dense = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(opt_dense, "mse", metrics=[mean_sq_err])
#model.summary()

# Fit the model using ALL data
results = model.fit(inputdata, outputdata, epochs=500, batch_size=30, verbose=0)

# Make predictions
model_preds = model.predict(inputdata)

# model metric for predictions
def mse_preds(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)

# calculate model mean error with predictions
model_me = mse_preds(outputdata, model_preds)

# scatterplot actual versus predicted
plt.scatter(outputdata[:,0], model_preds[:,0])
plt.xlabel('CLM Model Output')
plt.ylabel('NN Predictions')
plt.xlim(np.amin([outputdata[:,0],model_preds[:,0]])-0.1,np.amax([outputdata[:,0],model_preds[:,0]])+0.1)
plt.ylim(np.amin([outputdata[:,0],model_preds[:,0]])-0.1,np.amax([outputdata[:,0],model_preds[:,0]])+0.1)
plt.savefig("validation_scatter_finalize_SVD_md_mode1_c4.pdf")
plt.show()
plt.scatter(outputdata[:,1], model_preds[:,1])
plt.xlabel('CLM Model Output')
plt.ylabel('NN Predictions')
plt.xlim(np.amin([outputdata[:,1],model_preds[:,1]])-0.1,np.amax([outputdata[:,1],model_preds[:,1]])+0.1)
plt.ylim(np.amin([outputdata[:,1],model_preds[:,1]])-0.1,np.amax([outputdata[:,1],model_preds[:,1]])+0.1)
plt.savefig("validation_scatter_finalize_SVD_md_mode2_c4.pdf")
plt.show()
plt.scatter(outputdata[:,2], model_preds[:,2])
plt.xlabel('CLM Model Output')
plt.ylabel('NN Predictions')
plt.xlim(np.amin([outputdata[:,2],model_preds[:,2]])-0.1,np.amax([outputdata[:,2],model_preds[:,2]])+0.1)
plt.ylim(np.amin([outputdata[:,2],model_preds[:,2]])-0.1,np.amax([outputdata[:,2],model_preds[:,2]])+0.1)
plt.savefig("validation_scatter_finalize_SVD_md_mode3_c4.pdf")
plt.show()

print("Model Mean Error: %.2g" % results.history['mean_sq_err'][-1])
print("Prediction Mean Error: %.2g" % model_me)

r_array = []
# linear regression of actual vs predicted
slope, intercept, r_value, p_value, std_err = stats.linregress(outputdata[:,0],
        model_preds[:,0])
print("r-squared: %.2g" % r_value**2)
r_array.append(r_value**2)
slope, intercept, r_value, p_value, std_err = stats.linregress(outputdata[:,1],
        model_preds[:,1])
print("r-squared: %.2g" % r_value**2)
r_array.append(r_value**2)
slope, intercept, r_value, p_value, std_err = stats.linregress(outputdata[:,2],
        model_preds[:,2])
print("r-squared: %.2g" % r_value**2)
r_array.append(r_value**2)

print("wgt avg. r-squared: %.2g" % np.average(r_array,weights=svd_var))
