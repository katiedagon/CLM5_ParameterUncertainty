# For now run this ncar python env in the command line (or bash script)
# Not sure how to execute within python script:
#source /glade/work/kdagon/ncar_pylib_clone/bin/activate

# NN with multiple output fields
# 6/26/18
# Update with SVD output 
# 9/12/18

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l2

import keras.backend as K

from scipy import stats

import numpy as np
import matplotlib.pyplot as plt

# Fix random seed for reproducibility
np.random.seed(9)

# Read in input array
inputdata = np.load(file="lhc_100.npy")

# List of input variables
in_vars = ['medlynslope','dleaf','kmax','fff','dint','baseflow_scalar']

# Read multi-dimensional output in .npy file
#outputdata_raw = np.load(file="outputdata/outputdata_GPP_SVD.npy")
#outputdata = outputdata_raw[:,:3]
# First 3 modes account for over 97% of variance
outputdata = np.load(file="outputdata/outputdata_GPP_SVD_3modes.npy")
nmodes = outputdata.shape[1]

# Second output variable
#outputdata_raw_2 = np.load(file="outputdata/outputdata_ET_SVD.npy")

# String together first two modes
#outputdata = np.column_stack((outputdata_raw[:,0], outputdata_raw_2[:,0]))
#nmodes = outputdata.shape[1]

# Read multi-dimensional output by stacking csv files
#out1 = np.loadtxt("outputdata_ET_IAV.csv")
#out2 = np.loadtxt("outputdata_GPP_IAV.csv")
#outputdata = np.column_stack([out1,out2])

# Create 2(3)-layer simple model
model = Sequential()
# first layer with x nodes and relu/linear activation
# specify input_dim as number of parameters, not number of simulations
# l2 norm regularizer
model.add(Dense(14, input_dim=inputdata.shape[1], 
    activation='relu', kernel_regularizer=l2(.001)))
# second layer with y nodes and hyperbolic tangent activation
model.add(Dense(9, activation='tanh', kernel_regularizer=l2(.001)))
# third layer with z nodes and sigmoid activation
#model.add(Dense(9, activation='relu', kernel_regularizer=l2(.001)))
# output layer with linear activation and nmodes outputs
model.add(Dense(nmodes))

# Define model metrics
def mean_sq_err(y_true,y_pred):
    return K.mean((y_true-y_pred)**2)

# Compile model
# RMSprop optimizer gives best performance
# loss function is mean squared error
opt_dense = RMSprop(lr=0.005, rho=0.9, epsilon=None, decay=0.0)
#opt_dense = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
#        amsgrad=False)
#opt_dense = SGD(lr=0.005, momentum=0.0, decay=0.0, nesterov=False)
model.compile(opt_dense, "mse", metrics=[mean_sq_err])
#model.summary()

# Separate training/test/val data: 60/20/20 split
x_train = inputdata[0:60]
x_test = inputdata[60:80]
x_val = inputdata[80:]
y_train = outputdata[0:60]
y_test = outputdata[60:80]
y_val = outputdata[80:]

# Fit the model
results = model.fit(x_train, y_train, epochs=500, batch_size=30,
        verbose=0, validation_data=(x_test,y_test))
#print(results)
print("Training Mean Error:", results.history['mean_sq_err'][-1])
print("Validation Mean Error:", results.history['val_mean_sq_err'][-1])

# Plot histogram of model mean error
#plt.hist(results.history['mean_sq_err'], bins=20)
#plt.xlabel('Training Mean Error')
#plt.ylabel('Counts')
#plt.savefig("dist_train_me.eps")
#plt.show()

# Plot histogram of validation mean error
#plt.hist(results.history['val_mean_sq_err'], bins=20)
#plt.xlabel('Validation Mean Error')
#plt.ylabel('Counts')
#plt.savefig("dist_val_me.eps")
#plt.show()

# Plot training history by epoch
plt.plot(results.epoch, results.history['val_mean_sq_err'], label='test')
plt.plot(results.epoch, results.history['mean_sq_err'], label='train')
plt.legend()
#plt.hlines(y=0,xmin=0,xmax=60)
plt.ylabel('Mean Squared Error')
plt.xlabel('Epoch')
plt.title('Neural Network Training History')
#plt.savefig("train_history_GPP-SVD.pdf")
#plt.savefig("train_history_GPP_ET_SVD-mode1.pdf")
plt.savefig("train_history_GPP_SVD_md.pdf")
plt.show()

# Save out model
#model.save('NN_multi-dim.h5')

# Make predictions using validation
model_train = model.predict(x_train)
model_test = model.predict(x_test)
model_preds = model.predict(x_val)
#print(model_preds.shape)
#print(model_preds)

# plot histogram of predictions
#plt.hist(model_preds, bins=10)
#plt.xlabel('Predictions')
#plt.ylabel('Counts') 
#plt.show()

# model metric for predictions
def mse_preds(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)

# calculate model mean error with predictions
model_me = mse_preds(y_val, model_preds)
print("Prediction Mean Error: ", model_me)

# plot histogram of prediction error
#plt.hist((y_val-model_preds)**2, bins=10)
#plt.xlabel('Prediction Error')
#plt.ylabel('Counts')
#plt.savefig("dist_preds.eps")
#plt.show()

# scatterplot actual versus predicted (validation set)
# first mode only
plt.scatter(y_val[:,0], model_preds[:,0], label='validation')
plt.scatter(y_train[:,0], model_train[:,0], label='training')
plt.scatter(y_test[:,0], model_test[:,0], label= 'testing')
plt.legend()
plt.xlabel('CLM Model Output')
plt.ylabel('NN Predictions')
plt.title('EOF1 GPP')
plt.xlim(np.amin([y_val[:,0],model_preds[:,0]])-0.1,np.amax([y_val[:,0],model_preds[:,0]])+0.1)
plt.ylim(np.amin([y_val[:,0],model_preds[:,0]])-0.1,np.amax([y_val[:,0],model_preds[:,0]])+0.1)
#plt.savefig("validation_scatter_GPP-SVD-mode1.pdf")
#plt.savefig("validation_scatter_GPP_ET_SVD-mode1GPP.pdf")
plt.savefig("validation_scatter_training_GPP_SVD_md_mode1.pdf")
plt.show()
# mode 2
plt.scatter(y_val[:,1], model_preds[:,1], label='validation')
plt.scatter(y_train[:,1], model_train[:,1], label='training')
plt.scatter(y_test[:,1], model_test[:,1], label='testing')
plt.legend()
plt.xlabel('CLM Model Output')
plt.ylabel('NN Predictions')
plt.title('EOF2 GPP')
plt.xlim(np.amin([y_val[:,1],model_preds[:,1]])-0.1,np.amax([y_val[:,1],model_preds[:,1]])+0.1)
plt.ylim(np.amin([y_val[:,1],model_preds[:,1]])-0.1,np.amax([y_val[:,1],model_preds[:,1]])+0.1)
#plt.savefig("validation_scatter_GPP-SVD-mode2.pdf")
#plt.savefig("validation_scatter_GPP_ET_SVD-mode1ET.pdf")
plt.savefig("validation_scatter_training_GPP_SVD_md_mode2.pdf")
plt.show()
# mode 3
plt.scatter(y_val[:,2], model_preds[:,2], label='validation')
plt.scatter(y_train[:,2], model_train[:,2], label='training')
plt.scatter(y_test[:,2], model_test[:,2], label='testing')
plt.legend()
plt.xlabel('CLM Model Output')
plt.ylabel('NN Predictions')
plt.title('EOF3 GPP')
plt.xlim(np.amin([y_val[:,2],model_preds[:,2]])-0.1,np.amax([y_val[:,2],model_preds[:,2]])+0.1)
plt.ylim(np.amin([y_val[:,2],model_preds[:,2]])-0.1,np.amax([y_val[:,2],model_preds[:,2]])+0.1)
#plt.savefig("validation_scatter_GPP-SVD-mode3.pdf")
plt.savefig("validation_scatter_training_GPP_SVD_md_mode3.pdf")
plt.show()

# linear regression of actual vs predicted
# reshape into a single vector?
#y_test_reshape = np.reshape(y_test, 20*2)
#model_preds_reshape = np.reshape(model_preds, 20*2)
#slope, intercept, r_value, p_value, std_err = stats.linregress(y_test_reshape,
#        model_preds_reshape)
# Mode 1
slope, intercept, r_value, p_value, std_err = stats.linregress(y_val[:,0],
        model_preds[:,0])
print("Mode 1 prediction r-squared:", r_value**2)
slope, intercept, r_value, p_value, std_err = stats.linregress(y_train[:,0],
                model_train[:,0])
print("Mode 1 training r-squared:", r_value**2)
slope, intercept, r_value, p_value, std_err = stats.linregress(y_test[:,0],
                        model_test[:,0])
print("Mode 1 testing r-squared:", r_value**2)

# Mode 2
slope, intercept, r_value, p_value, std_err = stats.linregress(y_val[:,1],
                model_preds[:,1])
print("Mode 2 prediction r-squared:", r_value**2)
slope, intercept, r_value, p_value, std_err = stats.linregress(y_train[:,1],
                        model_train[:,1])
print("Mode 2 training r-squared:", r_value**2)
slope, intercept, r_value, p_value, std_err = stats.linregress(y_test[:,0],
                                model_test[:,0])
print("Mode 2 testing r-squared:", r_value**2)

# Mode 3
slope, intercept, r_value, p_value, std_err = stats.linregress(y_val[:,2],
                        model_preds[:,2])
print("Mode 3 prediction r-squared:", r_value**2)
slope, intercept, r_value, p_value, std_err = stats.linregress(y_train[:,2],
                        model_train[:,2])
print("Mode 3 training r-squared:", r_value**2)
slope, intercept, r_value, p_value, std_err = stats.linregress(y_test[:,2],
                                model_test[:,2])
print("Mode 3 testing r-squared:", r_value**2)
