# For now run this ncar python env in the command line (or bash script)
# Not sure how to execute within python script:
#source /glade/p/work/kdagon/ncar_pylib_clone/bin/activate

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from keras.regularizers import l2

import keras.backend as K

from scipy import stats

import numpy as np
import matplotlib.pyplot as plt

# Fix random seed for reproducibility
#np.random.seed(7) # ET, GPP IAV output
np.random.seed(3) # GM ET, TSA output

# Read in input array
inputdata = np.load(file="lhc_100.npy")

# List of input variables
in_vars = ['medlynslope','dleaf','kmax','fff','dint','baseflow_scalar']

# Read in output array
#outputdata = np.loadtxt("outputdata_ET_IAV.csv")
#outputdata = np.loadtxt("outputdata_ET.csv")
#outputdata = np.loadtxt("outputdata_TSA.csv")
#outputdata = np.loadtxt("outputdata_GPP_IAV.csv")

# multi-dimensional output
out1 = np.loadtxt("outputdata_ET_IAV.csv")
out2 = np.loadtxt("outputdata_GPP_IAV.csv")
outputdata = np.column_stack([out1,out2])

# histogram of 1D output
#plt.hist(outputdata, bins=20)
#plt.hist(outputdata, bins=10) #TSA
#plt.xlabel('Global Mean 5-year ET IAV (mm/yr)')
#plt.xlabel('Global Mean ET (mm/yr)')
#plt.xlabel('Global Mean Temp (C)')
#plt.xlabel('Global Mean 5-year GPP IAV (umol/m2s)')
#plt.ylabel('Counts')
#plt.savefig("dist_outputdata_ET-IAV.eps")
#plt.savefig("dist_outputdata_ET.eps")
#plt.savefig("dist_outputdata_TSA.eps")
#plt.savefig("dist_outputdata_GPP-IAV.eps")
#plt.show()

# Create 2-layer simple model
model = Sequential()
# first layer with 20 nodes and rectified linear activation
# specify input_dim as number of parameters, not number of simulations
# l2 norm regularizer
model.add(Dense(20, input_dim=inputdata.shape[1], 
    activation='relu', kernel_regularizer=l2(.001)))
# output layer with linear activation
model.add(Dense(2))

# Define model metrics
#def mean_error(y_true,y_pred):
#    return K.mean(y_true-y_pred)
def mean_sq_err(y_true,y_pred):
    return K.mean((y_true-y_pred)**2)

# Compile model
# using a stochastic gradient descent optimizer
# loss function is mean squared error
opt_dense = SGD(lr=0.001, momentum=0.99, decay=1e-4, nesterov=True)
model.compile(opt_dense, "mse", metrics=[mean_sq_err])
#model.summary()

# Separate training/test data: 60/40 split
x_train = inputdata[0:60,:]
x_test = inputdata[60:,:]
y_train = outputdata[0:60]
y_test = outputdata[60:]

# Fit the model
results = model.fit(x_train, y_train, epochs=150, batch_size=30,
        validation_data=(x_test,y_test))
#print(results)
print("Training Mean Error:", results.history['mean_sq_err'][-1])
print("Validation Mean Error:", results.history['val_mean_sq_err'][-1])

# Plot histogram of model mean error
plt.hist(results.history['mean_sq_err'], bins=20)
plt.xlabel('Training Mean Error')
plt.ylabel('Counts')
#plt.savefig("dist_train_me.eps")
plt.show()

# Plot histogram of validation mean error
plt.hist(results.history['val_mean_sq_err'], bins=20)
plt.xlabel('Validation Mean Error')
plt.ylabel('Counts')
#plt.savefig("dist_val_me.eps")
plt.show()

# Plot training history by epoch
plt.plot(results.epoch, results.history['val_mean_sq_err'], label='validation')
plt.plot(results.epoch, results.history['mean_sq_err'], label='train')
plt.legend()
plt.hlines(y=0,xmin=0,xmax=150)
plt.ylabel('Mean Error')
plt.xlabel('Epoch')
plt.title('Neural Network Training History')
#plt.savefig("train_history_ET-IAV.eps")
#plt.savefig("train_history_ET.eps")
#plt.savefig("train_history_TSA.eps")
#plt.savefig("train_history_GPP-IAV.eps")
plt.show()

# Evaluate the model using test data
score = model.evaluate(x_test, y_test, batch_size=10)
#print(score)

# Make predictions using validation
#model_preds = model.predict(x_test)[:,0]
model_preds = model.predict(x_test)
#print(model_preds)

# plot histogram of predictions
plt.hist(model_preds, bins=10)
plt.xlabel('Predictions')
plt.ylabel('Counts') 
plt.show()

# model metric for predictions
def model_error_preds(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)

# calculate model mean error with predictions
model_me = model_error_preds(y_test, model_preds)
print("Prediction Mean Error: ", model_me)

# plot histogram of prediction error
plt.hist(y_test-model_preds, bins=10)
plt.xlabel('Prediction Error')
plt.ylabel('Counts')
#plt.savefig("dist_preds.eps")
plt.show()

# scatterplot actual versus predicted (validation set)
plt.scatter(y_test, model_preds)
#plt.xlabel('CLM Output ET IAV')
#plt.ylabel('NN Predicted ET IAV')
plt.xlabel('CLM Output')
plt.ylabel('NN Predicted')
#plt.xlim(np.amin([y_test,model_preds])-0.01,np.amax([y_test,model_preds])+0.01)
#plt.ylim(np.amin([y_test,model_preds])-0.01,np.amax([y_test,model_preds])+0.01)
#plt.xlim(np.amin([y_test,model_preds])-0.1,np.amax([y_test,model_preds])+0.1)
#plt.ylim(np.amin([y_test,model_preds])-0.1,np.amax([y_test,model_preds])+0.1)
#plt.xlim(np.amin([y_test,model_preds])-1,np.amax([y_test,model_preds])+1)
#plt.ylim(np.amin([y_test,model_preds])-1,np.amax([y_test,model_preds])+1)
plt.xlim(np.amin([y_test,model_preds])-10,np.amax([y_test,model_preds])+10)
plt.ylim(np.amin([y_test,model_preds])-10,np.amax([y_test,model_preds])+10)
#plt.savefig("validation_scatter_ET-IAV.eps")
#plt.savefig("validation_scatter_ET.eps")
#plt.savefig("validation_scatter_TSA.eps")
#plt.savefig("validation_scatter_GPP-IAV.eps")
plt.show()

# linear regression of actual vs predicted
slope, intercept, r_value, p_value, std_err = stats.linregress(y_test,
        model_preds)
print("r-squared:", r_value**2)

