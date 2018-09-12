# For now run this ncar python env in the command line (or bash script)
# Not sure how to execute within python script:
#source /glade/work/kdagon/ncar_pylib_clone/bin/activate

# Developing the 2-layer single output Neural Network
# 6/29/18

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

# transform input values to match output (not necessary)
#inputdata = np.log(inputdata.astype(np.float64))
#inputdata = inputdata**10

# input image plot
#inputfloat=inputdata.astype(float)
#plt.imshow(inputfloat)
#plt.colorbar()
#plt.savefig("img_inputdata_log.eps")                                                                                                             
#plt.show()

# List of input variables
in_vars = ['medlynslope','dleaf','kmax','fff','dint','baseflow_scalar']

# Read in output array
# use NCL script to generate global mean GPP array 
# from 100-member ensemble in python readable format
outputdata = np.loadtxt("outputdata/outputdata_GPP.csv")
#outputdata = np.loadtxt("outputdata_ET_IAV.csv")
#print(outputdata)

# transform GPP to reduce left skew
#outputdata = outputdata**10
#outputdata = np.log(outputdata)
#outputdata = 1/outputdata

#plt.hist(outputdata, bins=20)
#plt.xlabel('Global Mean GPP (umol/m2s)')
#plt.xlabel('log(Global Mean GPP)')
#plt.xlabel('inv(Global Mean GPP)')
#plt.xlabel('Global Mean GPP^10')
#plt.xlabel('Global Mean 5-year ET IAV (mm/yr)')
#plt.ylabel('Counts')
#plt.savefig("dist_outputdata_GPP.eps")
#plt.savefig("dist_outputdata_logGPP.eps")
#plt.show()

# Create 2-layer simple model
model = Sequential()
# first layer with 4 nodes and rectified linear activation
# specify input_dim as number of parameters, not number of simulations
# l2 norm regularizer
model.add(Dense(4, input_dim=inputdata.shape[1], activation='relu',
    kernel_regularizer=l2(.001)))
# second layer with 7 nodes and hyperbolic tangent activation
model.add(Dense(7, activation='tanh', kernel_regularizer=l2(.001)))
# output layer with linear activation
model.add(Dense(1))
#model.add(Dense(1, activation='relu')) 
#force positive output for log-transformed input data

# Define model metrics
def mean_sq_err(y_true,y_pred):
    return K.mean((y_true-y_pred)**2)

# Compile model
# using a stochastic gradient descent optimizer
#opt_dense = SGD(lr=0.001, momentum=0.99, decay=1e-4, nesterov=True)
opt_dense = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(opt_dense, "mse", metrics=[mean_sq_err])
model.summary()

# Separate training/test/val data: 60/20/20 split
x_train = inputdata[0:60,:]
x_test = inputdata[60:80,:]
x_val = inputdata[80:,:]
y_train = outputdata[0:60]
y_test = outputdata[60:80]
y_val = outputdata[80:]

# Two sets only (train/val, 60/40):
#x_train = inputdata[0:60,:]
#x_val = inputdata[60:,:]
#y_train = outputdata[0:60]
#y_val = outputdata[60:]

# Fit the model
#results = model.fit(x_train, y_train, epochs=60, batch_size=30,
#        validation_data=(x_test,y_test))
#results = model.fit(x_train, y_train, epochs=38, batch_size=30,
#        validation_data=(x_test,y_test))
results = model.fit(x_train, y_train, epochs=500, batch_size=30,
        verbose=0, validation_data=(x_test,y_test))
# fit with stopping criteria when val_loss starts increasing
#es = EarlyStopping(monitor='val_loss', min_delta=0, 
#        patience=0, verbose=1, mode='auto')
#results = model.fit(x_train, y_train, epochs=60, batch_size=30,
#        callbacks=[es], validation_data=(x_test,y_test)) 

print("Training Mean Error:", results.history['mean_sq_err'][-1])
print("Validation Mean Error:", results.history['val_mean_sq_err'][-1])

# Plot histogram of training mean error
#plt.hist(results.history['mean_sq_err'], bins=20)
#plt.xlabel('Training Mean Error')
#plt.ylabel('Counts')
#plt.savefig("dist_train_me.eps")
#plt.show()

# Plot histogram of validation (testing) mean error
#plt.hist(results.history['val_mean_sq_err'], bins=20)
#plt.xlabel('Test Mean Error')
#plt.ylabel('Counts')
#plt.savefig("dist_test_me.eps")
#plt.show()

# Plot training history by epoch
plt.plot(results.epoch, results.history['val_mean_sq_err'], label='test')
plt.plot(results.epoch, results.history['mean_sq_err'], label='train')
#plt.xticks(results.epoch)
plt.legend()
#plt.hlines(y=0,xmin=0,xmax=15)
#plt.hlines(y=0,xmin=0,xmax=40)
plt.ylabel('Mean Squared Error')
plt.xlabel('Epoch')
plt.title('Neural Network Training History')
#plt.savefig("train_history_logGPP.eps")
#plt.savefig("train_history_ET-IAV_v3.eps")
#plt.savefig("train_history_logGPP_v2.eps")
#plt.savefig("train_history_logGPP_v3.eps")
#plt.savefig("train_history_logGPP_nonlinear.eps")
#plt.savefig("train_history_RMSprop.eps")
plt.show()

# Make predictions - using validation set
model_preds = model.predict(x_val)[:,0]
model_test = model.predict(x_test)[:,0]
model_train = model.predict(x_train)[:,0]
print(model_preds.shape)
test = model.predict(x_val)
print(test.shape)

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
#plt.scatter(y_val, model_preds)
plt.scatter(y_val, model_preds, label='validation')
plt.scatter(y_train, model_train, label='train')
plt.scatter(y_test, model_test, label='test')
plt.legend()
plt.xlabel('CLM Model Output')
plt.ylabel('NN Predictions')
plt.xlim(np.amin([y_val,model_preds])-0.1,np.amax([y_val,model_preds])+0.1)
plt.ylim(np.amin([y_val,model_preds])-0.1,np.amax([y_val,model_preds])+0.1)
#plt.xlim(np.amin([y_val,model_preds])-0.5,np.amax([y_val,model_preds])+0.5)                                                                   
#plt.ylim(np.amin([y_val,model_preds])-0.5,np.amax([y_val,model_preds])+0.5)
# trying to get a 1:1 line to show up (still not working!!)
#x1,x2=ax.Axes.get_xlim()
#y1,y2=ax.Axes.get_ylim()
#plt.plot([x1,x2], [y1,y2], ls="--", c=".3")
#plt.savefig("validation_scatter_logGPP.eps")
#plt.savefig("validation_scatter_ET-IAV_v3.eps")
#plt.savefig("validation_scatter_logGPP_v2.eps")
#plt.savefig("validation_scatter_logGPP_v3.eps")
#plt.savefig("validation_scatter_logGPP_nonlinear.eps")
#plt.savefig("validation_scatter_RMSprop.eps")
#plt.show()

# linear regression of actual vs predicted
slope, intercept, r_value, p_value, std_err = stats.linregress(y_val,
                model_preds)
print("r-squared:", r_value**2)


# scatterplot input/output for kmax
#plt.scatter(inputdata[:,2],outputdata)
#plt.ylabel('CLM Output')
#plt.xlabel('LHC Values')
#plt.title('kmax')
#plt.show()

# scatterplots of NN vs input
# this works because NN predictions can be 
# stiched back together in the same order
# (no resampling)
total_preds = np.append(np.append(model_train,model_test),model_preds)
for x, y in enumerate(in_vars):
    plt.scatter(inputdata[:,x], total_preds)
    plt.ylabel('NN Predictions') 
    plt.xlabel('LHC Values')
    plt.title(y)
    #plt.savefig(y)
    plt.show()


