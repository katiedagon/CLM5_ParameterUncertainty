# Testing out different NN optimizers
# 10/23/19

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

import csv

# Fix random seed for reproducibility
#np.random.seed(9)

# Read in input array
inputdata = np.load(file="lhc_100.npy", allow_pickle=True)

# Read in output array
#outputdata_all = np.load("outputdata/outputdata_GPP_SVD.npy")
#outputdata = np.load("outputdata/outputdata_GPP_SVD_3modes.npy")
#outputdata = np.load("outputdata/outputdata_LHF_SVD_3modes.npy")
#outputdata = np.load("outputdata/outputdata_GPP_SVD_3modes_fc.npy")
#outputdata = np.load("outputdata/outputdata_GPP_SVD_3modes_diff.npy")
#outputdata = np.load("outputdata/outputdata_LHF_SVD_3modes_diff.npy")

# Training to predict global mean diff GPP, LHF
var = "GPP"
f=nc.netcdf_file("outputdata/outputdata_"+var+"_GM_100_diff.nc",'r', mmap=False) 
X = f.variables[var]
outputdata = X[:]

# Separate training/test/val data: 60/20/20 split
x_train = inputdata[0:60]
x_test = inputdata[60:80]
x_val = inputdata[80:]
y_train = outputdata[0:60]
y_test = outputdata[60:80]
y_val = outputdata[80:]

# Set # of nodes
firstn = 5
secondn = 15

# Define model metrics
def mean_sq_err(y_true,y_pred):
    return K.mean((y_true-y_pred)**2)

# fit a model and plot learning curve
# note that optimizers will use their default values
def fit_model(trainX, trainY, testX, testY, optimizer):
    # define model
    model = Sequential()
    model.add(Dense(firstn, input_dim=inputdata.shape[1], activation='relu',
        kernel_regularizer=l2(.001)))
    model.add(Dense(secondn, activation='tanh', kernel_regularizer=l2(.001)))
    model.add(Dense(1))
    # compile model
    model.compile(optimizer=optimizer, loss="mse", metrics=[mean_sq_err])
    # fit model
    #results = model.fit(trainX, trainY, epochs=500, batch_size=30,
    #        verbose=0, validation_data=(testX,testY))
    # fit model with early stopping
    es = EarlyStopping(monitor='val_loss', min_delta=1,
            patience=50, verbose=1, mode='min')
    results = model.fit(trainX, trainY, epochs=5000, batch_size=30,
            callbacks=[es], verbose=0, validation_data=(testX,testY))
    maxep = max(results.epoch)
    # plot learning curves
    plt.plot(results.history['mean_sq_err'], label='train')
    plt.plot(results.history['val_mean_sq_err'], label='test')
    #plt.ylabel("Mean Squared Error") 
    #plt.xlabel("Epochs")
    plt.title('opt='+optimizer)
    # plot learning curve (last half epochs only)
    #halfep = int(maxep/2) # get ~halfway pt
    #plt.plot(results.epoch[halfep:], 
    #        results.history['mean_sq_err'][halfep:],
    #        label='train')
    #plt.plot(results.epoch[halfep:],
    #        results.history['val_mean_sq_err'][halfep:],
    #        label='test')
    #plt.ylabel("Mean Squared Error")
    #plt.xlabel("Epochs")
    #plt.title('opt='+optimizer)

# create learning curves for different optimizers
momentums = ['sgd', 'rmsprop', 'adagrad', 'adam', 'adadelta']

# set up plots
plt.figure(figsize=(8,10))                                                                                                                
plt.subplots_adjust(wspace=0.2,hspace=0.5)  

# Loop over optimizers
for i in range(len(momentums)):
    np.random.seed(9)
    # determine the plot number
    plot_no = 320 + (i+1)
    plt.subplot(plot_no)
    fit_model(x_train,y_train,x_test,y_test,momentums[i])
    if i == 0:
        plt.xlabel("Epochs")
        plt.ylabel("Mean Squared Error")
        plt.legend()

#plt.savefig("tune_opt_"+var+"_GM_diff.pdf")
plt.savefig("tune_opt_es_"+var+"_GM_diff.pdf")
#plt.savefig("tune_opt_lasteps_"+var+"_GM_diff.pdf")
#plt.savefig("tune_opt_lasteps_es_"+var+"_GM_diff.pdf")
plt.show()



