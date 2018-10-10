# For now run this ncar python env in the command line (or bash script)
# Not sure how to execute within python script:
#source /glade/work/kdagon/ncar_pylib_clone/bin/activate

# Testing out different NN configurations
# Focusing on 1 or 2 hidden layers
# Number of nodes ranging from 1 to 10 in each layer
# 8/15/18

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

import csv

# Fix random seed for reproducibility
#np.random.seed(7)

# Read in input array
inputdata = np.load(file="lhc_100.npy")

# Read in output array
#outputdata = np.loadtxt("outputdata/outputdata_GPP.csv")
outputdata_all = np.load("outputdata/outputdata_GPP_SVD.npy")

# Specify mode (SVD only)
mode = 1
outputdata = outputdata_all[:,mode-1]

# Separate training/test/val data: 60/20/20 split
x_train = inputdata[0:60,:]
x_test = inputdata[60:80,:]
x_val = inputdata[80:,:]
y_train = outputdata[0:60]
y_test = outputdata[60:80]
y_val = outputdata[80:]

# Loop over # of nodes
metrics=[]
# First layer
for i in range(1,11):
    # Second layer
    for j in range(1,11):

        print(i,j)

        # Random seed for reproducibility
        # Still not sure why this need to be specified for each loop
        # But otherwise, the answers change from running the script 
        # separately each time for a different node value
        np.random.seed(7)

        # Create 2-layer simple model
        model = Sequential()
        # first hidden layer with variable # nodes and relu or linear activation
        # specify input_dim as number of parameters, not number of simulations
        # l2 norm regularizer
        model.add(Dense(i, input_dim=inputdata.shape[1], activation='linear',
            kernel_regularizer=l2(.001)))
        # second layer with varible #  nodes and hyperbolic tangent activation
        model.add(Dense(j, activation='tanh', kernel_regularizer=l2(.001)))
        # output layer with linear activation
        model.add(Dense(1))

        # Define model metrics
        def mean_sq_err(y_true,y_pred):
            return K.mean((y_true-y_pred)**2)

        # Compile model
        # using RMSprop optimizer
        opt_dense = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
        model.compile(opt_dense, "mse", metrics=[mean_sq_err])
        #model.summary()

        # Fit the model
        results = model.fit(x_train, y_train, epochs=500, batch_size=30,
            verbose=0, validation_data=(x_test,y_test))

        # Make predictions - using validation set
        model_preds = model.predict(x_val)[:,0]
        model_test = model.predict(x_test)[:,0]
        model_train = model.predict(x_train)[:,0]

        # model metric for predictions
        def mse_preds(y_true,y_pred):
            return np.mean((y_true-y_pred)**2)

        # calculate model mean error with predictions
        model_me = mse_preds(y_val, model_preds)

        # linear regression of actual vs predicted
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_val,
            model_preds)
        metrics.append([results.history['mean_sq_err'][-1],
            results.history['val_mean_sq_err'][-1], model_me, r_value**2])

# Different formatting for printing out metrics (2 sig figs)
#metricsFormat = [["%.2f" % m for m in msub] for msub in metrics]
#metricsFormat = [[round(m,2) for m in msub] for msub in metrics]
metricsFormat = [["%.2g" % m for m in msub] for msub in metrics]
#print(metricsFormat)

# Write out metric data to csv
with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(metricsFormat)
