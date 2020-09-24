# Testing out different NN configurations

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import keras.backend as K

from scipy import stats
import numpy as np
import csv

# Read in input array
inputdata = np.load(file="../lhc_100.npy", allow_pickle=True)

# Read in output array
#outputdata = np.load("../outputdata/outputdata_GPP_SVD_3modes.npy")
outputdata = np.load("../outputdata/outputdata_LHF_SVD_3modes.npy")

# Multi-dimension
nmodes = outputdata.shape[1]

# Separate training/test/val data: 60/20/20 split
x_train = inputdata[0:60]
x_test = inputdata[60:80]
x_val = inputdata[80:]
y_train = outputdata[0:60]
y_test = outputdata[60:80]
y_val = outputdata[80:]

# Max # of nodes
maxnode = 15

# Min # of nodes
minnode = 5

# Loop over # of nodes
metrics=[]
#eps = []

# First layer
for i in range(minnode,maxnode+1):
    # Second layer
    for j in range(minnode,maxnode+1):

        print("Node configuration:")
        print(i,j)

        # Random seed for reproducibility
        np.random.seed(9)

        # Create 2-layer simple model
        model = Sequential()
        model.add(Dense(i, input_dim=inputdata.shape[1], activation='relu',
        #model.add(Dense(i, input_dim=inputdata.shape[1], activation='tanh',
            kernel_regularizer=l2(.001)))
        model.add(Dense(j, activation='tanh', kernel_regularizer=l2(.001)))
        model.add(Dense(nmodes))

        # Define model metrics
        def mean_sq_err(y_true,y_pred):
            return K.mean((y_true-y_pred)**2)

        # Compile model
        opt_dense = RMSprop(lr=0.005, rho=0.9, epsilon=None, decay=0.0)
        model.compile(opt_dense, "mse", metrics=[mean_sq_err])

        # Fit the model w/ EarlyStopping
        #es = EarlyStopping(monitor='val_loss', min_delta=1,
        #        patience=50, verbose=1, mode='min')
        results = model.fit(x_train, y_train, epochs=500, batch_size=30,
        #        callbacks=[es], verbose=0, validation_data=(x_test,y_test))
            verbose=0, validation_data=(x_test,y_test))

        # Make predictions - using validation set
        model_preds = model.predict(x_val)
        print("Y_pred shape:")
        print(model_preds.shape)

        # model metric for predictions
        def mse_preds(y_true,y_pred):
            return np.mean((y_true-y_pred)**2)

        # calculate model mean error with predictions
        model_me = mse_preds(y_val, model_preds)
        print("MSE=")
        print(model_me)

        # linear regression of actual vs predicted
        r_array = []
        for k in range(0,nmodes):
            slope, intercept, r_value, p_value, std_err = stats.linregress(y_val[:,k],model_preds[:,k])
            r_array.append(r_value**2)

        # save out total epochs
        #eps.append(max(results.epoch)+1)

        # save metrics - multi-dim
        metrics.append([results.history['mean_sq_err'][-1],
            results.history['val_mean_sq_err'][-1],model_me,r_array[0],
            r_array[1],r_array[2]])

# Different formatting for printing out metrics (2 sig figs)
metricsFormat = [["%.4f" % m for m in msub] for msub in metrics]

# Write out metric data to csv
with open('NN_test.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(metricsFormat)

# Write out epochs to txt
#epsFormat = ["%d" % e for e in eps]
#with open('NN_test_eps.txt', 'w') as f:
#    for i,e in enumerate(epsFormat):
#        f.write(epsFormat[i]+'\n')
