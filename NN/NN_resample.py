# Resampling train/test/val
# For the best 2-layer Neural Network configurations
# 8/16/18

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

import keras.backend as K

from scipy import stats
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from scipy.io import netcdf as nc

# Fix random seed for reproducibility
#np.random.seed(7)

# Read in input array
inputdata = np.load(file="lhc_100.npy", allow_pickle=True)

# Read in output array
#outputdata = np.loadtxt("outputdata/outputdata_GPP.csv")
#outputdata_all = np.load("outputdata/outputdata_GPP_SVD.npy")
#outputdata = np.load("outputdata/outputdata_GPP_SVD_3modes.npy")
#outputdata = np.load("outputdata/outputdata_LHF_SVD_3modes.npy")
#outputdata = np.load("outputdata/outputdata_GPP_SVD_3modes_diff.npy")
#outputdata = np.load("outputdata/outputdata_LHF_SVD_3modes_diff.npy")

# Training to predict global mean GPP, LHF
#var = "GPP"
var = "LHF"
f=nc.netcdf_file("outputdata/outputdata_"+var+"_GM_100_diff.nc",'r', mmap=False)
X = f.variables[var]
outputdata = X[:]

# Specify mode (SVD only)
#mode = 3
#outputdata = outputdata_all[:,mode-1]

# Multi-dimension
#outputdata = outputdata_all[:,:3]
#nmodes = outputdata.shape[1]

# Percent of variance (for weighted avg R^2)
#svd_var = [0.8341, 0.1349, 0.0119] # GPP
#svd_var = [0.7701996, 0.12915632, 0.05642754] #LHF
#svd_var = [0.43263328, 0.19826488, 0.13297316] # GPP Diff
#svd_var = [0.49752492, 0.14868388, 0.11127292] #LHF Diff

metricsME = []
metricsRsq = []
#metricswRsq = []
eps = []
# Resample k times
samps = 100
#samps = 10
for k in range(1,samps+1):
    print(k)

    # Separate training/test/val data: 60/20/20 split
    # Randomly using sklearn
    x,x_test,y,y_test = train_test_split(inputdata,outputdata,test_size=0.2,train_size=0.8)
    x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.25,train_size=0.75)

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
    #model.add(Dense(nmodes))

    # Define model metrics
    def mean_sq_err(y_true,y_pred):
        return K.mean((y_true-y_pred)**2)

    # Compile model
    opt_dense = RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)
    #opt_dense = SGD(lr=0.001, momentum=0.99, decay=1e-4, nesterov=True)
    model.compile(opt_dense, "mse", metrics=[mean_sq_err])
    #model.summary()

    # Fit the model
    #results = model.fit(x_train, y_train, epochs=500, batch_size=30,
    #    verbose=0, validation_data=(x_test,y_test))
    #print("Training Mean Error:", results.history['mean_sq_err'][-1])
    #print("Validation Mean Error:", results.history['val_mean_sq_err'][-1])

    # Fit the model w/ EarlyStopping
    es = EarlyStopping(monitor='val_loss', min_delta=1,
        patience=50, verbose=0, mode='min')
    results = model.fit(x_train, y_train, epochs=500,batch_size=20,
        callbacks=[es], verbose=0, validation_data=(x_test,y_test))

    # Plot training history by epoch
    #plt.plot(results.epoch, results.history['val_mean_sq_err'], label='test')
    #plt.plot(results.epoch, results.history['mean_sq_err'], label='train')
    #plt.legend()
    #plt.ylabel('Mean Squared Error')
    #plt.xlabel('Epoch')
    #plt.title('Neural Network Training History')
    #plt.savefig("train_history_RMSprop.eps")
    #plt.show()

    # Make predictions - using validation set (single dim)
    model_preds = model.predict(x_val)[:,0]
    #model_test = model.predict(x_test)[:,0]
    #model_train = model.predict(x_train)[:,0]

    # Predictions - multi-dim
    #model_preds = model.predict(x_val)
    #model_test = model.predict(x_test)
    #model_train = model.predict(x_train)

    # model metric for predictions
    def mse_preds(y_true,y_pred):
        return np.mean((y_true-y_pred)**2)

    # calculate model mean error with predictions
    model_me = mse_preds(y_val, model_preds)
    #print("Prediction Mean Error: ", model_me)

    # scatterplot actual versus predicted (validation set)
    #plt.scatter(y_val, model_preds, label='validation')
    #plt.scatter(y_train, model_train, label='train')
    #plt.scatter(y_test, model_test, label='test')
    #plt.legend()
    #plt.xlabel('CLM Model Output')
    #plt.ylabel('NN Predictions')
    #plt.xlim(np.amin([y_val,model_preds])-0.1,np.amax([y_val,model_preds])+0.1)
    #plt.ylim(np.amin([y_val,model_preds])-0.1,np.amax([y_val,model_preds])+0.1)
    #plt.xlim(np.amin([y_val,model_preds])-0.5,np.amax([y_val,model_preds])+0.5)                                                                   
    #plt.ylim(np.amin([y_val,model_preds])-0.5,np.amax([y_val,model_preds])+0.5)
    #plt.savefig("validation_scatter_RMSprop.eps")
    #plt.show()

    # linear regression of actual vs predicted - single dim
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_val,model_preds)

    # linear regression - multi-dim
    #r_array = []
    #for k in range(0,nmodes):
        #print(k)
    #    slope, intercept, r_value, p_value, std_err = stats.linregress(y_val[:,k],
    #            model_preds[:,k])
    #    r_array.append(r_value**2)

    #print("Prediction Mean Error: %.2g" % model_me)
    #print("r-squared: %.2g" % r_value**2)
    #print("avg. r-squared: %.2g" % np.mean(r_array))
    #print("wgt avg. r-squared: %.2g" % np.average(r_array,weights=svd_var))
    metricsME.append(model_me)
    metricsRsq.append(r_value**2) # single dim
    #metricsRsq.append(np.mean(r_array)) # multi dim, average
    #metricswRsq.append(np.average(r_array,weights=svd_var)) # multi dim, wgt avg
    
    # save number of epochs
    eps.append(max(results.epoch)+1)

print("Mean Validation MSE:")
#print("%.2f (+/- %.2f)" % (np.mean(metricsME), np.std(metricsME)))
print("%.4f (+/- %.4f)" % (np.mean(metricsME), np.std(metricsME)))
print("Mean Validation r^2:")
print("%.2f (+/- %.2f)" % (np.mean(metricsRsq), np.std(metricsRsq)))
#print("%.4f (+/- %.4f)" % (np.mean(metricsRsq), np.std(metricsRsq)))
#print("%.2g (+/- %.2g)" % (np.mean(metricswRsq), np.std(metricswRsq)))
print("Mean Number of Epochs:")
print("%d (+/- %.2f)" % (np.mean(eps), np.std(eps)))
