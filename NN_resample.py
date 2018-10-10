# For now run this ncar python env in the command line (or bash script)
# Not sure how to execute within python script:
#source /glade/work/kdagon/ncar_pylib_clone/bin/activate

# Resampling train/test/val
# For the best 2-layer single output Neural Network
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

metricsME = []
metricsRsq = []
# Resample 10 times
for k in range(1,11):

    # Separate training/test/val data: 60/20/20 split
    # Randomly using sklearn
    x,x_test,y,y_test = train_test_split(inputdata,outputdata,test_size=0.2,train_size=0.8)
    x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.25,train_size=0.75)

    # Create 2-layer simple model
    model = Sequential()
    # specify input_dim as number of parameters, not number of simulations
    # l2 norm regularizer
    model.add(Dense(8, input_dim=inputdata.shape[1], activation='relu',
        kernel_regularizer=l2(.001)))
    # second layer with hyperbolic tangent activation
    model.add(Dense(9, activation='tanh', kernel_regularizer=l2(.001)))
    # output layer with linear activation
    model.add(Dense(1))

    # Define model metrics
    def mean_sq_err(y_true,y_pred):
        return K.mean((y_true-y_pred)**2)

    # Compile model
    opt_dense = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(opt_dense, "mse", metrics=[mean_sq_err])
    #model.summary()

    # Fit the model
    results = model.fit(x_train, y_train, epochs=500, batch_size=30,
        verbose=0, validation_data=(x_test,y_test))
    #print("Training Mean Error:", results.history['mean_sq_err'][-1])
    #print("Validation Mean Error:", results.history['val_mean_sq_err'][-1])

    # Plot training history by epoch
    #plt.plot(results.epoch, results.history['val_mean_sq_err'], label='test')
    #plt.plot(results.epoch, results.history['mean_sq_err'], label='train')
    #plt.legend()
    #plt.ylabel('Mean Squared Error')
    #plt.xlabel('Epoch')
    #plt.title('Neural Network Training History')
    #plt.savefig("train_history_RMSprop.eps")
    #plt.show()

    # Make predictions - using validation set
    model_preds = model.predict(x_val)[:,0]
    model_test = model.predict(x_test)[:,0]
    model_train = model.predict(x_train)[:,0]

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

    # linear regression of actual vs predicted
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_val,
        model_preds)

    print("Prediction Mean Error: %.2g" % model_me)
    print("r-squared: %.2g" % r_value**2)
    metricsME.append(model_me)
    metricsRsq.append(r_value**2)

print("%.2g (+/- %.2g)" % (np.mean(metricsME), np.std(metricsME)))
print("%.2g (+/- %.2g)" % (np.mean(metricsRsq), np.std(metricsRsq)))

