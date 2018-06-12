# For now run this ncar python env in the command line (or bash script)
# Not sure how to execute within python script:
#source /glade/p/work/kdagon/ncar_pylib_clone/bin/activate

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from keras.regularizers import l2

import keras.backend as K

import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt

# Fix random seed for reproducibility
np.random.seed(5)

# Read in input array
inputdata = np.load(file="lhc_100.npy")
#inputreshape = np.reshape(inputdata, 100*6)
#plt.hist(inputreshape, bins=50)

#hist, bin_edges = np.histogram(inputdata, bins=50)
#plt.bar(bin_edges[:-1], hist, width=0.02)
#plt.xlim(min(bin_edges), max(bin_edges))
#plt.xlabel('LHC values')
#plt.ylabel('Counts')
#plt.show()

# List of input variables
in_vars = ['medlynslope','dleaf','kmax','fff','dint','baseflow_scalar']

# Rescale inputdata to be centered around 0, get scaling values
#def normalize_multivariate_data(data, scaling_values=None):
#    normed_data = np.zeros(data.shape, dtype=data.dtype)
#    scale_cols = ["mean", "std"]
#    if scaling_values is None:
#        scaling_values = pd.DataFrame(np.zeros((data.shape[-1],
#            len(scale_cols)), dtype=np.float32), columns=scale_cols)
#    for i in range(data.shape[-1]):
#        scaling_values.loc[i, ["mean", "std"]] = [data[:, i].mean(), 
#            data[:, i].std()]
#        normed_data[:, i] = (data[:, i] - scaling_values.loc[i, 
#            "mean"]) / scaling_values.loc[i, "std"]
#    return normed_data, scaling_values
#norm_in_data, scaling_values = normalize_multivariate_data(inputdata)

# Read in output array
# use NCL script to generate global mean GPP array 
# from 100-member ensemble in python readable format
outputdata = np.loadtxt("outputdata.csv")
#print(outputdata)

plt.hist(outputdata, bins=20)
plt.xlabel('Global Mean GPP (umol/m2s)')
plt.ylabel('Counts')
plt.savefig("dist_outputdata.eps")
plt.show()

# Create 2-layer simple model
model = Sequential()
# first layer with 20 nodes and rectified linear activation
# specify input_dim as number of parameters, not number of simulations
# l2 norm regularizer
model.add(Dense(20, input_dim=inputdata.shape[1], activation='relu',
    kernel_regularizer=l2(.001)))
# output layer with linear activation
model.add(Dense(1))

# Define model metrics
def mean_error(y_true,y_pred):
    return K.mean(y_true-y_pred)

# Compile model
#model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
#model.compile(optimizer='adam',loss='mse',metrics=[mean_pred])
# using a stochastic gradient descent optimizer
opt_dense = SGD(lr=0.001, momentum=0.99, decay=1e-4, nesterov=True)
model.compile(opt_dense, "mse", metrics=[mean_error])
model.summary()

# Separate training/test data: 60/40 split
# how does test split into validation/verification? try 20/20
x_train = inputdata[0:60,:]
x_test = inputdata[60:,:]
#x_val = inputdata[80:,:]
y_train = outputdata[0:60]
y_test = outputdata[60:]
#y_val = outputdata[80:]


# Fit the model
#model.fit(x_train, y_train, epochs=20, batch_size=10)
#model.fit(x_train, y_train, epochs=40, batch_size=10)
#print(model)
results = model.fit(x_train, y_train, epochs=150, batch_size=30,
        validation_data=(x_test,y_test))

#print(results.history)
print("Training Mean Error:", results.history['mean_error'][-1])

# Plot histogram of model mean error
plt.hist(results.history['mean_error'], bins=20)
plt.xlabel('Training Mean Error')
plt.ylabel('Counts')
plt.savefig("dist_train_me.eps")
plt.show()

# Plot training history by epoch
plt.plot(results.epoch, results.history['val_mean_error'], label='validation')
plt.plot(results.epoch, results.history['mean_error'], label='train')
#plt.xticks(results.epoch)
plt.legend()
plt.hlines(y=0,xmin=0,xmax=150)
plt.ylabel('Mean Error')
plt.xlabel('Epoch')
plt.title('Neural Network Training History')
plt.savefig("train_history.eps")
plt.show()

# Evaluate the model using test data
#score = model.evaluate(x_test, y_test, batch_size=10)
#print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
#print(score)

# Make predictions - using validation subsection?
#predictions = model.predict(inputdata)
#predictions = model.predict(x_val)
#print(predictions)
model_preds = model.predict(x_test)[:,0]

# model metric for predictions
def model_error_preds(y_true,y_pred):
    return np.mean(y_true-y_pred)

# calculate model mean error with predictions
model_me = model_error_preds(y_test, model_preds)
print("Validation Mean Error: ", model_me)

# plot histogram of model mean error with predictions
plt.hist(y_test-model_preds, bins=20)
plt.xlabel('Validation Mean Error')
plt.ylabel('Counts')
plt.savefig("dist_val_me.eps")
plt.show()

# Model interpretation by permutation variable importance
def variable_importance(model, data, labels, input_vars, score_func, num_iters=10):
    preds = model.predict(data)[:, 0]
    score_val = score_func(labels, preds)
    indices = np.arange(data.shape[0])
    imp_scores = np.zeros((len(input_vars), num_iters))
    shuf_data = np.copy(data)
    for n in range(num_iters):
        print(n)
        np.random.shuffle(indices)
        for v, var in enumerate(input_vars):
            print(var)
            shuf_data[:, v] = shuf_data[indices, v]
            shuf_preds = model.predict(shuf_data)[:, 0]
            imp_scores[v, n] = score_func(labels, shuf_preds)
            shuf_data[:, v] = data[:, v]
    return score_val - imp_scores

model_imp_scores = variable_importance(model, x_test, y_test, in_vars,
        model_error_preds, num_iters=3)
model_mean_scores = model_imp_scores.mean(axis=1)
#for v, var in enumerate(in_vars):
#    print(var, model_mean_scores[v])

# Visualize what input most activates the output layer through feature
# optimization
out_diff = K.mean((model.layers[-1].output - 1) ** 2)
grad = K.gradients(out_diff, [model.input])[0]
grad /= K.maximum(K.sqrt(K.mean(grad ** 2)), K.epsilon())
iterate = K.function([model.input, K.learning_phase()],
        [out_diff, grad])
input_img_data = np.zeros(shape=(1, len(in_vars)))
#print(input_img_data.shape)

#for i in range(20):
#    out_loss, out_grad = iterate([input_img_data, 0])
#    input_img_data -= out_grad * 0.1
#    print(out_loss, out_grad.max())

# Plot the output layer feature optimization
#not working right now - input data is wrong size 
#plt.pcolormesh(input_img_data[0,0] * scaling_values.loc[0,"std"] +
#        scaling_values.loc[0,"mean"], vmin=-10, vmax=80,
#        cmap="gist_ncar")
#plt.colorbar()
#plt.quiver(input_img_data[0,-2]*scaling_values.loc[1,"std"] + 
#        scaling_values.loc[1,"mean"], 
#        input_img_data[0,-1]*scaling_values.loc[2,"std"] + 
#        scaling_values.loc[2,"mean"], scale=500)
#plt.title("Neural Net Output Layer Feature Optimization")
#plt.show()

#plt.pcolormesh(input_img_data[0,0], vmin=-5, vmax=5, cmap="RdBu_r")
#plt.quiver(input_img_data[0,1], input_img_data[0,2], scale=100)
#plt.show()

# Visualize the distribution of model weights in the output layer
#plt.hist(model.layers[-2].get_weights()[0], bins=50)
#plt.xlabel("Neural Net Output Layer Weights")
#plt.ylabel("Counts")
#plt.show()
