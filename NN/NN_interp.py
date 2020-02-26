# Model Interpretation for the 2-layer multiple output Neural Network
# Variable Importance
# 6/6/19

#from scipy import stats
#from scipy.optimize import minimize

import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.axes as ax
from sklearn.metrics import brier_score_loss, roc_curve, roc_auc_score

# Fix random seed for reproducibility
#np.random.seed(9)

# Read in input array
inputdata = np.load(file="lhc_100.npy", allow_pickle=True)

# List of input variables
in_vars = ['medlynslope','dleaf','kmax','fff','dint','baseflow_scalar']
npar = len(in_vars)

# Read in output array
# Calculated in SVD.py
# After processing in outputdata/process_outputdata_SVD.ncl
outputdata_GPP = np.load(file="outputdata/outputdata_GPP_SVD_3modes.npy",
        allow_pickle=True)
outputdata_LHF = np.load(file="outputdata/outputdata_LHF_SVD_3modes.npy",
        allow_pickle=True)
nmodes = outputdata_GPP.shape[1]

# Use trained NN as model
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import RMSprop
#from keras.regularizers import l2
import keras.backend as K
#import tensorflow as tf

#model = Sequential()
#model.add(Dense(9, input_dim=inputdata.shape[1], activation='relu',
#    kernel_regularizer=l2(.001)))
#model.add(Dense(9, activation='tanh', kernel_regularizer=l2(.001)))
#model.add(Dense(nmodes))
#opt_dense = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
def mean_sq_err(y_true,y_pred):
    return K.mean((y_true-y_pred)**2)
#model.compile(opt_dense, "mse", metrics=[mean_sq_err])
#results = model.fit(inputdata, outputdata, epochs=500, batch_size=30, verbose=0)

# Load previously trained model
from keras.models import load_model
model_GPP = load_model('NN_GPP_finalize_multi-dim.h5', custom_objects={'mean_sq_err':
    mean_sq_err})
model_LHF = load_model('NN_LHF_finalize_multi-dim.h5',
    custom_objects={'mean_sq_err': mean_sq_err})

#model_preds_GPP = model_GPP.predict(inputdata)
#model_preds_LHF = model_LHF.predict(inputdata)

# Define score for predictions
def mse_preds(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)

# Define function for feature importance
def permutation_feature_importance(input_data, output_data, model,
        score=mse_preds):
    model_preds = model.predict(input_data)
    original_error = score(output_data, model_preds)
    permuted_scores = np.zeros(input_data.shape[1])
    permuted_data = np.copy(input_data)
    permuted_indices = np.arange(input_data.shape[0])
    for c in range(input_data.shape[1]):
        np.random.shuffle(permuted_indices)
        permuted_data[:, c] = input_data[permuted_indices, c]
        permuted_preds = model.predict(permuted_data)
        permuted_scores[c] = score(output_data,permuted_preds)
        permuted_data[:, c] = input_data[:, c]
    return original_error, permuted_scores

# Calculate feature importance
gpp_original, gpp_permuted = permutation_feature_importance(inputdata,
        outputdata_GPP, model_GPP)
lhf_original, lhf_permuted = permutation_feature_importance(inputdata,
        outputdata_LHF, model_LHF)

# Plot importance
# smaller bar is more important
plt.figure(figsize=(12, 5))
plt.subplots_adjust(wspace=0.4)
plt.subplot(1, 2, 1)
plt.barh(np.arange(gpp_permuted.size), gpp_permuted)
plt.barh(gpp_permuted.size, gpp_original)
plt.yticks(np.arange(gpp_permuted.size + 1),
        in_vars + ["GPP Original"])
plt.title("GPP Importances")
plt.subplot(1, 2, 2)
plt.barh(np.arange(lhf_permuted.size), lhf_permuted)
plt.barh(lhf_permuted.size, lhf_original)
plt.yticks(np.arange(lhf_permuted.size + 1),
        in_vars + ["LHF Original"])
plt.title("LHF Importances")
plt.show()



