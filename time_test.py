# Test time for Neural Network as an emulator
# 4/40/19

import time
start = time.time()

import numpy as np
import keras.backend as K

# define error function
def mean_sq_err(y_true,y_pred):
    return K.mean((y_true-y_pred)**2)

# import saved model
from keras.models import load_model
#model = load_model('NN_finalize_multi-dim.h5', custom_objects={'mean_sq_err':mean_sq_err})
model = load_model('emulators/NN_GPP_finalize_multi-dim.h5',
        custom_objects={'mean_sq_err':mean_sq_err})

# make predictions
inputdata_inflate = np.load(file="lhc_1000.npy", allow_pickle=True)
model_preds_inflate = model.predict(inputdata_inflate)
pred_time = time.time()

# print time
print(pred_time - start)
