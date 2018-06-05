# For now run this ncar python env in the command line (or bash script)
# Not sure how to execute within python script:
#source /glade/p/work/kdagon/ncar_pylib_clone/bin/activate

from keras.models import Sequential
from keras.layers import Dense

import numpy

# Fix random seed for reproducibility
numpy.random.seed(5)

# Read in input array
inputdata = numpy.load(file="lhc_100.npy")

# Read in output array
# use NCL script to generate global mean GPP array 
# from 100-member ensemble in python readable format
outputdata = numpy.loadtxt("outputdata.csv")

# Create 2-layer simple model
model = Sequential()
# first layer with 4 nodes and rectified linear activation
# specify input_dim as number of parameters, not number of simulations
model.add(Dense(4, input_dim=inputdata.shape[1], activation='relu'))
# output layer with hyperbolic tangent
model.add(Dense(1, activation='tanh')) 

# Compile model
model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
