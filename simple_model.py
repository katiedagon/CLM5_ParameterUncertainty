# Try out some simpler models between parameter values (LHC) and CLM model
# output
# 6/29/18

#source /glade/work/kdagon/ncar_pylib_clone/bin/activate

from scipy import stats
from sklearn import linear_model

import numpy as np
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt

# Read in input array
inputdata = np.load(file="lhc_100.npy")

# List of input variables
in_vars = ['medlynslope','dleaf','kmax','fff','dint','baseflow_scalar']

# Read in output array
outputdata = np.loadtxt("outputdata/outputdata_GPP.csv")

# Simple scatterplots and correlation coeffs
#for x in range(6):
#    plt.scatter(inputdata[:,x],outputdata)
#    plt.ylabel('CLM Output')
#    plt.xlabel('LHC values')
#    plt.title(in_vars[x])
#    plt.savefig("scatter_GPPvs%s.eps" % (in_vars[x], ))
#    plt.show()

#    rho, spval = stats.spearmanr(inputdata[:,x],outputdata)
#    print(rho, spval)

#    pcc, ppval = stats.pearsonr(inputdata[:,x],outputdata)
#    print(pcc, ppval)

# use eumerate to avoid range command
#for x, y in enumerate(in_vars):
#    plt.scatter(inputdata[:,x], outputdata)
# etc...except for two changes:
#    plt.title(y)
#    plt.savefig("scatter_GPPvs%s.eps" % (y, ))

# Multivariate linear regression
regr = linear_model.LinearRegression()
regr.fit(inputdata, outputdata)
print('Coefficients: \n', regr.coef_)

pred = regr.predict(inputdata)
print(pred.shape)

print(inputdata.shape, outputdata.shape)
plt.scatter(inputdata[:,0], outputdata,  color='black')
plt.plot(inputdata, pred, color='blue', linewidth=3)

#plt.xticks(())
#plt.yticks(())

plt.savefig('test.pdf')
plt.close()
#plt.show()

