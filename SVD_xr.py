# Calculating SVD for CLM output
# Trying to use xarray
# 3/7/19

#source /glade/work/kdagon/ncar_pylib_clone/bin/activate

import numpy as np
import xarray as xr
from eofs.xarray import Eof
#from scipy.io import netcdf as nc
import matplotlib.pyplot as plt

# Read netcdf file (pre-processed in NCL)
f=xr.open_dataset("outputdata/outputdata_GPP_forSVD_100.nc")
# Read variable data
d=f.GPP
m=f.datamask

# Try to create an eof solver
# Do I need to specify weights due to pre-processing in NCL?
# This function needs to see time as first dimension (coordinate)
#solver = Eof(d)

# Get dimensions
# the order here is important
#nens=d.shape[0]
#nlat=d.shape[1]
#nlon=d.shape[2]

# Reshape so input is (..,M,N) which is important svd 
# Where M=nens, N=ngrid=nlat*nlon
dr = d.stack(ngrid=('lat','lon'))
print(dr.shape)
#print(dr.ndim)
#plt.contourf(dr)
#plt.colorbar()
#plt.show()
mr = m.stack(ngrid=('lat','lon'))
print(mr.shape)

# Subset reshaped data by reshaped mask
drm = dr[:,mr==1]
print(drm.shape)
plt.contourf(drm)
plt.colorbar()
plt.show()

# SVD
U,s,Vh = np.linalg.svd(drm, full_matrices=False)
#print(U.shape)
print(U[:,0]) # first mode

# Compare with Observations
# Read netcdf file (pre-processed in NCL)
#fo=nc.netcdf_file("obs/obs_GPP_4x5_anom_forSVD.nc",'r',mmap=False)
# Read variable data
#Xo=fo.variables['GPP']
# Convert to numpy array
#do = Xo[:]
#print(do.shape)
#nenso=1
#nlato=do.shape[0]
#nlono=do.shape[1]
# Replace FillValue with zero (shouldn't impact SVD?)
#do[do == 1.e+36] = 0
# Reshape so input is (..,M,N) which is important svd 
# Where M=nyrs, N=ngrid=nlato*nlono
#dro = np.reshape(do,(nenso,nlato*nlono))
#print(dro.shape)

# Project obs into SVD space
#from numpy.linalg import pinv
#test = np.dot(smat,Vh)
#print(test.shape)
#U_obs = np.dot(dro,pinv(np.dot(smat,Vh)))
#print(U_obs.shape)
#print(U_obs)
#print(U_obs[0,:])
#plt.hist(U_obs[0,:],bins=20)
#plt.show()
