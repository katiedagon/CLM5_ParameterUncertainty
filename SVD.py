# Calculating SVD for CLM output
# 10/18/18

#source /glade/work/kdagon/ncar_pylib_clone/bin/activate

import numpy as np
from scipy.io import netcdf as nc
import matplotlib.pyplot as plt

# Read netcdf file (pre-processed in NCL)
f = nc.netcdf_file("outputdata/outputdata_GPP_forSVD_100.nc",'r',mmap=False)
#f = nc.netcdf_file("outputdata/outputdata_LHF_forSVD_100.nc",'r',mmap=False)

# Read variable data
X = f.variables['GPP']
#X = f.variables['LHF']
mask = f.variables['datamask']

# Convert to numpy array
d = X[:]
m = mask[:]

# Get dimensions
# the order here is important
nens=d.shape[0]
nlat=d.shape[1]
nlon=d.shape[2]

# Reshape so input is (..,M,N) which is important for svd 
dr = np.reshape(d,(nens,nlat*nlon))
#print(dr.shape)
mr = np.reshape(m,nlat*nlon)

# Replace masked gridpoints with zero (shouldn't impact SVD?)
dr[:,mr==0] = 0
# Replace FillValue with zero (shouldn't impact SVD?)
dr[dr==1.e+36] = 0

# SVD command (no trunc option)
U,s,Vh = np.linalg.svd(dr, full_matrices=False)
#print(U.shape)
print(U[:,0]) # first mode
#plt.hist(U[:,0], bins=20)
#plt.show()
#print(s.shape)
#print(s) # singular values
#print(Vh.shape)
# Columns of U are modes of variability
# And the rows are ensemble members
# Singular values are in s

# Sanity check - reconstruction
print(np.allclose(dr, np.dot(U*s, Vh)))
smat = np.diag(s)
print(np.allclose(dr, np.dot(U, np.dot(smat, Vh))))

# Save out first 10 modes from SVD
# Note: cannot save masked array to file (this way)
# Full SVD
#np.save("outputdata/outputdata_GPP_SVD", U[:,0:10])
#np.save("outputdata/outputdata_ET_SVD", U[:,0:10])

# Compare with Observations

# Read netcdf file (pre-processed in NCL)
# anomalies from ensemble mean where ensemble includes obs (n=101)
fo = nc.netcdf_file("obs/obs_GPP_4x5_anom_forSVD.nc",'r',mmap=False)
#fo = nc.netcdf_file("obs/obs_LHF_4x5_anom_forSVD.nc",'r',mmap=False)
# anomalies from ensemble mean where ensemble does NOT include obs (n=100)
#fo = nc.netcdf_file("obs/obs_GPP_4x5_anom_forSVD_alt.nc",'r',mmap=False)

# Read variable data
Xo = fo.variables['GPP']
#Xo = fo.variables['LHF']
masko = fo.variables['datamask']

# Convert to numpy array
do = Xo[:]
print(do.shape)
mo = masko[:]

# Get dims
nenso=1
nlato=do.shape[0]
nlono=do.shape[1]

# Replace masked gridpoints with zero (shouldn't impact SVD?)
do[mo==0] = 0
# Replace FillValue with zero (shouldn't impact SVD?)
do[do==-9999] = 0

# Reshape so input is (..,M,N) which is important svd 
# Where M=nenso, N=ngrid=nlato*nlono
dro = np.reshape(do,(nenso,nlato*nlono))
#print(dro.shape)

# Project obs into SVD space
from numpy.linalg import pinv
U_obs = np.dot(dro,pinv(np.dot(smat,Vh)))
#print(U_obs.shape)
print(U_obs)

# Plot first mode of model U-vector (distribution) with U_obs (vertical line)
plt.hist(U[:,0], bins=20)
plt.xlabel('Mode 1 of GPP SVD (U-vector)')
#plt.xlabel('Mode 1 of LHF SVD (U-vector)')
plt.ylabel('Counts')
plt.axvline(x=U_obs[:,0], color='r', linestyle='dashed', linewidth=2)
#plt.savefig("dist_outputdata_GPP_SVD_mode1_withobs.pdf")
#plt.savefig("dist_outputdata_LHF_SVD_mode1_withobs.pdf")
plt.show()

# Second mode
plt.hist(U[:,1], bins=20)
plt.xlabel('Mode 2 of GPP SVD (U-vector)')
plt.ylabel('Counts')
plt.axvline(x=U_obs[:,1], color='r', linestyle='dashed', linewidth=2)
plt.show()

# Third mode
plt.hist(U[:,2], bins=20)
plt.xlabel('Mode 3 of GPP SVD (U-vector)')
plt.ylabel('Counts')
plt.axvline(x=U_obs[:,2], color='r', linestyle='dashed', linewidth=2)
plt.show()
