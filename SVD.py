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
#plt.contourf(d[0,:,:])
#plt.colorbar()
#plt.show()
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
#dr[:,mr==0] = 0
# Replace FillValue with zero (shouldn't impact SVD?)
#dr[dr==1.e+36] = 0

# Alternate: subset dr for land only grid points
drm = dr[:,mr==1]
#print(drm.shape)
# Still need to get rid of lingering FillValues (why do they not match the mask?
drm[drm==1.e+36] = 0
# Test plot
#plt.contourf(drm)
#plt.colorbar()
#plt.show()

# SVD command (no trunc option)
U,s,Vh = np.linalg.svd(drm, full_matrices=False)
#print(U.shape)
#print(U[:,0]) # first mode
#plt.hist(U[:,0], bins=20)
#plt.show()
#print(s.shape)
print(s) # singular values
prop_var = s**2/np.sum(s**2)
#print(prop_var) # proportion of variance explained
print(np.sum(prop_var)) # should be 100%
print(np.sum(prop_var[0:3])) # first 3 modes, total variance
print(prop_var[0:3]) # first 3 modes, individual variance
#print(Vh.shape)
# Columns of U are modes of variability
# And the rows are ensemble members
# Singular values are in s

# Sanity check - reconstruction
print(np.allclose(drm, np.dot(U*s, Vh)))
smat = np.diag(s)
print(np.allclose(drm, np.dot(U, np.dot(smat, Vh))))

# Plot first mode of model U-vector (distribution)
plt.hist(U[:,0], bins=20)
plt.xlabel('Mode 1 of GPP SVD (U-vector)')
#plt.xlabel('Mode 1 of LHF SVD (U-vector)')
plt.ylabel('Counts')
#plt.savefig("dist_outputdata_GPP_SVD_mode1.pdf")
#plt.savefig("dist_outputdata_LHF_SVD_mode1.pdf")
plt.show()

# Save out first n modes from SVD
# Note: cannot save masked array to file (this way)
# Full SVD
#np.save("outputdata/outputdata_GPP_SVD_3modes", U[:,0:3])
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
#print(do.shape)
mo = masko[:]
#print(mo.shape)

# Get dims
nenso=1
nlato=do.shape[0]
nlono=do.shape[1]

# Reshape so input is (..,M,N) which is important svd 
# Where M=nenso, N=ngrid=nlato*nlono
dro = np.reshape(do,(nenso,nlato*nlono))
#print(dro.shape)
mro = np.reshape(mo,nlato*nlono)
#print(mro.shape)

# Replace masked gridpoints with zero (shouldn't impact SVD?)
#do[mo==0] = 0
# Replace FillValue with zero (shouldn't impact SVD?)
#do[do==-9999] = 0

# Alternate: subset do for land only grid points
drom = dro[:,mro==1]
#print(drom.shape)
# FillValues persist
drom[drom==-9999] = 0
# Test plot
#plt.plot(drom[0,:])
#plt.show()

# Project obs into SVD space
from numpy.linalg import pinv
U_obs = np.dot(drom,pinv(np.dot(smat,Vh)))
#print(U_obs.shape)
#print(U_obs)

# Print out U_obs for first mode
#print(U_obs[:,0])
# First 3 modes
print(U_obs[:,0:3])

# Save out first n modes of U_obs
#np.save("obs/obs_GPP_SVD_3modes", U_obs[:,0:3])

# Plot first mode of model U-vector (distribution) with U_obs (vertical line)
plt.hist(U[:,0], bins=20)
plt.xlabel('Mode 1 of GPP SVD (U-vector)')
#plt.xlabel('Mode 1 of LHF SVD (U-vector)')
plt.ylabel('Counts')
plt.axvline(x=U_obs[:,0], color='r', linestyle='dashed', linewidth=2)
#plt.savefig("dist_outputdata_GPP_SVD_mode1_withobs.pdf")
#plt.savefig("dist_outputdata_LHF_SVD_mode1_withobs.pdf")
plt.show()

# Project test paramset into SVD space
#ft = nc.netcdf_file("outputdata/test_paramset_001_GPP_forSVD.nc",'r',mmap=False)
#Xt = ft.variables['GPP']
#maskt = ft.variables['datamask']
#dt = Xt[:]
#mt = maskt[:]
#drt = np.reshape(dt,(nenso,nlato*nlono))
#mrt = np.reshape(mt,nlato*nlono)
#drtm = drt[:,mrt==1]
#print(drtm.shape)
#drtm[drtm==1.e+36] = 0
#U_test = np.dot(drtm,pinv(np.dot(smat,Vh)))
#print(U_test[:,0])

# Plot distribution with U_obs and U_test
#plt.hist(U[:,0], bins=20)
#plt.xlabel('Mode 1 of GPP SVD (U-vector)')
#plt.ylabel('Counts')
#plt.axvline(x=U_obs[:,0], color='r', linestyle='dashed', linewidth=2)
#plt.axvline(x=U_test[:,0], color='b', linestyle='dashed', linewidth=2)
#plt.show()

# Project model with default params into SVD space
fd= nc.netcdf_file("outputdata/CLM_default_GPP_forSVD.nc",'r',mmap=False)
Xd = fd.variables['GPP']
maskd = fd.variables['datamask']
dd = Xd[:]
md = maskd[:]
drd = np.reshape(dd,(nenso,nlato*nlono))
mrd = np.reshape(md,nlato*nlono)
drdm = drd[:,mrd==1]
#print(drdm.shape)
drdm[drdm==1.e+36] = 0
U_default = np.dot(drdm,pinv(np.dot(smat,Vh)))
print(U_default[:,0:3])

# Save out first n modes of U_default
#np.save("outputdata/modeldefault_GPP_SVD_3modes", U_default[:,0:3])

# Plot distribution with U_default
plt.hist(U[:,0], bins=20)
plt.xlabel('Mode 1 of GPP SVD (U-vector)')
plt.ylabel('Counts')
plt.axvline(x=U_obs[:,0], color='r', linestyle='dashed', linewidth=2)
#plt.axvline(x=U_test[:,0], color='b', linestyle='dashed', linewidth=2)
plt.axvline(x=U_default[:,0], color='k', linestyle='dashed', linewidth=2)
#plt.savefig("dist_outputdata_GPP_SVD_mode1_withobs_anddefault.pdf")
plt.show()

# Second mode
plt.hist(U[:,1], bins=20)
plt.xlabel('Mode 2 of GPP SVD (U-vector)')
plt.ylabel('Counts')
plt.axvline(x=U_obs[:,1], color='r', linestyle='dashed', linewidth=2)
plt.axvline(x=U_default[:,1], color='k', linestyle='dashed', linewidth=2)
#plt.savefig("dist_outputdata_GPP_SVD_mode2_withobs_anddefault.pdf")
plt.show()

# Third mode
plt.hist(U[:,2], bins=20)
plt.xlabel('Mode 3 of GPP SVD (U-vector)')
plt.ylabel('Counts')
plt.axvline(x=U_obs[:,2], color='r', linestyle='dashed', linewidth=2)
plt.axvline(x=U_default[:,2], color='k', linestyle='dashed', linewidth=2)    
#plt.savefig("dist_outputdata_GPP_SVD_mode3_withobs_anddefault.pdf")
plt.show()
