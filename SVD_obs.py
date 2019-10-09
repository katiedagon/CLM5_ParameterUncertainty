# Calculating SVD for obs (all years, sd)
# 11/6/18

#source /glade/work/kdagon/ncar_pylib_clone/bin/activate

import numpy as np
from scipy.io import netcdf as nc
import matplotlib.pyplot as plt
from numpy.linalg import pinv

# Read netcdf file (pre-processed in NCL)
#f=nc.netcdf_file("obs/obs_GPP_4x5_anom_forSVD.nc",'r',mmap=False)
#f=nc.netcdf_file("obs/obs_GPP_4x5_anom_forSVD_allyrs.nc",'r',mmap=False)
f=nc.netcdf_file("obs/obs_LHF_4x5_anom_forSVD_allyrs.nc",'r',mmap=False)
# Read variable data
#X=f.variables['GPP']
X=f.variables['LHF']
mask=f.variables['datamask']

# Convert to numpy array
d = X[:]
m = mask[:]

# Ensemble mean (i.e., time mean)
d_em = np.mean(d,axis=0)
#print(d_em.shape)
# Test plot
#plt.contourf(d_em)
#plt.colorbar()
#plt.show()

# Get dimensions
# the order here is important
ntime=d.shape[0]
nlat=d.shape[1]
nlon=d.shape[2]

# Reshape so input is (..,M,N) which is important svd 
# Where M=ntime, N=ngrid=nlat*nlon
dr = np.reshape(d,(ntime,nlat*nlon))
#print(dr.shape)
mr = np.reshape(m,nlat*nlon)

# Mask data for land only grid points
drm = dr[:,mr==1]
#print(drm.shape)

# Plot test for FillValues
#plt.contourf(drm)
#plt.colorbar()
#plt.show()

# If FillValues persist
drm[drm==-9999] = 0
#plt.contourf(drm)
#plt.colorbar()
#plt.show()

# Calculate model SVD
#fm = nc.netcdf_file("outputdata/outputdata_GPP_forSVD_100.nc",'r',mmap=False)
fm = nc.netcdf_file("outputdata/outputdata_LHF_forSVD_100.nc",'r',mmap=False)
#Xm = fm.variables['GPP']
Xm = fm.variables['LHF']
maskm = fm.variables['datamask']
dm = Xm[:]
mm = maskm[:]
nens=dm.shape[0]
nlatm=dm.shape[1]
nlonm=dm.shape[2]
dmr = np.reshape(dm,(nens,nlatm*nlonm))
mmr = np.reshape(mm,nlatm*nlonm)
dmrm = dmr[:,mmr==1]
dmrm[dmrm==1.e+36] = 0
U,s,Vh = np.linalg.svd(dmrm, full_matrices=False)
#print(U.shape)
#print(s.shape)
#print(Vh.shape)

# Project each year of obs using model SVD
#print(drm.shape)
smat = np.diag(s)
#print(smat.shape)
U_obs = np.dot(drm,pinv(np.dot(smat,Vh)))
print(U_obs.shape)

# Calculate sd across years
U_obs_sd = np.std(U_obs,axis=0)
#print(U_obs_sd)
print(U_obs_sd.shape)
print(U_obs_sd[0:3])

# Save out first 3 modes from SVD
#np.save("obs/obs_GPP_SVD_3modes_allyrs_sd", U_obs_sd[0:3])
np.save("obs/obs_LHF_SVD_3modes_allyrs_sd", U_obs_sd[0:3])
