# Calculating SVD for CLM output
# 10/18/18

#source /glade/work/kdagon/ncar_pylib_clone/bin/activate

import numpy as np
from scipy.io import netcdf as nc
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank

# Read netcdf file (pre-processed in NCL)
f=nc.netcdf_file("outputdata/outputdata_GPP_forSVD_100.nc",'r',mmap=False)
#f=nc.netcdf_file("outputdata/outputdata_ET_forSVD_100.nc",'r',mmap=False)
# Read variable data
X=f.variables['X']
# Convert to numpy array
d = X[:]
# Get dimensions
# the order here is important
nens=d.shape[0]
nlat=d.shape[1]
nlon=d.shape[2]

# Replace FillValue with zero (shouldn't impact SVD?)
d[d == 1.e+36] = 0
#print(d.shape)

# Ensemble mean
d_em = np.mean(d,axis=0)
#print(d_em.shape)
plt.contourf(d_em)
plt.colorbar()
plt.show()

# Reshape so input is (..,M,N) which is important svd 
# Where M=nens, N=ngrid=nlat*nlon
dr = np.reshape(d,(nens,nlat*nlon))
print(dr.shape)
#plt.contourf(dr)
#plt.colorbar()
#plt.show()

# SVD command (no trunc option)
U,s,Vh = np.linalg.svd(dr, full_matrices=False)
print(U.shape)
#print(np.linalg.matrix_rank(U))
#print(U[:,0]) # first mode
print(s.shape)
#print(s) # singular values
print(Vh.shape)
# Columns of U are modes of variability
# And the rows are ensemble members
# Singular values are in s

# Sanity check - reconstruction
print(np.allclose(dr, np.dot(U*s, Vh)))
#plt.contourf(dr - np.dot(U*s, Vh))
#plt.colorbar()
#plt.show()
smat = np.diag(s)
print(np.allclose(dr, np.dot(U, np.dot(smat, Vh))))

# Alternate function using sklearn
# used to calculate % variance
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=10)
svd.fit(dr)
#print(svd.explained_variance_)
#print(svd.explained_variance_ratio_)
#print(svd.explained_variance_ratio_.sum())
#print(svd.singular_values_) # equivalent to s array
#print(svd.components_.shape) # equivalent to Vh (sign flip?)
# The above doesn't explicitly produce U matrix
# or I don't know how to produce U

# Workaround to produce U matrix
from sklearn.utils.extmath import randomized_svd
Uk, sk, Vhk = randomized_svd(dr, n_components=10)
#print(Uk.shape)
#print(np.linalg.matrix_rank(Uk))
#print(Uk[:,0])
#print(sk.shape)
#print(Vhk.shape)

# Sanity checks - partial reconstruction
#sanity_check = np.dot(Uk*sk, Vhk)
# probably have to set tols high enough for this not to fail
#print(np.allclose(dr, sanity_check, rtol=1e-05, atol=1e-07))
#print(np.allclose(dr, sanity_check))
# clearly some parts of the matrix do not close
# likely because of truncation
# another reason to use U from the full SVD
#skmat = np.diag(sk)
#sanity_check_2 = np.dot(Uk, np.dot(skmat, Vhk))
#print(np.allclose(dr, sanity_check_2))

# Testing the difference between U vectors for full/partial SVDs
#plt.contourf(Uk - U[:,0:10])
#plt.colorbar()
#plt.show()

# Save out first 10 modes from SVD
# Note: cannot save masked array to file (this way)
# Full SVD
#np.save("outputdata/outputdata_GPP_SVD", U[:,0:10])
#np.save("outputdata/outputdata_ET_SVD", U[:,0:10])
# Truncated SVD
#np.save("outputdata/outputdata_GPP_SVD", Uk)

# Compare with Observations
# Read netcdf file (pre-processed in NCL)
fo=nc.netcdf_file("obs/obs_GPP_4x5_anom_forSVD.nc",'r',mmap=False)
# Read variable data
Xo=fo.variables['GPP']
# Convert to numpy array
do = Xo[:]
print(do.shape)
nyrs=do.shape[0]
#nyrs=1
nlato=do.shape[1]
nlono=do.shape[2]
# Replace FillValue with zero (shouldn't impact SVD?)
do[do == 1.e+36] = 0
# Reshape so input is (..,M,N) which is important svd 
# Where M=nyrs, N=ngrid=nlato*nlono
dro = np.reshape(do,(nyrs,nlato*nlono))
print(dro.shape)

# Project obs into SVD space
from numpy.linalg import pinv
test = np.dot(smat,Vh)
print(test.shape)
U_obs = np.dot(dro,pinv(np.dot(smat,Vh)))
print(U_obs.shape)
print(U_obs)
plt.hist(U_obs[0,:],bins=20)
plt.show()
