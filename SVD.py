#source /glade/work/kdagon/ncar_pylib_clone/bin/activate

import numpy as np
import numpy.ma as ma
from scipy.io import netcdf as nc
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank

# Read netcdf file (pre-processed in NCL)
f=nc.netcdf_file("outputdata/outputdata_GPP_forSVD_100.nc",'r',mmap=False)
# Read variable data
X=f.variables['X']
# Convert to numpy array
d = X[:]
#print(d.shape)
#print(d.ndim)
nens=d.shape[0]
nlat=d.shape[1]
nlon=d.shape[2]

# Quick check plots
#plt.contourf(d[:,:,0])
#plt.colorbar()
#plt.show()

# Ensemble mean
#d_em = np.mean(d,axis=2)
#print(d_em.shape)
# Find NaN gridpoints (ocean)
#ind = d_em<1.e+35
#print(ind.shape)
#fd = d_em[ind]
#print(fd.shape)
# The above is 1D...how to preserve spatial structure when removing ocean
# gridpoints?

# Replace FillValue with zero (shouldn't impact SVD?)
d[d == 1.e+36] = 0
#print(d.shape)

# Quick check plots
#d_em = np.mean(d, axis=2)
#plt.contourf(d[:,:,0])
#plt.contourf(d_em)
#plt.colorbar()
#plt.show()

# Mask NCL fillvalue (non-land gridpoints)
#c = ma.masked_where(d == 1.e+36, d)
#plt.contourf(c[:,:,0])
#plt.colorbar()
#plt.show()

# Reshape so input is (..,M,N) which seems important for np.linalg.svd 
# Where M=nens, N=ngrid=nlat*nlon
dr = np.reshape(d,(nens,nlat*nlon))
#print(dr.shape)
#print(dr[:,0])
#plt.contourf(dr)
#plt.colorbar()
#plt.show()

# Reshape masked array
#cr = np.reshape(c,(nens,nlat*nlon))
#print(cr.shape)
#print(cr.ndim)
# The following gives rank 0?
#print(np.linalg.matrix_rank(cr))
# Error on this command:
#print(np.linalg.matrix_rank(c))

# SVD command (no trunc option)
U,s,Vh = np.linalg.svd(dr, full_matrices=False)
#U,s,Vh = np.linalg.svd(dr, full_matrices=True)
#print(U.shape)
#print(np.linalg.matrix_rank(U))
print(U[:,0]) # first mode?
# check that norm of each U column = 1
#print(np.linalg.norm(U[:,0:10], axis=0))
# norm checks out
# check that dot product of 2 U columns = 0
#print(np.dot(U[:,0],U[:,1]))
# dot product checks out
#print(U[0,:]) # first ensemble member?
#print(U[:,99])
# First 10 modes?
#print(U[:,0:10].shape)
#plt.contourf(U[:,0:10])
#plt.colorbar()
#plt.hist(U[:,0:10])
#plt.show()
#print(s.shape)
print(s)
#print(Vh.shape)
#print(Vh[0,:])
#print(Vh[:,99])
#print(Vh[0,0:100])
# This works but how to interpret output?
# I think the columns of U are modes of variability
# And the rows are ensemble members
# Still not sure...

# Sanity check - reconstruction
print(np.allclose(dr, np.dot(U*s, Vh)))
#plt.contourf(dr - np.dot(U*s, Vh))
#plt.colorbar()
#plt.show()
smat = np.diag(s)
print(np.allclose(dr, np.dot(U, np.dot(smat, Vh))))
# Passes these checks!

# No difference in SVD result when masking fillvalue
#Uc,sc,Vhc = np.linalg.svd(cr, full_matrices=False)
#print(Uc[0,:] - U[0,:])
#print(Uc[99,:] - U[99,:])
# Sanity check
#print(np.allclose(cr, np.dot(Uc*sc, Vhc)))
#smat = np.diag(sc)
#print(np.allclose(cr, np.dot(Uc, np.dot(smat, Vhc))))

# What about reshape with input as (..N,M) so spatial data/larger dim comes first?
#er = np.reshape(d,(nlat*nlon,nens))
#print(er.shape)
#Ue,se,Vhe = np.linalg.svd(er, full_matrices=True)
#print(Ue[0,:])
#print(Ue.shape)
#print(se.shape)
#print(Vhe.shape)
#print(Vhe[0,:])
# Not sure how to interpret these results...

# Sanity check - reconstruction
#print(np.allclose(er, np.dot(Ue[:,:nens]*se, Vhe)))
#smat = np.zeros((nlat*nlon,nens), dtype=complex)
#smat[:nens,:nens] = np.diag(se)
#print(np.allclose(er, np.dot(Ue, np.dot(smat, Vhe))))

# Other ways of masking or obtaining non-masked indices
#fr = cr[cr.mask == False]
#fr = cr[~cr.mask]
#ind = dr!=1.e+36
#fr = np.where(dr!=1.e+36)

# Alternate function using sklearn
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=10)
#svd.fit(d) # Doesn't work because ndims = 3
svd.fit(dr) # Works if FillValue is replaced by zero
#svd.fit(cr) # Doesn't work because NaNs (despite masking)
print(svd.explained_variance_)
print(svd.explained_variance_ratio_)
print(svd.explained_variance_ratio_.sum())
print(svd.singular_values_) # equivalent to s array
#print(svd.components_.shape) # equivalent to Vh (sign flip?)
#print(svd.components_[0,0:100].shape)
#print(svd.components_[0,0:100])
# The above works but doesn't explicitly produce U matrix
# or I don't know how to produce U

# Workaround to produce U matrix
from sklearn.utils.extmath import randomized_svd
Uk, sk, Vhk = randomized_svd(dr, n_components=10)
#print(Uk.shape)
print(Uk[:,0]) # first mode?
# check norm - good
#print(np.linalg.norm(Uk, axis=0))
# check dot product - good
#print(np.dot(Uk[:,0],Uk[:,1]))
#print(sk.shape)
print(sk)
#print(Vhk.shape)
#print(Vhk[0,0:100])

# Check difference in truncated vs. non-truncated SVDs
print(Uk[:,0] - U[:,0])
print(Uk[:,2] - U[:,2])
#plt.contourf(Uk - U[:,0:10])
#plt.colorbar()
#plt.hist(Uk)
#plt.show()
#plt.contourf(Uk)
#plt.colorbar()
#plt.show()
# some of the modes have significant differences,
# comparing truncated (Uk) vs. non-truncated (U) SVDs
# so I will use the first 10 modes of the full SVD
# but note that both generate the same first 10 singular values

# Sanity checks - partial reconstruction
sanity_check = np.dot(Uk*sk, Vhk)
#print(sanity_check.shape)
#print(np.sum(np.isnan(dr)))
#print(np.sum(np.isnan(sanity_check)))
# probably have to set tols high enough for this not to fail
#print(np.allclose(dr, sanity_check, rtol=1e-05, atol=1e-07))
print(np.allclose(dr, sanity_check))
#print(np.mean(dr - sanity_check))
# clearly some parts of the matrix do not close
# likely because of truncation
# another reason to use U from the full SVD
#plt.contourf(dr - sanity_check)
#plt.colorbar()
#plt.show()
# interesting nonzero features in some ens members?

skmat = np.diag(sk)
#print(skmat.shape)
sanity_check_2 = np.dot(Uk, np.dot(skmat, Vhk))
print(np.allclose(dr, sanity_check_2))
#print(np.mean(dr - sanity_check_2))
#plt.contourf(dr - sanity_check_2)
#plt.colorbar()
#plt.show()
# same features here...

# Function for sparse matrix
#from scipy.sparse.linalg import svds
#Us, ss, Vhs = svds(dr, k=10)
#print(Us.shape)
#print(Us[:,0]) # first mode?
#print(ss)
# This produces s values in reverse order
# And U does not look like other U's?

# Save out first 10 modes from SVD
# Note: cannot save masked array to file (this way)
#np.save("outputdata/outputdata_GPP_SVD", U[:,0:10])
#np.save("outputdata/outputdata_GPP_SVD", Uk)
