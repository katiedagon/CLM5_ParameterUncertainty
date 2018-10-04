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
nlat=d.shape[0]
nlon=d.shape[1]
nens=d.shape[2]

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

# Quick check plots
#plt.contourf(d[:,:,0])
#plt.colorbar()
#plt.show()

# Mask NCL fillvalue (non-land gridpoints)
c = ma.masked_where(d == 1.e+36, d)
#plt.contourf(c[:,:,0])
#plt.colorbar()
#plt.show()

# Reshape so input is (..,M,N) which seems important for np.linalg.svd 
# Where M=nens, N=ngrid=nlat*nlon
dr = np.reshape(d,(nens,nlat*nlon))
#print(dr.shape)

# Reshape masked array
cr = np.reshape(c,(nens,nlat*nlon))
#print(cr.shape)
#print(cr.ndim)
# The following gives rank 0?
#print(np.linalg.matrix_rank(cr))
# Error on this command:
#print(np.linalg.matrix_rank(c))

# SVD command (no trunc option)
U,s,Vh = np.linalg.svd(dr, full_matrices=False)
#U,s,Vh = np.linalg.svd(dr, full_matrices=True)
print(U.shape)
#print(np.linalg.matrix_rank(U))
print(U[:,0]) # first mode?
#print(U[0,:]) # first ensemble member?
# First 10 modes?
#print(U[:,0:10].shape)
print(s.shape)
print(Vh.shape)
# This works but how to interpret output?
# I think the columns of U are modes of variability
# And the rows are ensemble members
# Still not sure...

# Sanity check - partial reconstruction (fails)
print(np.allclose(dr, np.dot(U*s, Vh)))
smat = np.diag(s)
print(np.allclose(dr, np.dot(U, np.dot(smat, Vh))))

# No difference in SVD result when masking fillvalue
Uc,sc,Vhc = np.linalg.svd(cr, full_matrices=False)
print(Uc[0,:] - U[0,:])
print(Uc[99,:] - U[99,:])
# Sanity check
print(np.allclose(cr, np.dot(Uc*sc, Vhc)))
smat = np.diag(sc)
print(np.allclose(cr, np.dot(Uc, np.dot(smat, Vhc))))

# What about reshape with input as (..N,M) so spatial data/larger dim comes first?
er = np.reshape(d,(nlat*nlon,nens))
print(er.shape)
Ue,se,Vhe = np.linalg.svd(er, full_matrices=True)
#print(Ue[0,:])
print(Ue.shape)
print(se.shape)
print(Vhe.shape)
#print(Vhe[0,:])
# Not sure how to interpret these results...

# Sanity check - full reconstruction (fails)
print(np.allclose(er, np.dot(Ue[:,:nens]*se, Vhe)))
smat = np.zeros((nlat*nlon,nens), dtype=complex)
smat[:nens,:nens] = np.diag(se)
print(np.allclose(er, np.dot(Ue, np.dot(smat, Vhe))))

# Other ways of masking or obtaining non-masked indices
#fr = cr[cr.mask == False]
#fr = cr[~cr.mask]
#ind = dr!=1.e+36
#fr = np.where(dr!=1.e+36)

# Alternate function using sklearn
#from sklearn.decomposition import TruncatedSVD
#svd = TruncatedSVD()
#svd.fit(d) # Doesn't work because ndims = 3
#svd.fit(dr) # Doesn't work because NaNs
#svd.fit(cr) # Doesn't work because NaNs (despite masking)

# Save out first 10 modes from SVD
# Note: cannot save masked array to file (this way)
#np.save("outputdata/outputdata_GPP_SVD", U[:,0:10])
