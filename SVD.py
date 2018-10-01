#source /glade/work/kdagon/ncar_pylib_clone/bin/activate

import numpy as np
import numpy.ma as ma
from scipy.io import netcdf as nc
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank

# Read netcdf file (pre-processed)
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

# Quick plots
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
print(dr.shape)
print(dr[:,0])
dr_em = np.mean(dr,axis=0)
print(dr_em.shape)
print(dr_em[1:100])
ind = dr_em<1.e+35
print(ind.shape)
fr = dr_em[ind]
print(fr.shape)

# Reshape masked array
cr = np.reshape(c,(nens,nlat*nlon))
#print(cr.shape)
#print(cr.ndim)
#print(cr[:,0])
# The following gives rank 0?
#print(np.linalg.matrix_rank(cr))
# Error on this command:
#print(np.linalg.matrix_rank(c))
 
U,s,Vh = np.linalg.svd(cr, full_matrices=False)
#print(U.shape)
#print(np.linalg.matrix_rank(U))
print(U[:,0])
#print(U[:,1])
# First 10 modes
#print(U[:,0:10].shape)
#print(U[0,:])
#print(U[1,:])
#print(s.shape)
#print(s)
#print(Vh.shape)
# This works but how to interpret output?
# I think the columns of U are modes of variability
# And the rows are ensemble members
# Still not sure...

# No difference in SVD result when not masking fillvalue
#Ud,sd,Vhd = np.linalg.svd(dr, full_matrices=False)
#print(Ud[0,:] - U[0,:])
#print(Ud[99,:] - U[99,:])

# What about reshape with input is (..N,M) so spatial data comes first?
#er = np.reshape(c,(nlat*nlon,nens))
#print(er.shape)
#Ue,se,Vhe = np.linalg.svd(er, full_matrices=False)
#print(Ue[0,:])
#print(Ue.shape)
#print(se.shape)
#print(Vhe.shape)
#print(Vhe[0,:])
# Not sure how to interpret these results...

# Other ways of masking or obtaining non-masked indices
#fr = cr[cr.mask == False]
#fr = cr[~cr.mask]
#ind = dr!=1.e+36
#print(ind.shape)
#print(ind[:,0])
#fr = dr[ind[0],ind[1]]
#print(fr.shape)
#print(fr[:,0])
#fr = np.where(dr!=1.e+36)
#print(len(fr))
#print(fr[0])
#print(fr[0][0])
#fd = dr[fr]
#print(fd.shape)
#Uf,sf,Vhf = np.linalg.svd(fd, full_matrices=False)
#print(Uf[:,0])

# Alternate function using sklearn
#from sklearn.decomposition import TruncatedSVD
#svd = TruncatedSVD()
#svd.fit(d) # Doesn't work because ndims = 3
#svd.fit(dr) # Doesn't work because NaNs
#svd.fit(cr) # Doesn't work because NaNs

# Save out first 10 modes from SVD
#np.save("outputdata/outputdata_GPP_SVD", U[:,0:10])
