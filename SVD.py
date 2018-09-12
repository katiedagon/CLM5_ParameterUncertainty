#source /glade/work/kdagon/ncar_pylib_clone/bin/activate

import numpy as np
from scipy.io import netcdf as nc
#from numpy.linalg import matrix_rank

# Read netcdf file (pre-processed)
f=nc.netcdf_file("outputdata/outputdata_GPP_forSVD_100.nc",'r',mmap=False)
# Read variable data
X=f.variables['X']
# Convert to numpy array
d = X[:]
print(d.shape)
#print(d.ndim)
nlat=d.shape[0]
nlon=d.shape[1]
nens=d.shape[2]

# This works without error, but how to interpret output?
#U, s, Vh = np.linalg.svd(d)
#U.shape
#s.shape
#Vh.shape
# I think the dimensions of input d should be reordered

# Reorder output dimensions
# I think lat/lon should be last 2 dims
# But still not sure how to interpret output matrices
#c = np.transpose(d,(2,0,1))
#print(c.shape)
#U, s, Vh = np.linalg.svd(c)
#print(U.shape)
#print(s.shape)
#print(Vh.shape)
# The above dimensions seem weird...how to interpret?

# Reshape so input is (..,M,N) which seems important for np.linalg.svd 
# Where M=nens, N=ngrid=nlat*nlon
dr = np.reshape(d,(nens,nlat*nlon))
#print(dr.shape)
#print(dr.ndim)
# doesn't like this rank command...
#print(np.linalg.matrix_rank(dr))
#print(np.linalg.matrix_rank(d))
 
U,s,Vh = np.linalg.svd(dr, full_matrices=False)
#print(U.shape)
#print(np.linalg.matrix_rank(U))
#print(U[:,0])
# First 10 modes
#print(U[:,0:10].shape)
#print(U[:,1])
#print(U[0,:])
#print(s.shape)
#print(s)
#print(Vh.shape)
# This works but how to interpret output?
# I think the columns of U are modes of variability
# And the rows are ensemble members
# Still not sure...

# Alternate function using sklearn
#from sklearn.decomposition import TruncatedSVD
#svd = TruncatedSVD()
#svd.fit(d) # Doesn't work because ndims = 3
#svd.fit(dr) # Doesn't work because NaNs

# Save out first 10 modes from SVD
np.save("outputdata/outputdata_GPP_SVD", U[:,0:10])
