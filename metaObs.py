# Experimenting with modifying observational targets
# to account for structural uncertainty
# 09/25/19

#conda activate analysis

import numpy as np
from scipy.io import netcdf as nc
import matplotlib.pyplot as plt

# Select output variable
#var = "GPP"
var="LHF"

# Read PPE first 3 modes from SVD
PPE_modes = np.load("outputdata/outputdata_"+var+"_SVD_3modes.npy")
#np.save("outputdata/outputdata_GPP_SVD_3modes", U[:,0:3])
#np.save("outputdata/outputdata_LHF_SVD_3modes", U[:,0:3])
#np.save("outputdata/outputdata_GPP_SVD_3modes_v2", U[:,0:3])
#np.save("outputdata/outputdata_LHF_SVD_3modes_v2", U[:,0:3])
#np.save("outputdata/outputdata_GPP_SVD_3modes_fc", U[:,0:3])
#np.save("outputdata/outputdata_LHF_SVD_3modes_fc", U[:,0:3])
#np.save("outputdata/outputdata_GPP_SVD_3modes_diff", U[:,0:3])
#np.save("outputdata/outputdata_LHF_SVD_3modes_diff", U[:,0:3])

# Read obs first 3 modes from SVD
obs_modes = np.load("obs/obs_"+var+"_SVD_3modes.npy")

# Read sd of obs (all years) first 3 modes from SVD
norm = np.load("obs/obs_"+var+"_SVD_3modes_allyrs_sd.npy")

# Calculate likelihood based on these results
#L = np.sum(((U_test[:,0:3]-U_obs[:,0:3])/sd)**2, axis=1)
# weighted L (LHF)
#B = 1.3
#B = 1.49
#L = B*np.sum(((U_test[:,0:3]-U_obs[:,0:3])/sd)**2, axis=1)
#print(L)

# Read default model first 3 modes from SVD
default_modes = np.load("outputdata/modeldefault_"+var+"_SVD_3modes.npy")

# Calculate "meta-obs" combining obs and model default
ff = 2
mobs_modes = default_modes + (obs_modes)/ff

# Plot distribution with obs and default
fig, axs = plt.subplots(1,3, figsize=(10, 4), sharey=True)
axs = axs.ravel()

for i in range(3):
    axs[i].hist(PPE_modes[:,i], bins=20)
    axs[i].set_xlabel("EOF"+str(i+1)+" "+var)
    #axs[i].set_ylabel('Counts')
    axs[i].axvline(x=obs_modes[:,i], color='r', linestyle='dashed', linewidth=2)
    axs[i].axvline(x=default_modes[:,i], color='k', linestyle='dashed', linewidth=2)
    #axs[i].axvline(x=mobs_modes[:,i], color='b', linestyle='dashed', linewidth=2)

axs[0].set_ylabel("Counts")
#fig.savefig("dist_outputdata_"+var+"_SVD_withobs_anddefault.pdf")
fig.show()

