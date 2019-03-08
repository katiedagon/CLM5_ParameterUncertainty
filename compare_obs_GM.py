# Comparing observations with global mean model output
# 11/15/18

#source /glade/work/kdagon/ncar_pylib_clone/bin/activate

import numpy as np
#from scipy.io import netcdf as nc
import matplotlib.pyplot as plt

# GM GPP from process_obs_GM.ncl
GPP_obs_GM = 2.356544

# GM GPP from CLM5 PPE
GPP_PPE_GM = np.loadtxt("outputdata/outputdata_GPP.csv")

plt.hist(GPP_PPE_GM, bins=20)
plt.xlabel('Global Mean GPP (umol/m2s)')
plt.ylabel('Counts')
plt.axvline(x=GPP_obs_GM, color='r', linestyle='dashed', linewidth=2)
#plt.savefig("dist_outputdata_GPP_withobs.pdf")
plt.show()

# Read in parameter scalings (input to NN) and actual values (input to CLM PPE)
inputdata = np.load(file="lhc_100.npy")
parameters = np.load(file="parameter_files/parameters_LHC_100.npy")

# Isolate "best match" parameter set
diff = abs(GPP_PPE_GM - GPP_obs_GM)
#print(diff)
pset = np.argmin(diff)
print(pset)

# Print best match (scaling values)
#print(inputdata[pset,:])
# Print best match (parameter values)
#print(parameters[pset,:])

# GM ET from CLM5 PPE
ET_PPE_GM = np.loadtxt("outputdata/outputdata_ET.csv")

ET_set = ET_PPE_GM[pset]

plt.hist(ET_PPE_GM, bins=20)
plt.xlabel('Global Mean ET (mm/yr)')
plt.ylabel('Counts')
plt.axvline(x=ET_set, color='r', linestyle='dashed', linewidth=2)
plt.show()




