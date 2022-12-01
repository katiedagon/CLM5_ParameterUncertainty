# Plotting MCMC results
# 9/6/19

# Import some modules
import numpy as np
import matplotlib.pyplot as plt

# Read in MCMC results
res =  np.load(file="MCMC_lastsample_1e5epochs_v2.npy")
#print(res.shape)
ndim = res.shape[1]

# List input variables
in_vars = ['medlynslope','dleaf','kmax','fff','dint','baseflow_scalar']

# Plot distributions
#fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
fig, axes = plt.subplots(nrows=2, ncols=3, sharex=False, sharey='row',
        figsize=(10,8))
labels = in_vars
# flatten axes object to iterate
axes = axes.flatten()
for i in range(ndim):
    ax = axes[i]
    ax.hist(res[:, i])
    #ax.set_xlim(0, 1)
    ax.set_title(labels[i])

axes[0].set_ylabel("counts")
axes[3].set_ylabel("counts")

plt.savefig("MCMC_lastsample_dist_1e5epochs_v2.pdf")
#plt.show()
