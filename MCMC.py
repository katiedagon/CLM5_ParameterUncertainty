# MCMC for optimizing the 2-layer multiple output Neural Network
# 7/2/19

# Import some modules
import emcee
import numpy as np
import matplotlib.pyplot as plt

# Load previously trained NN models
import keras.backend as K
def mean_sq_err(y_true,y_pred):
    return K.mean((y_true-y_pred)**2)
from keras.models import load_model
model_GPP = load_model('NN_GPP_finalize_multi-dim.h5',
    custom_objects={'mean_sq_err': mean_sq_err})
model_LHF = load_model('NN_LHF_finalize_multi-dim.h5',
    custom_objects={'mean_sq_err': mean_sq_err})

# List input variables
in_vars = ['medlynslope','dleaf','kmax','fff','dint','baseflow_scalar']
npar = len(in_vars)

# Read in obs targets and calculated variance
obs_GPP = np.load(file="obs/obs_GPP_SVD_3modes.npy", allow_pickle=True)
obs_LHF = np.load(file="obs/obs_LHF_SVD_3modes.npy", allow_pickle=True)
sd_GPP = np.load(file="obs/obs_GPP_SVD_3modes_allyrs_sd.npy", allow_pickle=True)
sd_LHF = np.load(file="obs/obs_LHF_SVD_3modes_allyrs_sd.npy", allow_pickle=True)

# Define normalized error function
# Weighting factor previously determined by midpoint of regimes
# see NN_opt_wgt.py
B = 1.49
def normerr(x):
    xt = x.reshape(1,-1)
    model_preds_GPP = model_GPP.predict(xt)
    model_preds_LHF = model_LHF.predict(xt)
    L = np.sum(((model_preds_GPP-obs_GPP)/sd_GPP)**2, axis=1) +\
        B*np.sum(((model_preds_LHF-obs_LHF)/sd_LHF)**2, axis=1)
    return L

# Define the prior
# Uniform distribution between [0,1]
def lnprior(x):
    if all(x > 0) and all(x < 1):
        return 0.0
    return -np.inf

# Define the full log probability function
def lnprob(x):
    lp = lnprior(x)
    if not np.isfinite(lp):
        return -np.inf
    return lp + normerr(x)

# Set number of walkers, number of dims (= number of params)
nwalkers = 200
ndim = npar

# Initialize walkers (random initial states)
p0 = [np.random.rand(ndim) for i in range(nwalkers)]

# Multiprocessing
#from multiprocessing import Pool
#pool = Pool(processes=10)

# Set up sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
#sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
# Run sampler for set number of epochs
# should really think about ~10^6 epochs (batch job?)
epochs = 10**4
#pos, prob, state = sampler.run_mcmc(p0, epochs)
sampler.run_mcmc(p0, epochs, progress=True)

# Print mean acceptance fraction
print("Mean acceptance fraction: {0:.3f}"
        .format(np.mean(sampler.acceptance_fraction)))

# Look at the sampler chain
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples_all = sampler.get_chain()
labels = in_vars
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples_all[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples_all))
    ax.set_ylabel(labels[i])
    #ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
#plt.savefig("MCMC_sampler_chain_1e3epochs.pdf")
plt.savefig("MCMC_sampler_chain_1e4epochs.pdf")
plt.show()

# Look at the log probs
probs_all = sampler.get_log_prob()
#print(probs_all.shape)
#print(np.min(probs_all))
#print(np.max(probs_all))
probs_all_scaled = probs_all*(10**(-7))
#print(np.min(probs_all_scaled))
#print(np.max(probs_all_scaled))
err_all = np.exp(-probs_all_scaled)
plt.plot(err_all, "k", alpha=0.3)
plt.xlabel("step number")
#plt.ylabel("log prob")
plt.ylabel("scaled error")
#plt.savefig("MCMC_scaled_error_1e3epochs.pdf")
plt.savefig("MCMC_scaled_error_1e4epochs.pdf")
plt.show()

# Integrated autocorrelation time
#tau = sampler.get_autocorr_time()
#print(tau)

# Discard the initial N steps
#samples = sampler.chain[:, 500:, :].reshape((-1, ndim))
#flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
flat_samples = sampler.get_chain(discard=100, flat=True)
print(flat_samples.shape)

# Corner plot
import corner
fig = corner.corner(flat_samples, labels=in_vars)
#plt.savefig("MCMC_corner_1e3epochs.pdf")
plt.savefig("MCMC_corner_1e4epochs.pdf")
plt.show()


