# MCMC for optimizing the 2-layer multiple output Neural Network
# Experiment with single objective cost functions
# 9/5/19

# Import some modules
import emcee
import numpy as np
import matplotlib.pyplot as plt

# Define objective
#ob = "GPP"
ob = "LHF"

# Load previously trained NN models
import keras.backend as K
def mean_sq_err(y_true,y_pred):
    return K.mean((y_true-y_pred)**2)
from keras.models import load_model
model = load_model('emulators/NN_'+ob+'_finalize_multi-dim.h5',
    custom_objects={'mean_sq_err': mean_sq_err})

# List input variables
in_vars = ['medlynslope','dleaf','kmax','fff','dint','baseflow_scalar']
npar = len(in_vars)

# Read in obs targets and calculated variance
obs = np.load(file="obs/obs_"+ob+"_SVD_3modes.npy", allow_pickle=True)
sd = np.load(file="obs/obs_"+ob+"_SVD_3modes_allyrs_sd.npy", allow_pickle=True)

# Define normalized error function
def normerr(x):
    xt = x.reshape(1,-1)
    model_preds = model.predict(xt)
    L = -np.sum(((model_preds-obs)/sd)**2, axis=1)
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

# Set up sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
epochs = 5*10**4
#pos, prob, state = sampler.run_mcmc(p0, epochs)
sampler.run_mcmc(p0, epochs, progress=True)

# Print mean acceptance fraction
acc = sampler.acceptance_fraction
print("Mean acceptance fraction: {0:.3f}"
        .format(np.mean(sampler.acceptance_fraction)))

# Integrated autocorrelation time
tau = sampler.get_autocorr_time(tol=0)
print("Mean autocorrelation time: {0:6.3f}"
        .format(np.mean(tau)))

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
plt.savefig("MCMC_"+ob+"_sampler_chain_5e4epochs.pdf")
plt.show()
#plt.close()

# Marginalized density
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
for i in range(ndim):
    ax = axes[i]
    chain = samples_all[:, :, i].T
    #print(chain.flatten().shape)
    ax.hist(chain.flatten(), 100)
    ax.set_yticks([])
    ax.set_ylabel(labels[i])

axes[-1].set_xlabel("parameter value")
#plt.savefig("MCMC_param_dists_1e6epochs_v2.pdf")
plt.show()
#plt.close()

# Explore autocorrelation estimators for different chain lengths

# first define some functions
def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))
    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
    acf /= 4*n
    # Optionally normalize
    if norm:
        acf /= acf[0]
    return acf

# automated windowing following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

# Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0*np.cumsum(f)-1.0
    window = auto_window(taus, c)
    return taus[window]

def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0*np.cumsum(f)-1.0
    window = auto_window(taus, c)
    return taus[window]

# start with first dimension only
chain = samples_all[:,:,0].T
N = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), 10)).astype(int)
gw2010 = np.empty(len(N))
new = np.empty(len(N))

# automated windowing following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

# Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0*np.cumsum(f)-1.0
    window = auto_window(taus, c)
    return taus[window]

for i, n in enumerate(N):
    gw2010[i] = autocorr_gw2010(chain[:, :n])
    new[i] = autocorr_new(chain[:, :n])

# need to run for a longer chain to get a meaningful figure
plt.loglog(N, gw2010, "o-", label="G&W 2010")
plt.loglog(N, new, "o-", label="new")
ylim = plt.gca().get_ylim()
plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
plt.ylim(ylim)
plt.xlabel("number of samples, $N$")
plt.ylabel(r"$\tau$ estimates")
plt.legend(fontsize=14)
#plt.savefig("MCMC_autocorr_1e4epochs.pdf")
plt.show()
#plt.close()

# Look at the log probs
probs_all = sampler.get_log_prob()
plt.plot(probs_all, "k", alpha=0.3)
plt.xlabel("step number")
plt.ylabel("log prob")
#plt.savefig("MCMC_logprob_1e6epochs_v2.pdf")
plt.show()
#plt.close()

# Flatten chain; discard the initial N steps
#flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
#flat_samples = sampler.get_chain(discard=10000, flat=True)
flat_samples = sampler.get_chain(flat=True)
#print(flat_samples.shape)

# Corner plot
import corner
fig = corner.corner(flat_samples, labels=in_vars)
plt.savefig("MCMC_"+ob+"_corner_5e4epochs.pdf")
#plt.show()
plt.close()

# Get last sample
last_sample = sampler.chain[:,epochs-1,:]
print(np.mean(last_sample, axis=0)) # average (over walkers) last position of each parameter
# Save last sample
#np.save("MCMC_lastsample_1e5epochs_v2", last_sample)
