# MCMC for optimizing the 2-layer multiple output Neural Network
# Try to get parallel processing to speed up compute time
# 8/21/19

# Import some modules
import emcee
import numpy as np
import matplotlib.pyplot as plt

# Load previously trained NN models
#import keras.backend as K
#def mean_sq_err(y_true,y_pred):
#    return K.mean((y_true-y_pred)**2)
#from keras.models import load_model
#model_GPP = load_model('emulators/NN_GPP_finalize_multi-dim.h5',
#    custom_objects={'mean_sq_err': mean_sq_err})
#model_LHF = load_model('emulators/NN_LHF_finalize_multi-dim.h5',
#    custom_objects={'mean_sq_err': mean_sq_err})

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
B = 1.49
#B = 1
def normerr(x):
    import keras.backend as K
    def mean_sq_err(y_true,y_pred):
        return K.mean((y_true-y_pred)**2)
    from keras.models import load_model
    model_GPP = load_model('emulators/NN_GPP_finalize_multi-dim.h5',
            custom_objects={'mean_sq_err': mean_sq_err})
    model_LHF = load_model('emulators/NN_LHF_finalize_multi-dim.h5',
            custom_objects={'mean_sq_err': mean_sq_err})
    xt = x.reshape(1,-1)
    model_preds_GPP = model_GPP.predict(xt)
    model_preds_LHF = model_LHF.predict(xt)
    L = -(np.sum(((model_preds_GPP-obs_GPP)/sd_GPP)**2, axis=1) +\
        B*np.sum(((model_preds_LHF-obs_LHF)/sd_LHF)**2, axis=1))
    return L

# Define the prior
# Uniform distribution between [0,1]
def lnprior(x):
#    if all(x > 0) and all(x < 1):
#    the following line doesn't work (bad logicals)
#    if x.all() > 0 and x.all() < 1:
    if np.all(x > 0) and np.all(x < 1):
        return 0.0
    return -np.inf

# Define the full log probability function
def lnprob(x):
    lp = lnprior(x)
    if not np.isfinite(lp):
        return -np.inf
    return lp + normerr(x)

# Trying to figure out Pool issues
#import pickle
#print("Imported pickle")
#test1 = pickle.dumps(lnprior)
#test2 = pickle.dumps(normerr)
#test3 = pickle.dumps(lnprob)
#print(test1,test2,test3)
#print("Dumped pickle")

# Set number of walkers, number of dims (= number of params)
#nwalkers = 200
nwalkers = 15
ndim = npar

# Initialize walkers (random initial states)
p0 = [np.random.rand(ndim) for i in range(nwalkers)]

# Multiprocessing
#from multiprocessing import set_start_method
#set_start_method('spawn', force=True)
from multiprocessing import Pool
#pool = Pool()
#pool = Pool(processes=10)
#print("Starting pool")
#from multiprocessing import get_context 
#with get_context("forkserver").Pool() as pool:
with Pool(processes=2) as pool:
#    print("Define sampler")
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool) 
    epochs = 1*10**1
#    print("Start sampler")
    sampler.run_mcmc(p0, epochs, progress=True)
#    print("Finish sampler")

# Threads
#sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=4)
#epochs = 1*10**3
#sampler.run_mcmc(p0, epochs, progress=True)

# Set up a "runner" to test different configs
#import timeit
#from itertools import product
# From Anderson
#def runner(njobs, epochs=5*10**3, ndim=npar, nwalkers=200, p0=p0, verbose=False):
#    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=njobs)
#    # Save the timings into a variable for later usage?
#    sampler.run_mcmc(p0, epochs, progress=True)
#    if verbose:
#        print(f"nthreads = {njobs}, epochs={epochs}")
#    profiler = {}
#    profiler['nthreads'] = njobs
#    #profiler['best_time'] = time.best
#    profiler['epochs'] = epochs
#    return profiler

# Create a cartesian product of parameters to try out
#parameters = list(product([1, 2], [5000]))

#timings = []
#for option in parameters:
#    timings.append(runner(njobs=option[0], epochs=option[1], verbose=True))

# Serial
#sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
# Run sampler for set number of epochs
# should really think about ~10^6 epochs (batch job?)
#epochs = 1*10**2
#pos, prob, state = sampler.run_mcmc(p0, epochs)
#sampler.run_mcmc(p0, epochs, progress=True)

# Print mean acceptance fraction
acc = sampler.acceptance_fraction
#print(acc)
print("Mean acceptance fraction: {0:.3f}"
        .format(np.mean(sampler.acceptance_fraction)))

# Integrated autocorrelation time
tau = sampler.get_autocorr_time(tol=0)
#print(tau.shape)
#print(tau)
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
#plt.savefig("MCMC_sampler_chain_1e3epochs.pdf")
#plt.savefig("MCMC_sampler_chain_1e4epochs.pdf")
#plt.savefig("MCMC_sampler_chain_2e4epochs.pdf")
#plt.savefig("MCMC_sampler_chain_2e4epochs_v2.pdf")
#plt.savefig("MCMC_sampler_chain_5e4epochs_v2.pdf")
#plt.savefig("MCMC_sampler_chain_1e5epochs_v2.pdf")
#plt.savefig("MCMC_sampler_chain_1e4epochs_v2.pdf")
#plt.savefig("MCMC_sampler_chain_2e5epochs_v2.pdf")
plt.show()

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
#plt.savefig("MCMC_param_dists_2e5epochs_v2.pdf")
#plt.savefig("MCMC_param_dists_1e5epochs_v2.pdf")
#plt.savefig("MCMC_param_dists_1e4epochs_v2.pdf")
plt.show()


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
#plt.savefig("MCMC_autocorr_2e4epochs.pdf")
#plt.savefig("MCMC_autocorr_2e4epochs_v2.pdf")
#plt.savefig("MCMC_autocorr_5e4epochs_v2.pdf")
#plt.savefig("MCMC_autocorr_1e5epochs_v2.pdf")
#plt.savefig("MCMC_autocorr_2e5epochs_v2.pdf")
plt.show()


# Look at the log probs
probs_all = sampler.get_log_prob()
#probs_all = sampler.get_log_prob(discard=100)
#print(probs_all.shape)
#print(np.min(probs_all))
#print(np.max(probs_all))
#probs_all_scaled = probs_all*(10**(-7))
#print(np.min(probs_all_scaled))
#print(np.max(probs_all_scaled))
plt.plot(probs_all, "k", alpha=0.3)
#err_all = np.exp(probs_all)
#plt.plot(err_all, "k", alpha=0.3)
#err_all_scaled = np.exp(-probs_all_scaled)
#plt.plot(err_all_scaled, "k", alpha=0.3)
plt.xlabel("step number")
plt.ylabel("log prob")
#plt.ylabel("prob")
#plt.ylabel("scaled error")
#plt.savefig("MCMC_logprob_2e4epochs.pdf")
#plt.savefig("MCMC_scaled_error_1e3epochs.pdf")
#plt.savefig("MCMC_scaled_error_1e4epochs.pdf")
#plt.savefig("MCMC_scaled_error_2e4epochs.pdf")
#plt.savefig("MCMC_scaled_error_2e4epochs_v2.pdf")
#plt.savefig("MCMC_scaled_error_5e4epochs_v2.pdf")
#plt.savefig("MCMC_scaled_error_1e5epochs_v2.pdf")
#plt.savefig("MCMC_prob_5e4epochs_v2.pdf")
#plt.savefig("MCMC_logprob_1e4epochs_v2.pdf")
#plt.savefig("MCMC_logprob_2e5epochs_v2.pdf")
plt.show()

# Discard the initial N steps
#samples = sampler.chain[:, 500:, :].reshape((-1, ndim))
#flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
#flat_samples = sampler.get_chain(discard=100, flat=True)
flat_samples = sampler.get_chain(flat=True)
#print(flat_samples.shape)

# Corner plot
import corner
fig = corner.corner(flat_samples, labels=in_vars)
#plt.savefig("MCMC_corner_1e3epochs.pdf")
#plt.savefig("MCMC_corner_1e4epochs.pdf")
#plt.savefig("MCMC_corner_2e4epochs.pdf")
#plt.savefig("MCMC_corner_2e4epochs_thin15.pdf")
#plt.savefig("MCMC_corner_2e4epochs_v2.pdf")
#plt.savefig("MCMC_corner_5e4epochs_v2.pdf")
#plt.savefig("MCMC_corner_1e5epochs_v2.pdf")
#plt.savefig("MCMC_corner_1e4epochs_v2.pdf")
#plt.savefig("MCMC_corner_2e5epochs_v2.pdf")
plt.show()

# Get last sample
#last_sample = sampler.get_last_sample()
#print(last_sample)
