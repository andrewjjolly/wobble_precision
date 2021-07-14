"""
A FIRST GO AT TRYING TO FIT A RADIAL VELOCITY CURVE TO THE PIPELINE RVS FOR GL667C USING THE EXOPLANET PACKAGE TUTORIAL AS A GUIDE
"""

#%%
"""
Import the things
"""
from datetime import time
import numpy as np
from numpy.core.defchararray import upper
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import pymc3_ext as pmx
import aesara_theano_fallback.tensor as tt
import arviz as az
import os
import wobble
import exoplanet as xo
from astropy.modeling import models, fitting
from astropy.timeseries import LombScargle

#%%
"""
Getting the results data
"""
results = '/home/z5345592/projects/gl667c_wobble/results_rvs_lr20.txt'
results = pd.read_csv(results, skiprows=3, delim_whitespace=True)
results_hdf5 = '/home/z5345592/projects/gl667c_wobble/results_no_bad_orders_lr20.hdf5'
results_hdf5 = wobble.Results(filename=results_hdf5)

# %%
"""
Assigning relevant data to variables
"""
times = np.array(results['dates'])
rvs = np.array(results['pipeline_rv'] - np.median(results['pipeline_rv'])) #pipeline RVs from the HARPS pipeline
rvs_err = np.array(results['pipeline_rv_err']) #pipeline RV errors

# %%
"""
Compute a reference time that is used to normalize the trends model - 
"""
time_ref = 0.5 * (times.min() + times.max()) #this is the midway point in the times


# %%
"""
Plot the data to have a look
"""
plt.errorbar(times, rvs, rvs_err, fmt='.k')
plt.xlabel('time [days]')
plt.ylabel('radial velocity [m/s]')

#%%
"""
fitting a linear trend to the data
"""
model = models.Linear1D()
fitter = fitting.LinearLSQFitter()
best_fit = fitter(model,times, rvs, weights = 1.0 / rvs_err ** 2)
print(best_fit)

#%%
"""
Plotting with the fit
"""
plt.errorbar(times, rvs, rvs_err, fmt='.k')
plt.plot(times, best_fit(times), color = 'r', linewidth = 1)
plt.xlabel('time [days]')
plt.ylabel('radial velocity [m/s]')

#%%
"""
Detrending
"""
detrended = [rvs[i] - (best_fit(times))[i] for i in range(0, len(times))]

#%%
"""
Plotting the detrended data & checking that the slope is basically 0
"""
model = models.Linear1D()
fitter = fitting.LinearLSQFitter()
best_fit_d = fitter(model,times, detrended, weights = 1.0 / rvs_err ** 2)
print(best_fit_d)

plt.errorbar(times, detrended, rvs_err, fmt='.k')
plt.plot(times, best_fit_d(times), color = 'r', linewidth = 1)
plt.xlabel('time [days]')
plt.ylabel('radial velocity [m/s]')

#%%
"""
Reassigning the variables so that they work with the rest of the code.
"""
rvs = np.array(detrended)

#%%
"""
Periodogram to find the peaks in the data
"""
frequency, power = LombScargle(times, rvs, rvs_err).autopower() #run the LS, get freq & power
plt.plot(1/frequency, power) #plotting to check
plt.xlim(6.8,7.5)

#%%
"""
Pulling out the largest frequency peak
"""
max_freq = frequency[np.argmax(power)] #indexing into the frequency using the maximum of the power
period = 1 / max_freq
print(period)

#%%
"""
Orbital parameters - from Delfosse et al 2012 using period as detected using LS above.
In absence of how to calculate uncertainty keeping period err & t_0
"""
periods = period
periods_errs = [0.1 * period]
t0s = [2454443.1 + period/2]
t0s_errs = [0.001]

"""
Making a fine grid that spans the observation window that we can use for plotting.
"""
time_grid = np.linspace(t0s[0],t0s[0] + period,1000) #1000 steps from 5 before and 5 after the first and last obvs


# %%
"""
Use exoplanet to estimate the semiamplitudes of the planets
"""
#this keeps giving me incorrect estimates - so going to hardcode from the paper for now
ks = xo.estimate_semi_amplitude(periods,times,rvs,rvs_err,t0s = t0s)
print(ks, 'm/s')

#ks = [3.8]
# %%
"""
Running the exoplanets model
"""
with pm.Model() as model:
    # Gaussian priors based on HARPS data (from Delfosse et al 2012)
    t0 = pm.Uniform("t0", lower = t0s - period/20, upper = t0s + period/20,  shape=1, testval = t0s)
    logP = pm.Uniform(
        "logP",
        lower = np.log(period - periods_errs),
        upper = np.log(period + periods_errs),
        shape=1,
        testval=np.log(periods),
    )
    P = pm.Deterministic("P", tt.exp(logP))

    # Wide log-normal prior for semi-amplitude
    logK = pm.Normal(
        "logK", mu=np.log(2*ks), sd=2.0, shape=1, testval=np.log(2*ks)
    )

    # Eccentricity & argument of periasteron
    ecs = pmx.UnitDisk("ecs", shape=(2, 1), testval=0.02 * np.ones((2, 1)))
    ecc = pm.Deterministic("ecc", tt.sum(ecs ** 2, axis=0))
    omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))
    xo.eccentricity.vaneylen19(
        "ecc_prior", multi=True, shape=1, fixed=True, observed=ecc
    )

    # Jitter & a quadratic RV trend
    logs = pm.Normal("logs", mu=np.log(np.median(rvs_err)), sd=5.0)
    trend = pm.Normal("trend", mu=0, sd=10.0 ** -np.arange(3)[::-1], shape=3)

    # Then we define the orbit
    orbit = xo.orbits.KeplerianOrbit(period=P, t0=t0, ecc=ecc, omega=omega)

    # And a function for computing the full RV model
    def get_rv_model(time_grid, name=""):
        # First the RVs induced by the planets
        vrad = orbit.get_radial_velocity(time_grid, K=tt.exp(logK))
        pm.Deterministic("vrad" + name, vrad)

        # Define the background model
        A = np.vander(time_grid - time_ref, 3)
        bkg = pm.Deterministic("bkg" + name, tt.dot(A, trend))

        # Sum over planets and add the background to get the full model
        #return pm.Deterministic("rv_model" + name, tt.sum(vrad, axis=-1) + bkg)
        return pm.Deterministic("rv_model" + name, vrad + bkg)

    # Define the RVs at the observed times
    rv_model = get_rv_model(times)

    # Also define the model on a fine grid as computed above (for plotting)
    rv_model_pred = get_rv_model(time_grid, name="_pred")

    # Finally add in the observation model. This next line adds a new contribution
    # to the log probability of the PyMC3 model
    err = tt.sqrt(rvs_err ** 2 + tt.exp(2 * logs))
    pm.Normal("obs", mu=rv_model, sd=err, observed=rvs)    

"""
This is not an amazing fit so need to fit for the maximum a posterior parameters.
"""
with model:
    map_soln = pmx.optimize(start=model.test_point, vars=[trend])
    map_soln = pmx.optimize(start=map_soln, vars=[t0, trend, logK, logP, logs])
    map_soln = pmx.optimize(start=map_soln, vars=[ecs])
    map_soln = pmx.optimize(start=map_soln)

# %%
"""
Plotting the model
"""
plt.errorbar(np.mod(times, period), rvs, rvs_err, fmt=".k")
plt.plot(np.mod(time_grid,period), map_soln["vrad_pred"], "--k", alpha=0.5)
plt.plot(np.mod(time_grid,period), map_soln["bkg_pred"], ":k", alpha=0.5)
plt.plot(np.mod(time_grid,period), map_soln["rv_model_pred"], label="model")

plt.legend(fontsize=10)
#plt.xlim(2454500, 2454750)
plt.xlabel("time [days]")
plt.ylabel("radial velocity [m/s]")
_ = plt.title("MAP model")

# %%
"""
Sampling & Convergence checks
"""
np.random.seed(42)
with model:
    trace = pmx.sample(
        tune=4000,
        draws=4000,
        cores=2,
        chains=2,
        target_accept=0.95,
    )

az.summary(
    trace, var_names=["trend", "logs", "omega", "ecc", "t0", "logK", "P"])

# %%
import corner

with model:
    _ = corner.corner(trace, var_names=["P","t0", "logK", "ecc", "omega"])
# %%
#%%
"""
Calculating residuals between model RV & data
"""
phased_time = np.mod(times,period)
data = rvs
data_err = np.array(rvs_err)
model = map_soln["rv_model"]
residuals = data - model

# %%
"""
Working out standard deviation of residiuals & identifying those > 2 sigma
"""
sigma = np.std(residuals) #calculating the standard deviation of the residuals
residuals_abs = np.abs(residuals) #converting residuals into absolute values
residuals_large_sig = residuals[residuals_abs > 2*sigma] #which residuals are > 2*sigma
times_large_sig = times[residuals_abs > 2*sigma] #which times correspond to the large sigmas
rvs_large_sig = data[residuals_abs > 2*sigma] #which are the rvs that have large sigmas in residuals
# %%
"""
Overplotting all the RVs with the RVs with large sigma residuals from RV model
"""
plt.figure()
plt.scatter(times, data)
plt.scatter(times_large_sig, rvs_large_sig)
# %%
"""
Subtracting model from data to run again 
"""
rvs_planet_removed = rvs - model

# %%
"""
Periodogram 2nd time
"""
star_name = 'GL667C (HARPS Pipeline)'

frequency, power = LombScargle(times, rvs_planet_removed, rvs_err).autopower() #run the LS, get freq & power
plt.figure()
plt.plot(frequency, power) #plotting to check
#plt.xlim(0,150)
plt.xlabel('frequency [days$^{-1}$]')
plt.ylabel('Lomb-Scargle Power')
plt.title("LS Periodogram for {0}".format(star_name))

plt.figure()
frequency, power = LombScargle(times, rvs_planet_removed, rvs_err).autopower() #run the LS, get freq & power
plt.plot(1/frequency, power) #plotting to check
plt.xlim(0,150)
plt.xlabel('period [days]')
plt.ylabel('Lomb-Scargle Power')
plt.title("LS Periodogram for {0}".format(star_name))

#%%
"""
Pulling out the largest frequency peak
"""
max_freq = frequency[np.argmax(power)] #indexing into the frequency using the maximum of the power
period = 1 / max_freq
print(period)

#%%
"""
Orbital parameters - from Delfosse et al 2012 using period as detected using LS above.
In absence of how to calculate uncertainty keeping period err & t_0
"""
periods = period
periods_errs = [0.01 * period]
t0s = [2454462 + period] #using the value for t_0 for the 28 day period planet
t0s_errs = [0.001]

"""
Making a fine grid that spans the observation window that we can use for plotting.
"""
time_grid = np.linspace(t0s[0],t0s[0] + period,1000) #1000 steps from 5 before and 5 after the first and last obvs
# %%
"""
Use exoplanet to estimate the semiamplitudes of the planets
"""
ks = xo.estimate_semi_amplitude(periods,times,rvs,rvs_err,t0s = t0s)
print(ks, 'm/s')# %%

# %%
"""
Running the exoplanets model
"""
with pm.Model() as model:
    # Gaussian priors based on HARPS data (from Delfosse et al 2012)
    t0 = pm.Uniform("t0", lower = t0s - period/20, upper = t0s + period/20,  shape=1, testval = t0s)
    logP = pm.Uniform(
        "logP",
        lower = np.log(period - periods_errs),
        upper = np.log(period + periods_errs),
        shape=1,
        testval=np.log(periods),
    )
    P = pm.Deterministic("P", tt.exp(logP))

    # Wide log-normal prior for semi-amplitude
    logK = pm.Normal(
        "logK", mu=np.log(2*ks), sd=2.0, shape=1, testval=np.log(2*ks)
    )

    # Eccentricity & argument of periasteron
    ecs = pmx.UnitDisk("ecs", shape=(2, 1), testval=0.02 * np.ones((2, 1)))
    ecc = pm.Deterministic("ecc", tt.sum(ecs ** 2, axis=0))
    omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))
    xo.eccentricity.vaneylen19(
        "ecc_prior", multi=True, shape=1, fixed=True, observed=ecc
    )

    # Jitter & a quadratic RV trend
    logs = pm.Normal("logs", mu=np.log(np.median(rvs_err)), sd=5.0)
    trend = pm.Normal("trend", mu=0, sd=10.0 ** -np.arange(3)[::-1], shape=3)

    # Then we define the orbit
    orbit = xo.orbits.KeplerianOrbit(period=P, t0=t0, ecc=ecc, omega=omega)

    # And a function for computing the full RV model
    def get_rv_model(time_grid, name=""):
        # First the RVs induced by the planets
        vrad = orbit.get_radial_velocity(time_grid, K=tt.exp(logK))
        pm.Deterministic("vrad" + name, vrad)

        # Define the background model
        A = np.vander(time_grid - time_ref, 3)
        bkg = pm.Deterministic("bkg" + name, tt.dot(A, trend))

        # Sum over planets and add the background to get the full model
        #return pm.Deterministic("rv_model" + name, tt.sum(vrad, axis=-1) + bkg)
        return pm.Deterministic("rv_model" + name, vrad + bkg)

    # Define the RVs at the observed times
    rv_model = get_rv_model(times)

    # Also define the model on a fine grid as computed above (for plotting)
    rv_model_pred = get_rv_model(time_grid, name="_pred")

    # Finally add in the observation model. This next line adds a new contribution
    # to the log probability of the PyMC3 model
    err = tt.sqrt(rvs_err ** 2 + tt.exp(2 * logs))
    pm.Normal("obs", mu=rv_model, sd=err, observed=rvs)    

"""
This is not an amazing fit so need to fit for the maximum a posterior parameters.
"""
with model:
    map_soln = pmx.optimize(start=model.test_point, vars=[trend])
    map_soln = pmx.optimize(start=map_soln, vars=[t0, trend, logK, logP, logs])
    map_soln = pmx.optimize(start=map_soln, vars=[ecs])
    map_soln = pmx.optimize(start=map_soln)

# %%
"""
Plotting the model
"""
plt.errorbar(np.mod(times, period), rvs, rvs_err, fmt=".k")
plt.plot(np.mod(time_grid,period), map_soln["vrad_pred"], "--k", alpha=0.5)
plt.plot(np.mod(time_grid,period), map_soln["bkg_pred"], ":k", alpha=0.5)
plt.plot(np.mod(time_grid,period), map_soln["rv_model_pred"], label="model")

plt.legend(fontsize=10)
#plt.xlim(2454500, 2454750)
plt.xlabel("time [days]")
plt.ylabel("radial velocity [m/s]")
_ = plt.title("MAP model")

# %%
