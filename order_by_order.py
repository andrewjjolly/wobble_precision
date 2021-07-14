"""
Producing Periodograms order by order for the HARPS pipeline data
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

#%%
"""
Detrending data
"""
detrended_rvs = []

for o in range(0,58):
    rvs = results['RV_order{}'.format(o)]-np.median(results['RV_order{}'.format(o)])
    rvs_err = results['RV_order{0}_err'.format(o)]
    model = models.Linear1D()
    fitter = fitting.LinearLSQFitter()
    best_fit = fitter(model,times, rvs) #took out weighting term because of NaN error
    print(best_fit)
    # plt.figure()
    # plt.errorbar(times, rvs, rvs_err, fmt='.k')
    # plt.plot(times, best_fit(times), color = 'r', linewidth = 1)
    # plt.xlabel('time [days]')
    # plt.ylabel('radial velocity [m/s]')
    detrended = [rvs[i] - (best_fit(times))[i] for i in range(0, len(times))]
    detrended_rvs.append(detrended)



# %%
LSP_by_order = [] #lomb-scargil periodogram peak power corresponding period

for o in range(0,58):
    rvs = results['RV_order{}'.format(o)]-np.median(results['RV_order{}'.format(o)])
    rvs_err = results['RV_order{0}_err'.format(o)]
    frequency, power = LombScargle(times, rvs, rvs_err).autopower()
    max_freq = frequency[np.argmax(power)]
    LSP_by_order.append(1/max_freq)
    # plt.figure()
    # plt.errorbar(times, rvs, rvs_err, fmt='.k')
    # plt.xlabel('time [days]')
    # plt.ylabel('radial velocity [m/s]')
    # plt.title('GL667C RVs from Order {}'.format(o))
    #plt.figure()
    #plt.title('LS Periodogram for Order {0}'.format(o))
    #plt.plot(1/frequency, power) #plotting to check
    #plt.xlim(0,200)

LSP_by_order = np.array(LSP_by_order)

# %%
"""
Plotting the Lomb-Scargil Period by echelle order
"""
orders = np.array(range(0,58))
plt.figure()
plt.scatter(orders, LSP_by_order)
plt.plot(orders, np.full(len(orders),7.2004), 'r--')
plt.plot(orders, np.full(len(orders),28.14), 'g--')
plt.plot(orders, np.full(len(orders),62.24	), 'b--')
plt.plot(orders, np.full(len(orders),39.026), 'y--')
plt.plot(orders, np.full(len(orders),91.61), 'm--')
plt.xlabel('Echelle Order')
plt.ylabel('Period from LSP')
plt.ylim(0,100)

# %%
"""
Looking at the RVs for the off scale high periods
"""
for o in orders[LSP_by_order > 100]:
    rvs = results['RV_order{}'.format(o)]-np.median(results['RV_order{}'.format(o)])
    rvs_err = results['RV_order{0}_err'.format(o)]
    rvs_std = np.std(rvs)
    print(rvs_std)
    plt.figure()
    plt.errorbar(times, rvs, rvs_err, fmt='.k')
    plt.plot(times,np.full(len(times),rvs_std),'r--')
    plt.ylabel('RV m/s')
    plt.xlabel('MJD')
    plt.title('Wobble RVs for order {}'.format(o))

# %%

