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
from scipy.optimize import curve_fit

#%%

"""
FUNCTIONS
"""

def load_results(results_txt, results_hdf5):
    results = results_txt
    results = pd.read_csv(results, skiprows=3, delim_whitespace=True)
    results_hdf5 = results_hdf5
    results_hdf5 = wobble.Results(filename = results_hdf5)

    return results, results_hdf5

def straight_line (x, m, c):
    return (m * x) + c

def find_nearest(array, value):
    array = np.array(array)
    idx = ((np.abs(array - value)).argmin())
    return idx

def plot_rvs(times, rvs, rvs_err, star_name):
    rvs = rvs - np.median(rvs)
    plt.figure()
    plt.errorbar(times, rvs, rvs_err, fmt='.k')
    plt.title('Combined Orders for {}'.format(star_name))
    plt.ylabel('Radial Velocity [ms$^{-1}$]')
    plt.xlabel('MJD')

def linear_fit(times, rvs, rvs_err):
    popt, pcov = curve_fit(straight_line, times, rvs, sigma = rvs_err)
    perr = np.sqrt(np.diag(pcov))
    gradient = popt[0]
    intercept = popt[1]
    gradient_err = perr[0]
    intercept_err = perr[1]
    return gradient, intercept, gradient_err, intercept_err

def normalise_rvs(rv_data):
    normalised_rvs = rv_data - np.median(rv_data)
    return normalised_rvs

def plot_rvs(times, rvs, rvs_err, star_name):
    rvs = normalise_rvs(rvs)
    gradient, intercept, gradient_err, intercept_err = linear_trend(times, rvs, rvs_err)
    plt.figure()
    plt.errorbar(times, rvs, rvs_err, fmt='.k')
    plt.plot(times, straight_line(times, gradient, intercept))w
    plt.title('Combined Orders for {}'.format(star_name))
    plt.ylabel('Radial Velocity [ms$^{-1}$]')
    plt.xlabel('MJD')

def detrend_rvs(times, rvs, rvs_err):
    gradient, intercept, gradient_err, intercept_err = linear_fit(times, rvs, rvs_err)
    detrended_rvs = [rvs[i] - (straight_line(times, gradient, intercept))[i] for i in range(0, len(times))]
    return detrended_rvs

def create_rvs_err_array(results, no_of_orders):
    rvs_errs_all_orders = []
    for order in range(no_of_orders):
        order_rvs_err = np.array(results['RV_order{}_err'.format(order)])
        rvs_errs_all_orders.append(order_rvs_err)
    rvs_errs_all_orders = np.array(rvs_errs_all_orders)
    return rvs_errs_all_orders

def nan_to_inf(rvs_err_array):
    nan_boolean = np.isnan(rvs_err_array) #creating a boolean mask for the location of the nans in the array
    rvs_err_array[nan_boolean] = np.inf #using the mask to turn the nans into infs
    return rvs_err_array, nan_boolean

def plot_nan_locations(nan_boolean):
    plt.figure()
    plt.imshow(nan_boolean, cmap='gray_r')
    plt.title('Location of NaN in RV errors')
    plt.xlabel('Epoch')
    plt.ylabel('Order')  
    plt.ylim(0,no_of_orders) 

#%%



data_dir = '/home/z5345592/projects/gl667c_wobble/results'
star_name = 'GL667C'

rv_lr_list = [20,50,100,200,400,1000] #list of learning rates that I have results for

for lr in rv_lr_list:
    
    txt_results_filename = data_dir + 'results_rvs_lr{}.txt'.format(lr)
    hdf5_results_filename = data_dir + 'results_no_bad_orders_lr{}.hdf5'.format(lr)

    load_results(txt_results_filename, hdf5_results_filename)

    times = np.array(results['dates'])
    combined_order_rvs = np.array(results['RV'])
    combined_order_rvs_err = np.array(results['RV_err'])