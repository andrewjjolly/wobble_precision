#%%
"""
PACKAGE IMPORTS
"""

from datetime import time
import numpy as np
from numpy.core.defchararray import upper
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
import matplotlib.cm as cm

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
    plt.title('{}'.format(star_name))
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

def detrend_rvs(times, rvs, rvs_err):
    gradient, intercept, gradient_err, intercept_err = linear_fit(times, rvs, rvs_err)
    detrended_rvs = [rvs[i] - (straight_line(times, gradient, intercept))[i] for i in range(0, len(times))]

    return detrended_rvs, gradient, gradient_err

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

def create_rvs_array(results, no_of_orders):
    rvs_all_orders = []
    for order in range(no_of_orders):
        order_rvs = np.array(results['RV_order{}'.format(order)])
        rvs_all_orders.append(order_rvs)
    rvs_all_orders = np.array(rvs_all_orders)

    return rvs_all_orders

#detrend_order_rvs has an error where it is shifting the RVS way down in radial velocity, this might be causeing my problem.

def detrend_order_rvs(times, rvs_array, rvs_err_array, no_of_orders, trend, trend_err):
    detrended_rvs_all_orders = []
    residual_trends = []
    residual_trends_err = []
    for order in range(no_of_orders):
        order_rvs = rvs_array[order, :]
        order_rvs_err = rvs_err_array[order, :]
        order_rvs_detrended = [order_rvs[i] - (straight_line(times, trend, trend_err))[i] for i in range(0, len(times))]
        order_trend, trend_cov = curve_fit(straight_line, times, order_rvs_detrended, sigma = order_rvs_err)
        order_trend_err = np.sqrt(np.diag(trend_cov))
        order_rvs_detrended = normalise_rvs(order_rvs_detrended) #added this in to see if it fixes things.
        detrended_rvs_all_orders.append(order_rvs_detrended)
        residual_trends.append(order_trend[0])
        residual_trends_err.append(order_trend_err[0])
    detrended_rvs_all_orders = np.array(detrended_rvs_all_orders)
    residual_trends = np.array(residual_trends)
    residual_trends_err = np.array(residual_trends_err)
   
    return detrended_rvs_all_orders, residual_trends, residual_trends_err

def sine_wave(x_data, amplitude, phase_shift):
    return amplitude * (np.sin((2 * np.pi) * (x_data + phase_shift)))

def subtract_periodic_signal(period, times, rvs, rvs_err):
    phase_folded = (times / period) % 1
    popt, pcov = curve_fit(sine_wave, phase_folded, rvs, sigma = rvs_err)
    signal_subtracted = rvs - sine_wave(phase_folded, popt[0], popt[1])

    return signal_subtracted

def lsp_snr_at_period(times, rvs, rvs_err):
    frequency, power = LombScargle(times, rvs, rvs_err).autopower() #assigning the frequency and power from the lsp
    period = 1 / frequency #turning frequency into period
    noise_b = np.std(power[(period > (lit_period_b-0.5*lit_period_b)) & (period < (lit_period_b + 0.5*lit_period_b))])
    noise_c = np.std(power[(period > (lit_period_c-0.5*lit_period_c)) & (period < (lit_period_c + 0.5*lit_period_c))])
    snr_b = power / noise_b
    snr_c = power / noise_c
    snr_at_b = snr_b[find_nearest(period, lit_period_b)]
    snr_at_c = snr_c[find_nearest(period, lit_period_c)]

    return snr_at_b, snr_at_c

def assign_wavelengths(data, no_of_orders):

    wavelengths = []
    wavelengths_err = []

    for order in range(no_of_orders):
        order_wl = (np.exp(np.mean(data.xs[order][0]))/10)
        order_wl_err = (np.exp(np.std(data.xs[order][0]))/10)
        wavelengths.append(order_wl)
        wavelengths_err.append(order_wl_err)

    wavelengths = np.array(wavelengths)
    wavelengths_err = np.array(wavelengths_err)

    return wavelengths, wavelengths_err

def load_data(data_filename, good_epochs, bad_orders):
    data = wobble.Data(filename=data_filename, epochs=good_epochs)
    data.delete_orders(bad_orders) #deletes the bad orders that have been identified in the 'bad orders' list above.
    data.drop_bad_orders(min_snr=3) #dropping bad orders
    data.drop_bad_epochs(min_snr=3) #dropping bad epochs

    return data

def count_orders(data):
    
    no_of_orders = len(data.orders)

    return no_of_orders

def create_combined_orders(results):
    combined_order_rvs = np.array(results['RV'])
    combined_order_rvs = normalise_rvs(combined_order_rvs)
    combined_order_rvs_err = np.array(results['RV_err'])

    return combined_order_rvs, combined_order_rvs_err    

#%%
data_dir = '/home/z5345592/projects/wobble_precision/data/'
data_filename = '/home/z5345592/projects/wobble_precision/data/gl667c.hdf5'

star_name = 'GL667C'

lit_period_b = 7.2
lit_period_c = 28.1

lit_amplitude_b = 3.9
lit_amplitude_c = 1.7

good_epochs = [ 0,  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
         13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
         26,  27,  28,  30,  31,  32,  33,  34,  36,  37,  38,  39,  40,
         41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,
         54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,
         67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  80,
         81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,
         94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 106, 107,
        108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
        121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
        134, 135, 136, 137, 138, 139, 141, 142, 143, 144, 145, 146, 147,
        148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
        161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
        174, 175, 176]

bad_orders = np.array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False,  True, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False,  True, False, False, False, False, False,
       False,  True,  True, False, False,  True,  True,  True,  True])

data = load_data(data_filename, good_epochs, bad_orders)

no_of_orders = count_orders(data)

wavelengths, wavelengths_err = assign_wavelengths(data, no_of_orders)

times = np.array(data.dates)

rv_lr_list = [20, 50, 100, 200, 400, 1000] #list of learning rates that I have results for

snr_at_b_all_lr = []
snr_at_c_all_lr = []

rvs_planet_b_all_lr = []
rvs_planet_c_all_lr = []

for lr in rv_lr_list:
    
    txt_results_filename = data_dir + 'results_rvs_lr{}.txt'.format(lr)
    hdf5_results_filename = data_dir + 'results_no_bad_orders_lr{}.hdf5'.format(lr)

    results, results_hdf5 = load_results(txt_results_filename, hdf5_results_filename)

    combined_order_rvs, combined_order_rvs_err = create_combined_orders(results)
    
    rvs_all_orders = create_rvs_array(results, no_of_orders)
    rvs_all_orders = normalise_rvs(rvs_all_orders)
    rvs_all_orders_err = create_rvs_err_array(results, no_of_orders)

    rvs_all_orders_err, rvs_err_nans = nan_to_inf(rvs_all_orders_err)

    detrended_rvs_combined, combined_rvs_trend, combined_rvs_trend_err = detrend_rvs(times, combined_order_rvs, combined_order_rvs_err)

    detrended_rvs_all_orders, residual_trends, residual_trends_err = detrend_order_rvs(times, rvs_all_orders, rvs_all_orders_err, no_of_orders, combined_rvs_trend, combined_rvs_trend_err)

    rvs_planet_b_removed = subtract_periodic_signal(lit_period_b, times, detrended_rvs_combined, combined_order_rvs_err)
    rvs_planet_c_removed = subtract_periodic_signal(lit_period_c, times, detrended_rvs_combined, combined_order_rvs_err)

    snr_at_b_all_orders = []
    snr_at_c_all_orders = []

    for order in range(no_of_orders):
        rvs_planet_b_removed = subtract_periodic_signal(lit_period_b, times, detrended_rvs_all_orders[order,:], rvs_all_orders_err[order,:])
        rvs_planet_c_removed = subtract_periodic_signal(lit_period_c, times, detrended_rvs_all_orders[order,:], rvs_all_orders_err[order,:])
        snr_at_b_no_c, snr_at_c_no_c = lsp_snr_at_period(times, rvs_planet_c_removed, rvs_all_orders_err[order,:])
        snr_at_b_no_b, snr_at_c_no_b = lsp_snr_at_period(times, rvs_planet_b_removed, rvs_all_orders_err[order,:])
        snr_at_b_all_orders.append(snr_at_b_no_c)
        snr_at_c_all_orders.append(snr_at_c_no_b)

    snr_at_b_all_orders = np.array(snr_at_b_all_orders)
    snr_at_c_all_orders = np.array(snr_at_c_all_orders)

    snr_at_b_all_lr.append(snr_at_b_all_orders)
    snr_at_c_all_lr.append(snr_at_c_all_orders)

snr_at_b_all_lr = np.array(snr_at_b_all_lr)
snr_at_c_all_lr = np.array(snr_at_c_all_lr)

colours = cm.get_cmap('magma')
colours = colours(np.linspace(0,1,6))

plt.figure()
for lr in range(len(rv_lr_list)):
    plt.plot(wavelengths, snr_at_b_all_lr[lr,:], label=str(rv_lr_list[lr]), c=colours[lr])
    plt.xlabel('wavelength, nm')
    plt.ylabel('LSP Power / std of LSP period $\pm$ 1 day')
    plt.legend()
    plt.title('LR performance GL667Cb')
    plt.yscale('log')

plt.figure()
for lr in range(len(rv_lr_list)):
    plt.plot(wavelengths, snr_at_c_all_lr[lr,:], label=str(rv_lr_list[lr]), c=colours[lr])
    plt.xlabel('wavelength, nm')
    plt.ylabel('LSP Power / std of LSP period $\pm$ 1 day')
    plt.legend()
    plt.title('LR performance GL667Cc')
    plt.yscale('log')

# %%
"""
Plotting the RVs for the orders identified
"""

orders_of_interest = [17,22,41,42,47,51] #identified manually by looking at the plot
rv_lr_list = [20, 50, 100, 200, 400, 1000]

for lr in rv_lr_list:

    txt_results_filename = data_dir + 'results_rvs_lr{}.txt'.format(lr)
    hdf5_results_filename = data_dir + 'results_no_bad_orders_lr{}.hdf5'.format(lr)
    results, results_hdf5 = load_results(txt_results_filename, hdf5_results_filename)

    rvs_all_orders = create_rvs_array(results, no_of_orders)
    rvs_all_orders = normalise_rvs(rvs_all_orders)

    rvs_all_orders_err = create_rvs_err_array(results, no_of_orders)
    rvs_all_orders_err, rvs_err_nans = nan_to_inf(rvs_all_orders_err)

    for order in orders_of_interest:
        order_rvs_b = subtract_periodic_signal(lit_period_c, times, rvs_all_orders[order,:], rvs_all_orders_err[order,:])
        plt.figure()
        plot_rvs(times, order_rvs_b, rvs_all_orders_err[order, :], 'RV for order {} and LR={}'.format(order, lr))
        frequency, power = LombScargle(times, order_rvs_b, rvs_all_orders_err[order,:]).autopower() #assigning the frequency and power from the lsp
        period = 1 / frequency #turning frequency into period
        noise_b = np.std(power[(period > (lit_period_b-0.1*lit_period_b)) & (period < (lit_period_b + 0.1*lit_period_b))])
        snr_b = power / noise_b
        snr_at_b = snr_b[find_nearest(power, lit_period_b)]
        plt.figure()
        plt.plot(period, snr_b)
        plt.title('Periodogram for order {} and LR={}'.format(order, lr))
        plt.xlim(6,8)
        plt.ylim(0,10)
        plt.axvline(lit_period_b, 'r.')

# %%
