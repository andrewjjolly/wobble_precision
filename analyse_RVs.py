#%%
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import wobble
from IPython.display import Image

#%%
bad_orders = np.array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False,  True, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False,  True, False, False, False, False, False,
       False,  True,  True, False, False,  True,  True,  True,  True])
#%%
good_epochs = [ 0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
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

#%%
#data = wobble.Data(filename='/home/ajolly/projects/gl667c_wobble/data/gl667c.hdf5', epochs=good_epochs) #taiga
data = wobble.Data(filename='/home/z5345592/projects/gl667c_wobble/data/gl667c.hdf5', epochs=good_epochs) #laptop for testing
#%%
data.delete_orders(bad_orders) #deletes the bad orders that have been identified in the 'bad orders' list above. Think of a way to do this beforehand?
data.drop_bad_orders(min_snr=3) #dropping bad orders
data.drop_bad_epochs(min_snr=3) #dropping bad epochs
#%%
results = '/home/z5345592/projects/gl667c_wobble/results_rvs_no_bad_orders.txt'
results = pd.read_csv(results, skiprows=3, delim_whitespace=True)
results_hdf5 = '/home/z5345592/projects/gl667c_wobble/results_no_bad_orders.hdf5'
results_hdf5 = wobble.Results(filename=results_hdf5)

# %%
dates = results['dates']
rvs = results['RV'] - np.mean(results['RV'])
rvs_err = results['RV_err']
plrvs = results['pipeline_rv'] - np.mean(results['pipeline_rv'])
plrvs_err = results['pipeline_rv_err']
# %%
"""
plotting the normalised RVs from wobble against the pipeline RVs
"""
plt.figure()
plt.errorbar(dates, rvs, rvs_err, fmt='o',ms=5, elinewidth=1, c='k', label='wobble')
plt.errorbar(dates, plrvs-75, plrvs_err, fmt='o', ms=5, elinewidth=1, c='r', label='pipeline') #factor is to plot them on top of each other
plt.xlabel('MJD')
plt.ylabel('normalised RV')
plt.legend()
#plt.ylim(-3950,-3500)
#plt.xlim(2454250,2454700)

#%%
"""
plotting point by point residuals
"""
plt.figure()
plt.scatter(dates, rvs - plrvs)

# %%
"""
plotting RV by date for all orders
"""

colours = cm.rainbow(np.linspace(0,1,np.size(results_hdf5.orders)))
plt.figure()
for o in (range(0,np.size(results_hdf5.orders))):
    dates = results['dates']
    rv = (results['RV_order' + str(o)]) - (np.mean(results['RV_order' + str(o)]))
    rv_err = results['RV_order' + str(o) + '_err']
    plt.scatter(dates,rv, color=colours[o])
# %%
"""
plotting RV scatter by wavelength
"""
colours = cm.rainbow(np.linspace(0,1,64))
plt.figure
for o in range(0,64):
    rv_scatter = np.std(results['RV_order' + str(o)])
    wavelengths = np.exp(data.xs[o][0])
    wavelengths_med = (np.median(wavelengths)) / 10
    plt.scatter(wavelengths_med, rv_scatter, color=colours[o])
    plt.xlabel('median wavelength of echelle order (nm)')
    plt.ylabel('std dev of RV for echelle order')

# %%
"""
Plotting spectra for particular orders
"""
for o in range(0,5):
    results_hdf5.plot_spectrum(o,0,data,'gl667c_order{}.png'.format(o))
# %%
# %%
order_stdev = []
for o in range(0,64):
    rv_scatter = np.std(results['RV_order' + str(o)])
    order_stdev.append(rv_scatter)
order_stdev = np.array(order_stdev)
rv_mask = order_stdev > 15
# %%
"""
plotting just the pipeline RVs
"""
plt.figure()
plt.errorbar(dates, plrvs, plrvs_err, fmt='o', ms=5, elinewidth=1, c='r', label='pipeline')
plt.xlabel('MJD')
plt.ylabel('normalised RV')
plt.legend()
plt.ylim(70,105)
#plt.xlim(2454250,2454700)

#clearly 5 points that are significantly below the mean - plotting the mean and the 5 points seperately:
#%%
good_plrvs_mask = plrvs > 70 #define the good plrvs as anything that has an RV above 70m/s after looking at data

good_dates = dates[good_plrvs_mask]
good_plrvs = plrvs[good_plrvs_mask]
good_plrvs_err = plrvs_err[good_plrvs_mask]

# %%
"""
plotting them again
"""
plt.figure()
plt.errorbar(good_dates, good_plrvs, good_plrvs_err, fmt='o', ms=5, elinewidth=1, c='r', label='pipeline')
plt.xlabel('MJD')
plt.ylabel('RV')
plt.legend()

# %%
"""
ok that looks good - now just want to normalise them so they are around the mean.
"""

x = good_dates
y = good_plrvs - np.mean(good_plrvs) #just subtracting off the mean so they sit around zero
y_err = good_plrvs_err #these don't need to change - the uncertainty remains

plt.figure()
plt.errorbar(x, y, y_err, fmt='o', ms=5, elinewidth=1, c='r', label='pipeline')
plt.xlabel('MJD')
plt.ylabel('Normalised RV')
plt.legend()
# %%
"""
Fine - that's good for the good RVs, but what about wanting to know what the dates of the bad RVs are so we can
look at the spectrum and decide whether to include or not include them.
"""
bad_epochs = data.epochs[~good_plrvs_mask] #creating an array of the indices of the 'bad epochs' from the inverse of the mask
for e in bad_epochs:
    results_hdf5.plot_spectrum(35,e, data,'gl667c_order_35_epoch_{}.png'.format(e))
    Image('gl667c_order_35_epoch_{}.png'.format(e))

#%%
"""
Now plotting same order for a good epoch. I will chose the one closest to the zero RV.
"""
for e in range(10,15):
    results_hdf5.plot_spectrum(35,e,data,'gl667c_order_35_epoch_{}.png'.format(e))
#hmm, they all look a bit different

# %%
"""
looking at the results for the good epochs but now varying learning rate - learning rate = 20
"""
results = '/home/z5345592/projects/gl667c_wobble/results_rvs_lr20.txt'
results = pd.read_csv(results, skiprows=3, delim_whitespace=True)
results_hdf5 = '/home/z5345592/projects/gl667c_wobble/results_no_bad_orders_lr20.hdf5'
results_hdf5 = wobble.Results(filename=results_hdf5)

# %%
"""
Assigning the variables.
"""
dates = results['dates']
rvs = results['RV'] - np.median(results['RV'])
rvs_err = results['RV_err']
plrvs = results['pipeline_rv'] - np.median(results['pipeline_rv'])
plrvs_err = results['pipeline_rv_err']
# %%
"""
plotting the normalised RVs from wobble against the pipeline RVs
"""
plt.figure()
plt.errorbar(dates, rvs, rvs_err, fmt='o',ms=5, elinewidth=1, c='k', label='wobble')
plt.errorbar(dates, plrvs, plrvs_err, fmt='o', ms=5, elinewidth=1, c='r', label='pipeline') #factor is to plot them on top of each other
plt.xlabel('MJD')
plt.ylabel('normalised RV')
plt.legend()
plt.ylim(-25,25)
#plt.xlim(2454250,2454700)

#%%
"""
plotting point by point residuals
"""
plt.figure()
plt.errorbar(dates, (rvs - plrvs), (rvs_err + plrvs_err), fmt='o', ms=5, elinewidth=1, c='k', label='residuals')
plt.ylim(-25,25)
plt.plot(dates, np.full(len(dates),np.mean(rvs-plrvs)))
print(('RV scatter = ' + str(np.std(rvs - plrvs)) + 'm/s'))

# %%
"""
Plotting the RV scatter by wavelength
"""
colours = cm.rainbow(np.linspace(0,1,len(data.orders)))
plt.figure
for o in range(0,len(data.orders)):
    rv_scatter = np.std(results['RV_order' + str(o)])
    wavelengths = np.exp(data.xs[o][0])
    wavelengths_med = (np.median(wavelengths)) / 10
    plt.scatter(wavelengths_med, rv_scatter, color=colours[o])
    plt.xlabel('median wavelength of echelle order (nm)')
    plt.ylabel('std dev of RV for echelle order')
# %%
"""
Plotting spectra for particular orders
"""
for o in range(0,5):
    results_hdf5.plot_spectrum(o,0,data,'gl667c_order{}.png'.format(o))
# %%
"""
looking at the results for the good epochs but now varying learning rate - learning rate = 100
"""
results = '/home/z5345592/projects/gl667c_wobble/results_rvs_lr100.txt'
results = pd.read_csv(results, skiprows=3, delim_whitespace=True)
results_hdf5 = '/home/z5345592/projects/gl667c_wobble/results_no_bad_orders_lr100.hdf5'
results_hdf5 = wobble.Results(filename=results_hdf5)

# %%
"""
Assigning the variables.
"""
dates = results['dates']
rvs = results['RV'] - np.median(results['RV'])
rvs_err = results['RV_err']
plrvs = results['pipeline_rv'] - np.median(results['pipeline_rv'])
plrvs_err = results['pipeline_rv_err']
# %%
"""
plotting the normalised RVs from wobble against the pipeline RVs
"""
plt.figure()
plt.errorbar(dates, rvs, rvs_err, fmt='o',ms=5, elinewidth=1, c='k', label='wobble')
plt.errorbar(dates, plrvs, plrvs_err, fmt='o', ms=5, elinewidth=1, c='r', label='pipeline') #factor is to plot them on top of each other
plt.xlabel('MJD')
plt.ylabel('normalised RV')
plt.legend()
#plt.ylim(-3950,-3500)
#plt.xlim(2454250,2454700)
# %%
"""
Plotting the RV scatter by wavelength
"""
colours = cm.rainbow(np.linspace(0,1,len(data.orders)))
plt.figure
for o in range(0,len(data.orders)):
    rv_scatter = np.std(results['RV_order' + str(o)])
    wavelengths = np.exp(data.xs[o][0])
    wavelengths_med = (np.median(wavelengths)) / 10
    plt.scatter(wavelengths_med, rv_scatter, color=colours[o])
    plt.xlabel('median wavelength of echelle order (nm)')
    plt.ylabel('std dev of RV for echelle order')
# %%
"""
looking at the results for the good epochs but now varying learning rate - learning rate = 1000
"""
results = '/home/z5345592/projects/gl667c_wobble/results_rvs_lr1000.txt'
results = pd.read_csv(results, skiprows=3, delim_whitespace=True)
results_hdf5 = '/home/z5345592/projects/gl667c_wobble/results_no_bad_orders_lr1000.hdf5'
results_hdf5 = wobble.Results(filename=results_hdf5)

# %%
"""
Assigning the variables.
"""
dates = results['dates']
rvs = results['RV'] - np.median(results['RV'])
rvs_err = results['RV_err']
plrvs = results['pipeline_rv'] - np.median(results['pipeline_rv'])
plrvs_err = results['pipeline_rv_err']
# %%
"""
plotting the normalised RVs from wobble against the pipeline RVs
"""
plt.figure()
plt.errorbar(dates, rvs, rvs_err, fmt='o',ms=5, elinewidth=1, c='k', label='wobble')
plt.errorbar(dates, plrvs, plrvs_err, fmt='o', ms=5, elinewidth=1, c='r', label='pipeline') #factor is to plot them on top of each other
plt.xlabel('MJD')
plt.ylabel('normalised RV')
plt.legend()
#plt.ylim(-3950,-3500)
#plt.xlim(2454250,2454700)
# %%
"""
Plotting the RV scatter by wavelength
"""
colours = cm.rainbow(np.linspace(0,1,len(data.orders)))
plt.figure
for o in range(0,len(data.orders)):
    rv_scatter = np.std(results['RV_order' + str(o)])
    wavelengths = np.exp(data.xs[o][0])
    wavelengths_med = (np.median(wavelengths)) / 10
    plt.scatter(wavelengths_med, rv_scatter, color=colours[o])
    plt.xlabel('median wavelength of echelle order (nm)')
    plt.ylabel('std dev of RV for echelle order')
# %%
