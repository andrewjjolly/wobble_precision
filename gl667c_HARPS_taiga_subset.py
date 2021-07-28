#%%
"""
Importing the things
"""
import tarfile
import glob
import tqdm
import os
import wobble
import numpy as np
from astropy.io import fits
import os
import glob
from matplotlib import pyplot as plt
import time

# # %%
# """
# Defining the location of the directory the data is saved in, and the directory in which the code is saved in.
# """
# data_dir = '/home/z5345592/projects/gl667c_wobble/data'
# out = '/home/z5345592/Code/wobble_tutorials/gl667c' #to do - name this something better than 'out'

# #%%
# for file in glob.glob(data_dir + '/*.tar'):
#         tar = tarfile.open(file,'r')
#         tar.extractall(path = out)

# # %%
# """
# Creating a list of the CCF files - so I can test with 1 file before jumping into the rest - question if I need this if I am
# going to be running the block below anyway.
# """
# filelist = []
# for filename in os.listdir(data_dir):
#     if filename.endswith('_ccf_M2_A.fits'):
#         print(os.path.join(data_dir, filename))
#         filelist.append(os.path.join(data_dir,filename))

# # %%
# """
# Identifying the filelist and grabbing the spectra and running it through the from_HARPS function.
# """
# data = wobble.Data()
# sp = wobble.Spectrum()
# for f in filelist:
#         sp.from_HARPS(f, process=True)
#         data.append(sp)

# #%%
# data.write('/home/z5345592/projects/gl667c_wobble/data/gl667c.hdf5')

#%%
orders_subset = range(30,42) #use these if you just want to use a few orders / epochs for testing then slice in wobble.Data
epochs_subset = range(0,10)

data = wobble.Data(filename='/home/ajolly/projects/gl667c_wobble/data/gl667c.hdf5', orders = orders_subset, epochs = epochs_subset)

data.drop_bad_orders(min_snr=3) #dropping bad orders
data.drop_bad_epochs(min_snr=3) #dropping bad epochs

# %%
results = wobble.Results(data = data)

# %%
for r in range(len(data.orders)):
        print('starting order {0} of {1}'.format(r+1, len(data.orders)))
        model = wobble.Model(data, results, r)
        model.add_star('star')
        model.add_telluric('tellurics', variable_bases=2)
        wobble.optimize_order(model)
print('done!')

# %%
results.combine_orders('star')
results.apply_drifts('star') # instrumental drift corrections
results.apply_bervs('star') # barycentric corrections
results.write_rvs('star', 'results_rvs.txt', all_orders=True) #results txt contains a column for each order

#%%
# n = 0 # epoch to plot
# r = 60 #order to plot
# results.plot_spectrum(r, n, data, 'gl667c_order{}.png'.format(r))
     
# # %%
# dates, rvs, rvs_err, plrvs, plrvs_err = np.genfromtxt('results_rvs.txt', skip_header=4, unpack=True)
# rvs_norm = rvs - np.mean(rvs)
# plrvs_norm = plrvs - np.mean(plrvs)
# plt.figure()
# plt.errorbar(dates, rvs_norm, rvs_err, fmt='o', ms=5, elinewidth=1, label='pipeline')
# plt.errorbar(dates, plrvs_norm, plrvs_err, fmt='o', ms=5, elinewidth=1, label='wobble, LR, tellurics = default')
# plt.xlabel('MJD')
# plt.ylabel(r'RV (m s$^{-1}$)');
# plt.legend()
# #plt.ylim([75,95])
# #plt.xlim([2454180,2454220])

# # %%
