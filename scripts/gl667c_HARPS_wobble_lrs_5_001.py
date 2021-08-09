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
data = wobble.Data(filename='/home/ajolly/projects/gl667c_wobble/data/gl667c.hdf5', epochs=good_epochs) #taiga
#data = wobble.Data(filename='/home/z5345592/projects/wobble_precision/data/gl667c.hdf5', epochs=good_epochs) #laptop for testing
#%%
data.delete_orders(bad_orders) #deletes the bad orders that have been identified in the 'bad orders' list above. Think of a way to do this beforehand?
data.drop_bad_orders(min_snr=3) #dropping bad orders
data.drop_bad_epochs(min_snr=3) #dropping bad epochs

# %%
rv_lrs = [5]
tell_temp_lrs = [0.01]

for rv_lr in rv_lrs:
    for tell_lr in tell_temp_lrs:

        results = wobble.Results(data = data)

        for r in range(len(data.orders)):
            print('starting order {0} of {1}'.format(r+1, len(data.orders)))
            model = wobble.Model(data, results, r)
            model.add_star('star', learning_rate_rvs=rv_lr)
            model.add_telluric('tellurics', variable_bases=2, learning_rate_template = tell_lr)
            wobble.optimize_order(model)

        results.combine_orders('star')
        results.apply_drifts('star') # instrumental drift corrections
        results.apply_bervs('star') # barycentric corrections
        results.write_rvs('star', 'results_rvs_rvlr{}telr{}.txt'.format(rv_lr, tell_lr), all_orders=True) #results txt contains a column for each order
        results.write('results_rvlr{}telr{}.hdf5'.format(rv_lr, tell_lr))