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
data = wobble.Data(filename='/home/ajolly/projects/gl667c_wobble/data/gl667c.hdf5')

bad_orders = np.array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False,  True, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False,  True, False, False, False, False, False,
       False,  True,  True, False, False,  True,  True,  True,  True])

data.drop_bad_orders(min_snr=3) #dropping bad orders
data.drop_bad_epochs(min_snr=3) #dropping bad epochs
data.delete_orders(bad_orders) #deletes the bad orders that have been identified in the 'bad orders' list above. Think of a way to do this beforehand?

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
results.write('results_no_bad_orders.hdf5')