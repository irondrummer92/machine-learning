

import numpy as np
import matplotlib.pyplot as plt
import pickle as pk

# importing some time series data
f = open("/home/ishwar/workspace/Data/clim-data/monthly_precip.txt")

txt = f.readlines()
# Time series from 190101 to 201412
precip_ts = np.array([float(mon.replace("\n","")) for mon in txt])

# remove seasonal fluctuations from the plot
# Get monthly mean of the data and remove to get anomaly
prec_mon = precip_ts.reshape(len(precip_ts)/12,12)
mon_clim = prec_mon.mean(axis = 0)

print(mon_clim)
prec_anom = precip_ts - np.repeat(mon_clim, len(precip_ts)/12)

x_len = len(precip_ts)
train_ind = np.int(np.floor(0.8 * x_len))

prec_train = precip_ts[0:train_ind].reshape([train_ind, 1])
y_train = precip_ts[1:(train_ind+1)].reshape([train_ind, 1])

prec_test = precip_ts[(train_ind):(x_len-1)].reshape([x_len-train_ind-1, 1])
y_test = precip_ts[(train_ind+1):x_len].reshape([x_len-train_ind-1,1])


