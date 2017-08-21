

import numpy as np
import matplotlib.pyplot as plt
import pickle as pk

# importing some time series data
f = open("/home/ishwar/workspace/Data/clim-data/monthly_precip.txt")

txt = f.readlines()
# Time series from 190101 to 201412
precip_ts = np.array([float(mon.replace("\n","")) for mon in txt])

tm_arr = np.empty(len(precip_ts), np.int32)
tm = np.arange(1901, 2015)
print tm
j = 0
for i in range(len(tm)):
    tm_arr[j:j+12] = tm[i] * 100 + np.arange(1, 13)
    j = j + 12

# remove seasonal fluctuations from the plot
# Get monthly mean of the data and remove to get anomaly
prec_mon = precip_ts.reshape(len(precip_ts)/12,12)
mon_clim = prec_mon.mean(axis = 0)

print(mon_clim)

# Get only JJAS
tm_jjas = precip_ts[[t % 100 in [6,7,8,9] for t in tm_arr]]
print len(tm_jjas)

# convert to 4D dataset
prec_jjas = tm_jjas.reshape(len(tm), 4)

# Plot the 4 time series (JJAS)
lbs = ("June", "July", "August", "September")
lns = ("ro-", "bs-", "gh-", "mD-")
#for i, ts in enumerate(prec_jjas.T):
#    plt.plot(tm, ts, lns[i], label = lbs[i], linewidth = 1.5, markersize = 4.0)
#    plt.xticks(np.arange(1901, 2015, 10))
#    plt.xlim([1901, 2014])
#    plt.legend()
#    plt.show()

# Calculate JJAS average rainfall for all 114 years
prec_annual = np.mean(prec_jjas, axis = 1)

prec_y = np.append(prec_annual[1:len(prec_annual)],0)
prec_data = np.append(prec_jjas, prec_annual.reshape(len(tm), 1), axis = 1)
prec_data = np.append(prec_data, prec_y.reshape(len(tm), 1), axis = 1)

# Calculate if current values can be correlated with t+1 values
cor = np.corrcoef(prec_data.T)
print cor

# Check if rainfall data is autoregressive (Annual data)
# Check autocorrelation of the series prec_annual with differen lag values
prec_treat = prec_annual - prec_annual.mean()
prec_acf = np.correlate(prec_treat, prec_treat, "full")
prec_acf = prec_acf/prec_acf.max()
x_ax = np.arange(-113, 114, 1.)

plt.plot(x_ax, prec_acf, 'bo-', linewidth = 1.5, markersize = 10)
plt.xlim([-20, 20])
plt.title("Autocorrelation function for Annual Precipitation")
plt.show()
