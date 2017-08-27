

import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
from scipy.fftpack import fft, ifft
from scipy import signal

#===================== Setup functions =======================#

# ==== ACF =======

def plt_acf(ts_data, title = "Autocorrelation", lims = [0, 40], remove_mean = True):

    # Check if rainfall data is autoregressive (Annual data)
    # Check autocorrelation of the series prec_annual with differen lag values
    if(remove_mean):
        ts_treat = ts_data - ts_data.mean()
    else:
        ts_treat = ts_data
    ts_acf = np.correlate(ts_treat, ts_treat, "full")
    ts_acf = ts_acf/ts_acf.max()
    
    # Show autocorrelation in monthly rainfall
    # Plot only positive half
    acf_len = len(ts_acf)/2
    x_ax = np.arange(1, acf_len+1, 1)
    y_ax = ts_acf[(acf_len + 1):len(ts_acf)]
    plt.plot(x_ax, y_ax, 'bo-', linewidth = 1.5, markersize = 5)
    plt.xlim(lims)
    plt.title(title)
    plt.show()
    



# importing some time series data
f = open("/home/ishwar/workspace/datasets/clim-data/monthly_precip.txt")

txt = f.readlines()
# Time series from 190101 to 201412
precip_ts = np.array([float(mon.replace("\n","")) for mon in txt])

# prec_y = np.append(precip_ts[1:len(precip_ts)],0)

# plt_acf(precip_ts, title = "Autocorrelation function for monthly rainfall")

# Data has 12 month seasonal cyclicity. 
# Option 1: removing the monthly climatology and recalculating the ACF on anomalies
prec_mon = precip_ts.reshape([len(precip_ts)/12, 12])
x_ax = np.arange(1901, 2015, 1)

prec_jjas2d = prec_mon[:,5:9]
prec_jjas = prec_jjas2d.reshape(prec_jjas2d.shape[0] * 4)
print prec_jjas.shape

# mon_clm = prec_mon.mean(axis = 0)
# clm_conf = np.tile(mon_clm, len(precip_ts)/12)
# prec_anom = precip_ts - clm_conf

# run acf and plot it first without removing mean and then removing mean
# plt_acf(prec_anom, title = "Autocorrelation Function for monthly anomaly", remove_mean = False)

# plt.plot(precip_ts)
# plt.show()
 
# Calculate annual average
# annual_avg = prec_jjas2d.sum(axis = 1)

# JJAS rainfall
plt_acf(prec_jjas, lims = [1, 80], title = "JJAS rainfall ACF", remove_mean = True)

# Plot low pass and try extracting ACF from low frequency values
# b, a = signal.butter(4, 0.4, 'low', output = 'ba')
# annual_filt = signal.filtfilt(b, a, annual_avg)
# plt.plot(x_ax, annual_avg, 'bo-')
# plt.plot(x_ax, annual_filt, 'rs-')
# plt.show()

# Plot low pass and try extracting ACF from low frequency values
b, a = signal.butter(4, 0.2, 'low', output = 'ba')
annual_filt = signal.filtfilt(b, a, prec_jjas)
plt.plot(prec_jjas, 'bo-')
plt.plot(annual_filt, 'rs-')
plt.show()

# ACF of filtered precipitation data
plt_acf(annual_filt, lims = [1,80], title = "Filtered Response ACF", remove_mean = True)

# Use MA smoothing to remove seasonal variations
