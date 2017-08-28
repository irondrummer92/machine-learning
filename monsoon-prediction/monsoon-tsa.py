

import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
from sklearn import linear_model

#===================== Setup functions =======================#

# ===================== ACF ====================== #

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


# ========== Deseasonalize time series ============ #

def deseasonalize_ts(ts_data, w_size):

    t = len(ts_data)
    
    # Empty array for storing deseasonalize data
    trend_ts = np.zeros((t-w_size+1))
    
    k = 0
    
    # For all time series
    while (k + w_size) <= t:
        trend_ts[k] = ts_data[k:(k+w_size)].mean()
        k+=1

    seas_ts = ts_data[(w_size-1):] - trend_ts
    return trend_ts, seas_ts


# ================ Scatter lag ===================== #

def scatter_lag(ts_data, lag = 1, plot_scatter = True):

    t = len(ts_data)

    ts_lag = ts_data[lag:]
    ts = ts_data[:(t-lag)]

    if(plot_scatter):
        plt.title("Scatter plots of ts and " + str(lag) + " lagged ts")    
        plt.scatter(ts_lag, ts)
        plt.show()

    return ts, ts_lag


# importing some time series data
f = open("/home/ishwar/workspace/datasets/clim-data/monthly_precip.txt")

txt = f.readlines()
# Time series from 190101 to 201412
precip_ts = np.array([float(mon.replace("\n","")) for mon in txt])

# Deseasonalize the time series
trnd, seas = deseasonalize_ts(precip_ts, 12)

# plotting the trend and seasonality
plt.plot(trnd, 'b-')
plt.show()

plt_acf(trnd)

# First order AR model can be built since high correlation exists between t and t - 1
# Build a regression model with assumptions (say hold true)
ts0, tsn_1 = scatter_lag(trnd, 1)
regr = linear_model.LinearRegression(fit_intercept = True)
regr.fit(tsn_1.reshape([-1,1]), ts0.reshape([-1,1]))

# Calculate error in dataset
pred = regr.predict(tsn_1.reshape([-1,1]))
mse = np.sqrt(np.mean((ts0 - pred) ** 2))
print ("Coefficient", regr.coef_, "Intercept", regr.intercept_, "MSE: ", mse)

ss, ssn_1 = scatter_lag(seas, 1, False)
mons_pred = pred.flatten() + ss

prec = trnd + seas

pr_ts, prn_1 = scatter_lag(prec, 1, False)

# Calculate error in prediction
mse = np.sqrt(np.mean((pr_ts - mons_pred) ** 2))
mape = np.mean(np.absolute(pr_ts - mons_pred))

print("Reseasonalized predictions: MSE = ", mse, "; MAD = ", mape)

plt.plot(pr_ts, 'b-', label = "Actual")
plt.plot(mons_pred.flatten(), 'r--', label = "Predicted")
plt.legend()
plt.show()
