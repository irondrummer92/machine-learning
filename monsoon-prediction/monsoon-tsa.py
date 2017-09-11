

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
    return trend_ts, seas_ts[1:]


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
f = open("/home/mudraje/workspace/data/clim-data/monthly_precip.txt")
txt = f.readlines()
# Time series from 190101 to 201412
precip_ts = np.array([float(mon.replace("\n","")) for mon in txt])


# Years from 1901 to 1990 used for prediction
# Indices 0 to 90 * 12
# Indices 90 * 12 : end for testing
prec_train = precip_ts[:(90*12)]
prec_test = precip_ts[90*12:]

print "Training Data: " + str(prec_train.shape)
print "Test Data: " + str(prec_test.shape)

# Deseasonalize for prediction and get monthly average
train, ss = deseasonalize_ts(prec_train, 12)
seas = ss.reshape([-1,12]).mean(axis=0)


test = prec_test - np.tile(seas, len(prec_test)/12)

train_0, train_1 = scatter_lag(train, 1)
test_0, test_1 = scatter_lag(test, 1)

regr = linear_model.LinearRegression(fit_intercept = True)
regr.fit(train_0.reshape([-1,1]), train_1.reshape([-1,1]))

print("Coefficient", regr.coef_, "Intercept", regr.intercept_)

train_pred = regr.predict(train_0.reshape([-1,1])).flatten()
train_pred += np.tile(seas, len(train_pred)/12)

plt.plot(train_pred, 'b-')
plt.plot(prec_train[12:], 'r--')
plt.show()

test_pred = regr.predict(test.reshape([-1,1])).flatten()
ss_test = np.tile(seas, len(test_pred)/12)
ss_test = np.append(ss_test[1:], ss_test[0])
test_pred += ss_test

mse_train = np.sqrt(np.mean((train_pred - prec_train[12:])**2))
mse_test = np.sqrt(np.mean((test_pred[:-1] - prec_test[1:])**2))

print("Train Error: ", mse_train, "Test Error", mse_test)
