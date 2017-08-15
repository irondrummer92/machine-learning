import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk


# User inputs
N_EPOCHS = 50000
SEQ_LEN = 7
N_INPUT = 1
N_OUTPUT = N_INPUT
N_HIDDEN = 5
LEARN_RATE = 5e-6
ECHO_OUT = 1000


# importing some time series data
f = open("/home/ishwar/workspace/datasets/clim-data/monthly_precip.txt")

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


# Defining functions for weight and bias defintion
def weight_variable(shape):
    initial = tf.random_uniform(shape) * tf.sqrt(2.0/(shape[0] * shape[1])) 
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.,shape=shape)
    return tf.Variable(initial)

def square_loss(y_t, y_t_):

    sq_err = (y_t - y_t_) ** 2
    err  = tf.sqrt(tf.reduce_mean(sq_err))
    return err

# Input and hidden vectors
x = tf.placeholder(tf.float32, [SEQ_LEN, N_INPUT], name = "input")   # Inputs to the network
y = tf.placeholder(tf.float32, [SEQ_LEN, N_OUTPUT], name = "labels")  # target variable
h_init = tf.placeholder(tf.float32, [1, N_HIDDEN], name = "prev") # Previous hidden state

# Declare RNN weights
Wxh = weight_variable([N_INPUT, N_HIDDEN])
Whh = weight_variable([N_HIDDEN, N_HIDDEN])
Why = weight_variable([N_HIDDEN, N_OUTPUT])

# Declare biases
bh = bias_variable([1, N_HIDDEN])
by = bias_variable([1, N_OUTPUT])

x_series = tf.unstack(x, axis = 0)
y_series = tf.unstack(y, axis = 0)

out_series = []

hprev = h_init

# FORWARD PASS through the network
for x_t in x_series:

    # Hidden outputs
    x_t = tf.reshape(x_t, [1, N_INPUT])
    ht = tf.nn.relu(tf.matmul(x_t, Wxh) + tf.matmul(hprev, Whh) + bh)

    # Output of the net
    # y_t = tf.nn.tanh(tf.matmul(ht, Why) + by)
    y_t = tf.matmul(ht, Why) + by

    hprev = ht

    out_series.append(y_t)


# Loss calculations
loss_series = [square_loss(ydash_t, y_t) for ydash_t, y_t in zip(y_series, out_series)]
total_loss = tf.reduce_mean(loss_series)

# Define optmizer
optim = tf.train.AdamOptimizer(learning_rate = LEARN_RATE)
train_step = optim.minimize(total_loss)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    i = 0

    _current_state = np.zeros([1, N_HIDDEN])

    loss_list = []

    for j in range(N_EPOCHS):

        if((i + SEQ_LEN) > len(prec_train) - 1):
            i = 0

        # Sequence of characters to pass
        seq = prec_train[i:(i+SEQ_LEN),:]

        # if input is char x, the output must be char x+1
        nxt = y_train[i:i + SEQ_LEN,:]

        _total_loss, _train_step, _current_state, _predictions = sess.run([total_loss, train_step, hprev, out_series], 
                feed_dict = {x:seq, y:nxt, h_init:_current_state})

        loss_list.append(_total_loss)

        if(j % ECHO_OUT == 0):
            # print Whh.eval()
                        # Evaluate test loss
            k = 0 # Feed in sequences

            # to get final state of the network use
            # Run for all sequence bits in 
            while(k < (len(prec_train) - SEQ_LEN)):
                x_tr = prec_train[k:(k+SEQ_LEN)]
                y_tr_out = y_train[k:(k+SEQ_LEN)]
                _train_state = np.zeros([1, N_HIDDEN])
                _train_loss, _train_state = sess.run([total_loss,hprev], feed_dict = {x:x_tr, y:y_tr_out, h_init:_train_state})
                k+=1

            k = 0
            test_series = []
            _test_state = np.copy(_train_state)
            # Run for all sequence bits in 
            while(k < (len(prec_test) - SEQ_LEN)):
                x_test = prec_test[k:(k+SEQ_LEN)]
                y_out = y_test[k:(k+SEQ_LEN)]
                _test_loss, _test_state, _preds = sess.run([total_loss,hprev, out_series], feed_dict = {x:x_test, y:y_out, h_init:_test_state})
                test_series.append(_test_loss)
                k+=1

                # print "Sequence: " + str(x_test.flatten())
                # print "Actual: " + str(y_out.flatten())
                # print "Predicted: " + str(np.concatenate(_preds).flatten())

            test_mean = np.mean(test_series)

            print("Epoch: ", j, "Train Loss: ", _total_loss, "Test Loss: ", test_mean)

        i += SEQ_LEN


