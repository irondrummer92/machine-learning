# =====================================================================================
# =====================================================================================
# Filename: char-rnn.py
# Purpose: To create a char-rnn model with tensorflow
# =====================================================================================
# =====================================================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk

# Defining functions for weight and bias defintion
def weight_variable(shape):
    initial = tf.random_uniform(shape) * 0.01
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.,shape=shape)
    return tf.Variable(initial)


# readlines from the input text file
fo  = open("../sherlock-ch1.txt", "r")
lns = fo.readlines()

# Convert lns to a a list of individual characters
ln_string = ''.join(lns)
rep = ('1','5','8','-',":","(",")")
for char in rep:
    ln_string = ln_string.replace(char, "")

ln_string.replace("'",'"')
ln_string.lower()

# Get unique indices
uniq_chars = ''.join(set(ln_string))
print uniq_chars

ix = {key: value for value, key in enumerate(uniq_chars)}

# Convert to one hot encoding
char_idx = np.array([ix[l] for l in ln_string], dtype = np.uint8)
enc_text = np.eye(len(ix))[char_idx]

# Input Parameters
N_EPOCHS = 100000
SEQ_LEN = 3 
N_INPUT = len(uniq_chars)
N_OUTPUT = N_INPUT
N_HIDDEN = 10
LEARN_RATE = 1e6
ECHO_OUT = 10000
SEED = "r"


print(ln_string[0:(N_EPOCHS * SEQ_LEN)])

# Declaring Hidden Inputs

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
loss_series = [tf.nn.softmax_cross_entropy_with_logits(labels = ydash_t, logits = y_t) for ydash_t, y_t in zip(y_series, out_series)]
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

        if((i + 10) > len(enc_text) - 1):
            i = 0

        # Sequence of characters to pass
        seq = enc_text[i:(i+SEQ_LEN),:]
        
        # if input is char x, the output must be char x+1
        nxt = enc_text[(i+1):(i + SEQ_LEN + 1),:]

        _total_loss, _train_step, _current_state, _predictions = sess.run([total_loss, train_step, hprev, out_series], 
                feed_dict = {x:seq, y:nxt, h_init:_current_state})

        loss_list.append(_total_loss)

        if(j % ECHO_OUT == 0):
            print("Epoch: ", j, "Total Loss: ", _total_loss) 
            in_seq = ''
            for out_n in seq: 
                char_out_n = ix.keys()[np.argmax(out_n)]
                in_seq = in_seq + char_out_n
            print("Input Sequence: ", in_seq)

            out_seq = ''
            for out_n in _predictions:
                char_out_n = ix.keys()[np.argmax(out_n)]
                out_seq = out_seq + char_out_n
            print("Output: ", out_seq)

        i += SEQ_LEN
