"""
ASAC (Active Sensing using Actor-Critic Model) (12/18/2018)
Active Sensing Function
"""

import numpy as np

# %% Necessary Packages
import tensorflow as tf
from tqdm import auto

# %% ASAC Function
"""
Inputs:
  - trainX, train Y (training set)
  - testX: testing features
  - cost: measurement costs

Outputs:
  - Selected training samples
  - Selected testing samples
"""


def ASAC(trainX, trainY, validX, testX, cost, iterations=200, learning_rate=0.01, lambda_=0.0001):
    ##### Initialization on the Graph
    tf.reset_default_graph()

    # Network Parameters
    seq_length = len(trainX[0][:, 0])
    data_dim = len(trainX[0][0, :])
    hidden_dim1 = 5
    hidden_dim2 = 5
    output_dim1 = data_dim
    output_dim2 = len(trainY[0][0, :])
    # learning_rate = 0.01

    # %% Preprocessing
    New_trainX = list()
    for i in range(len(trainX)):
        Temp = trainX[i].copy()
        Temp[1:, :] = Temp[: (seq_length - 1), :]
        Temp[0, :] = np.zeros([data_dim])

        New_trainX.append(Temp)

    New_validX = list()
    for i in range(len(validX)):
        Temp = validX[i].copy()
        Temp[1:, :] = Temp[: (seq_length - 1), :]
        Temp[0, :] = np.zeros([data_dim])

        New_validX.append(Temp)

    New_testX = list()
    for i in range(len(testX)):
        Temp = testX[i].copy()
        Temp[1:, :] = Temp[: (seq_length - 1), :]
        Temp[0, :] = np.zeros([data_dim])

        New_testX.append(Temp)

    # %% Network Building
    # input place holders

    New_X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
    X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
    Y = tf.placeholder(tf.float32, [None, seq_length, output_dim2])

    # build a LSTM network
    cell1 = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim1, state_is_tuple=True, activation=None, name="cell1")
    outputs1, _states1 = tf.nn.dynamic_rnn(cell1, New_X, dtype=tf.float32)
    Mask = tf.contrib.layers.fully_connected(
        outputs1, output_dim1, activation_fn=tf.sigmoid
    )  # We use the last cell's output
    New_Mask = tf.maximum(Mask - 0.499, 0)
    New_Mask = New_Mask * 10000
    New_Mask = tf.minimum(New_Mask, 1)

    cell2 = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim2, state_is_tuple=True, activation=tf.tanh, name="cell2")
    outputs2, _states2 = tf.nn.dynamic_rnn(cell2, X * New_Mask, dtype=tf.float32)
    Y_logits_pred = tf.contrib.layers.fully_connected(
        outputs2, output_dim2, activation_fn=None
    )  # We use the last cell's output
    Y_pred = tf.nn.softmax(Y_logits_pred)

    # cost/loss
    # loss1 = tf.reduce_sum(tf.square(tf.reshape(Y_pred, [-1,seq_length]) - Y))   # sum of the squares
    loss1 = tf.losses.softmax_cross_entropy(Y, tf.reshape(Y_logits_pred, [-1, seq_length, output_dim2]))
    loss2 = tf.reduce_sum(New_Mask * cost)
    loss = loss1 + lambda_ * loss2
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    # %% Sessions
    sess = tf.Session()

    # Initialization
    sess.run(tf.global_variables_initializer())

    # %% Training step
    for i in auto.trange(iterations):
        _, step_loss1, step_loss2 = sess.run([train, loss1, loss2], feed_dict={X: trainX, Y: trainY, New_X: New_trainX})

        if i % 100 == 0:
            print("step: " + str(i) + ", Loss1: " + str(step_loss1) + ", Loss2: " + str(step_loss2))

    # %% Test step
    train_mask = sess.run(Mask, feed_dict={X: trainX, New_X: New_trainX})
    valid_mask = sess.run(Mask, feed_dict={X: validX, New_X: New_validX})
    test_mask = sess.run(Mask, feed_dict={X: testX, New_X: New_testX})

    # %% Output
    # Selected Training / Testing Samples

    Final_train_mask = list()
    Final_valid_mask = list()
    Final_test_mask = list()

    for i in range(len(trainX)):
        Final_train_mask.append(np.round(train_mask[i, :, :]))

    for i in range(len(validX)):
        Final_valid_mask.append(np.round(valid_mask[i, :, :]))

    for i in range(len(testX)):
        Final_test_mask.append(np.round(test_mask[i, :, :]))

    return Final_train_mask, Final_valid_mask, Final_test_mask
