"""
ASAC (Active Sensing using Actor-Critic Model) (12/18/2018)
Prediction Function only with Selected Samples
"""

# %% Necessary packages
import tensorflow as tf
from tqdm import auto

# %% Prediction Function
"""
Inputs:
  - trainX, train Y (training set)
  - testX: testing features
  - trainG: mask vector for selected training samples
  - trainG: mask vector for selected testing samples

Outputs:
  - Prediction results on testing set
"""


def Predictor_G(trainX, testX, trainY, trainG, testG, iterations=200):
    # Initialization on the Graph
    tf.reset_default_graph()

    # %% Preprocessing
    Train_No = len(trainX)
    Test_No = len(testX)

    New_trainX = list()
    for i in range(Train_No):
        Temp = trainX[i]
        Temp = Temp * trainG[i]

        New_trainX.append(Temp)

    New_testX = list()
    for i in range(Test_No):
        Temp = testX[i]
        Temp = Temp * testG[i]

        New_testX.append(Temp)

    # %% Network Parameters
    seq_length = len(New_trainX[0][:, 0])
    data_dim = len(New_trainX[0][0, :])
    hidden_dim = 5
    output_dim = len(trainY[0][0, :])
    learning_rate = 0.01

    # %% Network Build
    # input place holders
    X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
    Y = tf.placeholder(tf.float32, [None, seq_length, output_dim])

    # build a LSTM network
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
    outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    # Y_pred = tf.contrib.layers.fully_connected(outputs, output_dim, activation_fn=None)  # We use the last cell's output
    Y_logits_pred = tf.contrib.layers.fully_connected(
        outputs, output_dim, activation_fn=None
    )  # We use the last cell's output
    Y_pred = tf.nn.softmax(Y_logits_pred)
    # cost/loss
    # loss = tf.reduce_sum(tf.square(tf.reshape(Y_pred, [-1,seq_length]) - Y))  # sum of the squares
    loss = tf.losses.softmax_cross_entropy(Y, tf.reshape(Y_logits_pred, [-1, seq_length, output_dim]))
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    # %% Sessions
    sess = tf.Session()

    # Initialization
    sess.run(tf.global_variables_initializer())

    # %% Training step
    for i in auto.trange(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={X: New_trainX, Y: trainY})

        if i % 100 == 0:
            print(f"[step: {i}] loss: {step_loss}")

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: New_testX})

    # %% Output

    Final = list()

    for i in range(len(testX)):
        Final.append(test_predict[i, :, :])

    return Final
