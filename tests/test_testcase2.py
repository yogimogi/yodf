import yodf as tf
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.e ** -x)


def sigmoid_gradient(x):
    s = sigmoid(x)
    return s * (1 - s)


def test_sigmoid():
    # Compute Sigmoid gradient in 2 different ways and compare
    initial_value = 0.7
    w = tf.Variable(initial_value)

    sig_1 = tf.sigmoid(w)
    sig_2 = 1 / (1 + tf.exp(-w))

    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        s.run([sig_1, sig_2])

    assert f"{sig_1.gradient:>.5f}" == f"{sig_2.gradient:>.5f}"
    assert f"{sigmoid_gradient(initial_value):>.5f}" == f"{sig_2.gradient:>.5f}"

