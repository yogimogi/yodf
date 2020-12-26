import yodf as tf
import numpy as np


def test_simple_gradients0():
    # Checks if computations of operation 2^10 and its gradient are correct
    initial_value = 2
    raise_to = 10
    w = tf.Variable(initial_value)
    op = w ** raise_to
    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        s.run(op)

    assert op.value == initial_value ** raise_to
    assert op.gradient == raise_to * (initial_value ** (raise_to - 1))


def test_simple_gradients1():
    # Checks if computations of sin, cos and sin*cos and their gradients are correct
    theta = 0.5
    w = tf.Variable(theta)
    op_s = tf.sin(w)
    op_c = tf.cos(w)
    op_p = op_s * op_c
    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        s.run([op_s, op_c, op_p])

    assert f"{op_s.value:0.4f}" == f"{np.sin(theta):0.4f}"
    assert f"{op_s.gradient:0.4f}" == f"{np.cos(theta):0.4f}"

    assert f"{op_c.value:0.4f}" == f"{np.cos(theta):0.4f}"
    assert f"{op_c.gradient:0.4f}" == f"{-np.sin(theta):0.4f}"

    assert f"{op_p.value:0.4f}" == f"{np.sin(theta)*np.cos(theta):0.4f}"
    assert f"{op_p.gradient:0.4f}" == f"{np.cos(theta)**2-np.sin(theta)**2:0.4f}"
