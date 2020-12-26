import yodf as tf


def _equal(a, b):
    return abs(a - b) < 1e-7


def test_w0_2_plus_w1_2():
    # Gradient Descent with 2 variables
    initial_value1 = -2.5
    initial_value2 = 2.5
    w0 = tf.Variable(initial_value1)
    w1 = tf.Variable(initial_value2)
    cost = w0 ** 2 + w1 ** 2

    learning_rate = 0.01
    iterations = 200
    train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(
        cost
    )

    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        for _ in range(iterations):
            # For TensorFlow, order of tensors/operations passed to run makes
            # no difference to the results.
            # For yodf, it does, specifically the cost tensor
            # If cost is passed after train, cost will get computed again
            # after weights have been adjusted
            cost_final, _, w0_final, w1_final = s.run([cost, train, w0, w1])

    # Numbers used for assertion obtained by running same code using Tensorflow 1.15
    assert _equal(w0_final, -0.043969862)
    assert _equal(w1_final, 0.043969862)
    assert _equal(cost_final, 0.0040261326)

