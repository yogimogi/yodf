import yodf as tf


def test_sin_x_2_plus_x_3():
    # Basic Gradient Descent testcase
    w = tf.Variable(-1.0)
    # Succinct way of writing
    # cost = tf.sin(tf.add(tf.pow(x, 2), tf.pow(x, 3)))
    cost = tf.sin(w ** 2 + w ** 3)

    learning_rate = 0.05
    iterations = 100
    train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(
        cost
    )

    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        for _ in range(iterations):
            s.run(train)
    # Numbers used for assertion obtained by running same code using Tensorflow 1.15
    assert f"{w.value:>.4f}" == "-1.6077"
    assert f"{cost.value:>.1f}" == "-1.0"
