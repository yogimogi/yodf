import yodf as tf
import numpy as np


def make_planar_dataset():
    np.random.seed(1)
    m = 400  # number of examples
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    y = np.zeros((m, 1), dtype="uint8")  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    X = X
    y = y

    return X, y


def initialize_weights(shape):
    np.random.seed(16180)
    return np.random.randn(shape[0], shape[1]) * 0.1


def sigmoid(x):
    return 1 / (1 + np.e ** -x)


def predict(X, W0, b0, W1, b1):
    a1 = sigmoid(X.dot(W0) + b0)
    h = sigmoid(a1.dot(W1) + b1)
    return (h > 0.5) + 0


def nn_model(
    X, y, input_layer_size, hidden_layer_size, K, iterations=1000, learning_rate=1.2
):
    m = X.shape[0]

    W0 = tf.Variable(initialize_weights((input_layer_size, hidden_layer_size)))
    b0 = tf.Variable(np.zeros((1, hidden_layer_size)))
    W1 = tf.Variable(initialize_weights((hidden_layer_size, K)))
    b1 = tf.Variable(np.zeros((1, K)))

    a1 = tf.sigmoid(tf.matmul(X, W0) + b0)
    h = tf.sigmoid(tf.matmul(a1, W1) + b1)

    cost = tf.log(h) * y + tf.log(1 - h) * (1 - y)
    cost = -1 / m * tf.reduce_sum(cost)

    train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(
        cost
    )

    init = tf.global_variables_initializer()

    costs = []
    with tf.Session() as s:
        s.run(init)
        for _ in range(iterations):
            cost_final, _, W0_final, b0_final, W1_final, b1_final = s.run(
                [cost, train, W0, b0, W1, b1]
            )
            costs.append(cost_final)

    return W0_final, b0_final, W1_final, b1_final, costs


def _equal(a, b, delta=1e-7):
    return abs(a - b) < delta


def test_nn():
    # Neural network with one hidden layer
    X, y = make_planar_dataset()
    W0_final, b0_final, W1_final, b1_final, costs = nn_model(
        X, y, X.shape[1], 2, 1, iterations=100
    )
    # Numbers used for assertion obtained by running same code using Tensorflow 1.15
    assert _equal(np.sum(W0_final), -0.26125149361875577)
    assert _equal(np.sum(b0_final), -0.1242844940700719)
    assert _equal(np.sum(W1_final), 1.2386396720687358)
    assert _equal(np.sum(b1_final), -0.56819478528124)
    assert _equal(costs[len(costs) - 1], 0.6659947865954474)
