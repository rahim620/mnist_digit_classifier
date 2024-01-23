import numpy as np


class Network:

    def __init__(self, sizes):
        """ Initializes a neural network

            'sizes' is a list of integers that contains the number of neurons in
            the respective layers"""
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Biases represented as a vector of the biases between the layers
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # Weights represented as a list of matrices
        # weights[0] would be the weights between 1st layer to the 2nd layer
        # weights[0][0] are the weights connected to the first neuron of the 2nd layer from
        # all the neurons in the previous 1st layer
        # weights[0][0][0] would be the weight between the first neuron of the second layer and
        # the first neuron of the second layer
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network for an input a represented as a vector"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. The neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def SGD(self, training_data, epochs, mini_batch_size, r,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent. The `training_data` is a list of tuples
        (x, y) representing the training inputs and desired
        outputs. If `test_data` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out."""

        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, r)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, r):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The `mini_batch` is a list of tuples (x, y), and 'r'
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (r / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (r / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple `(nabla_b, nabla_w)` representing the
        gradient for the cost function C_x.  `nabla_b` and
        `nabla_w` are layer-by-layer matrices, similar
        to `self.biases` and `self.weights` """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = cost_derivative(activations[-1], y) * sigmoid_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_derivative(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w


def sigmoid(x):
    """Sigmoid function of integer x"""
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    """Derivative of the sigmoid function of integer x"""
    return sigmoid(x) * (1 - sigmoid(x))


def cost_derivative(a, y):
    """Derivative of the cost function where 'a' represents the
    activations of the output layer and 'y' is the expected result"""
    return a - y
