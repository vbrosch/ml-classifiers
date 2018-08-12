"""
    Acknowledgements: inspired by (and sometimes completely based on ;)):
    https://github.com/joelgrus/data-science-from-scratch/
"""

import math
import random


def dot(x, y):
    return sum([v * w for v, w in zip(x, y)])


def relu(x):
    return max(x, 0)


def dx_relu(x):
    return 1 if x > 0 else 0


def softmax(x):
    return math.log(1 + math.exp(x))


def dx_softmax(x):
    return math.exp(x) / (1 + math.exp(x))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def dx_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


a = sigmoid
dx_a = dx_sigmoid


def neuron_output(weights, input_v, use_identity: bool):
    x = dot(weights, input_v)
    return x if use_identity else a(x)


def feed_forward(network, input_vector):
    layer_outputs = []

    for l_i, layer in enumerate(network):
        biased_input_vector = input_vector + [1]

        # for the output neurons a_i = y_i (identity function)
        output = [neuron_output(neuron, biased_input_vector, l_i == len(network)) for neuron in layer]

        layer_outputs.append(output)

        input_vector = output

    return layer_outputs


def backpropagate(network, input_vector, target_vector):
    hidden_outputs, outputs = feed_forward(network, input_vector)

    # deltas
    output_deltas = [output - target_vector[i] for i, output in enumerate(outputs)]

    # adjust output weights
    for i, output_weight in enumerate(network[-1]):
        for j, hidden_output in enumerate((hidden_outputs + [1])):
            output_weight[j] -= learning_rate * output_deltas[i] * hidden_output

    # calculate hidden deltas
    hidden_deltas = [dx_a(hidden_output) * dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]

    # adjust hidden weights
    for i, hidden_weight in enumerate(network[0]):
        for j, iv in enumerate((input_vector + [1])):
            hidden_weight[j] -= learning_rate * hidden_deltas[i] * iv


if __name__ == "__main__":
    raw_digits = [
        """11111
               1...1
               1...1
               1...1
               11111""",

        """..1..
               ..1..
               ..1..
               ..1..
               ..1..""",

        """11111
               ....1
               11111
               1....
               11111""",

        """11111
               ....1
               11111
               ....1
               11111""",

        """1...1
               1...1
               11111
               ....1
               ....1""",

        """11111
               1....
               11111
               ....1
               11111""",

        """11111
               1....
               11111
               1...1
               11111""",

        """11111
               ....1
               ....1
               ....1
               ....1""",

        """11111
               1...1
               11111
               1...1
               11111""",

        """11111
               1...1
               11111
               ....1
               11111"""]


    def make_digit(raw_digit):
        return [1 if c == '1' else 0
                for row in raw_digit.split("\n")
                for c in row.strip()]


    inputs = list(map(make_digit, raw_digits))

    targets = [[1 if i == j else 0 for i in range(10)]
               for j in range(10)]

    random.seed(0)  # to get repeatable results
    input_size = 25  # each input is a vector of length 25
    num_hidden = 5  # we'll have 5 neurons in the hidden layer
    output_size = 10  # we need 10 outputs for each input
    learning_rate = 1

    # each hidden neuron has one weight per input, plus a bias weight
    hidden_layer = [[random.random() for _ in range(input_size + 1)]
                    for _ in range(num_hidden)]

    # each output neuron has one weight per hidden neuron, plus a bias weight
    output_layer = [[random.random() for _ in range(num_hidden + 1)]
                    for _ in range(output_size)]

    # the network starts out with random weights
    network = [hidden_layer, output_layer]

    # 10,000 iterations seems enough to converge
    for __ in range(10000):
        for input_vector, target_vector in zip(inputs, targets):
            backpropagate(network, input_vector, target_vector)


    def predict(input):
        return feed_forward(network, input)[-1]


    for i, input in enumerate(inputs):
        outputs = predict(input)
        print(i, [round(p, 2) for p in outputs])

    print(""".@@@.
    ...@@
    ..@@.
    ...@@
    .@@@.""")
    print([round(x, 2) for x in
           predict([0, 1, 1, 1, 0,  # .@@@.
                    0, 0, 0, 1, 1,  # ...@@
                    0, 0, 1, 1, 0,  # ..@@.
                    0, 0, 0, 1, 1,  # ...@@
                    0, 1, 1, 1, 0])])  # .@@@.
    print()

    print(""".@@@.
    @..@@
    .@@@.
    @..@@
    .@@@.""")
    print([round(x, 2) for x in
           predict([0, 1, 1, 1, 0,  # .@@@.
                    1, 0, 0, 1, 1,  # @..@@
                    0, 1, 1, 1, 0,  # .@@@.
                    1, 0, 0, 1, 1,  # @..@@
                    0, 1, 1, 1, 0])])  # .@@@.
    print()
