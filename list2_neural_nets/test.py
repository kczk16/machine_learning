import numpy as np


def initialize_random_weights(inputs_amount, hidden_nodes_amount, outputs_amount):
    '''
    param inputs_amount: number of cells in input layer
    param output_amount: number of cells in output layer
    return: pair of tuples of weights and biases for proper layers
    '''
    np.random.seed(100)

    w1 = np.random.normal(0, 1 / np.sqrt(hidden_nodes_amount), (inputs_amount, hidden_nodes_amount))
    b1 = np.random.normal(0, 1, hidden_nodes_amount)

    w2 = np.random.normal(0, 1 / np.sqrt(outputs_amount), (hidden_nodes_amount, outputs_amount))
    b2 = np.random.normal(0, 1, outputs_amount)

    return (w1, w2), (b1, b2)

w, b = initialize_random_weights(inputs_amount = 2, hidden_nodes_amount = 3, outputs_amount=1)