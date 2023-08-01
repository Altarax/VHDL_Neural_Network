# The following code is based on Machine Learnia videos : https://github.com/MachineLearnia/Deep-Learning-Youtube

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

DATASET_PATH = 'py_src\A_Z Handwritten Data.csv'

# Initialisation
def initialisation(dimensions):
    parametres = {}
    C = len(dimensions)

    for c in range(1, C):
        parametres['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
        parametres['b' + str(c)] = np.random.randn(dimensions[c], 1)

    return parametres


# Forward propagation
def do_forward_propagation(X, parameters):
    activations = {'A0': X}

    C = len(parameters) // 2
    for c in range(1, C + 1):
        Z = (
            parameters['W' + str(c)].dot(activations['A' + str(c - 1)])
            + parameters['b' + str(c)]
        )
        activations['A' + str(c)] = 1 / (1 + np.exp(-Z))

    return activations


# Back propagation
def do_back_propagation(y, parameters, activations):
    m = y.shape[1]
    C = len(parameters) // 2

    dZ = activations['A' + str(C)] - y

    gradients = {}
    for c in reversed(range(1, C + 1)):
        gradients['dW' + str(c)] = 1 / m * np.dot(dZ, activations['A' + str(c - 1)].T)
        gradients['db' + str(c)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        if c > 1:
            dZ = (
                np.dot(parameters['W' + str(c)].T, dZ)
                * activations['A' + str(c - 1)]
                * (1 - activations['A' + str(c - 1)])
            )

    return gradients


# Update
def update(gradients, parameters, learning_rate):
    C = len(parameters) // 2

    for c in range(1, C + 1):
        parameters['W' + str(c)] = (
            parameters['W' + str(c)] - learning_rate * gradients['dW' + str(c)]
        )
        parameters['b' + str(c)] = (
            parameters['b' + str(c)] - learning_rate * gradients['db' + str(c)]
        )

    return parameters


# Predict
def predict_y(X, parameters) -> bool:
    activations = do_forward_propagation(X, parameters)
    C = len(parameters) // 2
    Af = activations['A' + str(C)]

    return Af >= 0.5


# Deep neural network
def create_neural_network(X, y, hidden_layers: int, learning_rate: int, n_iter: int):
    dimensions = list(HIDDEN_LAYERS)
    dimensions.insert(0, X.shape[0])
    dimensions.append(y.shape[0])
    parameters = initialisation(dimensions)
    C = len(parameters) // 2
    
    # Gradient descent
    training_history = np.zeros((int(n_iter), 2))
    for i in tqdm(range(n_iter)):
        activations = do_forward_propagation(X, parameters)
        gradients = do_back_propagation(y, parameters, activations)
        parameters = update(gradients, parameters, learning_rate)
        Af = activations['A' + str(C)]

        training_history[i, 0] = log_loss(y.flatten(), Af.flatten())
        y_pred = predict_y(X, parameters)
        training_history[i, 1] = accuracy_score(y.flatten(), y_pred.flatten())

    # Plot courbe d'apprentissage
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_history[:, 0], label='train loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(training_history[:, 1], label='train acc')
    plt.legend()
    plt.show()
    """

    return training_history, activations, parameters


# Prepare CSV dataset
def prepare_csv_dataset(dataset_path: str, subset_size) -> tuple:
    data = pd.read_csv('py_src\A_Z Handwritten Data.csv').astype('float32')

    X = data.drop('0', axis=1) 
    X = X.T
    y = data['0']
    
    # Select a subset of the dataset
    if subset_size:
        X_subset = X[:, :subset_size]
        y_subset = y[:subset_size]
        return X_subset, y_subset
    
    return X, y


def convert_floating_points_to_fixed_points(dimensions, parameters, activations):
    factor = 5
    upscale = 8
    inputFactor = 1 / ((2 ** upscale) - 1)
    
    neurons = []

    C = len(dimensions)
    for c in range(C):
        for i in range(dimensions[c]):
            neurons.append({"Neurons" + str(c) + str(i) : {'W' + str(c) : None, 'b' + str(c) : None}})

    for c in range(C):
        for i in range(dimensions[c]):
            neurons["Neurons" + str(c) + str(i)]['W'+str(c)] = 2**factor
    #network1 = (2 ** factor) * nnParams[1]
    #network2 = (2 ** factor) * nnParams[2]

    network1[:, 3] *= (2 ** upscale)
    network2[:, 3] *= (2 ** upscale)

    network1 = network1.astype(np.int32)
    network2 = network2.astype(np.int32)


if __name__ == '__main__':
    HIDDEN_LAYERS = (1, 3, 3)
    LEARNING_RATE = 0.1
    SUBSET_SIZE = 600
    N_ITER = 50

    # Let's try without a real case because I need to learn some VHDL and at this point it's not really working
    # X, y = prepare_csv_dataset(DATASET_PATH, SUBSET_SIZE)
    X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
    X = X.T
    y = y.reshape((1, y.shape[0]))
    
    training_history, activations, parameters = create_neural_network(X, y, HIDDEN_LAYERS, LEARNING_RATE, N_ITER)
    print(parameters)
    convert_floating_points_to_fixed_points(HIDDEN_LAYERS, activations, parameters)


