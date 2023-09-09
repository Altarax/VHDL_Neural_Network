# The following code is based on Machine Learnia videos : https://github.com/MachineLearnia/Deep-Learning-Youtube

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from tools import *

DATASET_PATH = 'py_src\A_Z Handwritten Data.csv'
Z_AND_ACT_VALUES = []


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

    # Ploting learning curve
    """ Commented to test without losing time displaying something
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


# Create a test dataset
def create_circle_dataset(n_samples=100, noise=0.1, factor=0.3, random_state=0):
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    X = X.T
    y = y.reshape((1, y.shape[0]))
    return X, y


# Convertir floating point to fixed point
def convert_to_fixed_point(parameters, activations, factor=5, upscale=8):
    converted_parameters = {}
    calculated_factor = (2 ** factor)
    
    for key, value in parameters.items():
        if isinstance(value, np.ndarray):
            if key.startswith('W') or key.startswith('b'):
                scaled_value = calculated_factor * value
                converted_parameters[key] = np.int32(scaled_value)
            else:
                converted_parameters[key] = value
        elif isinstance(value, (int, float)):
            converted_parameters[key] = int(value)
        else:
            converted_parameters[key] = value
    
    converted_activations = {}
    for key, value in activations.items():
        if isinstance(value, np.ndarray) and key.startswith('A'):
            scaled_value = calculated_factor * value
            converted_activations[key] = np.int32(scaled_value)
        else:
            converted_activations[key] = value
    
    return converted_parameters, converted_activations


if __name__ == '__main__':
    HIDDEN_LAYERS = (2, 2)
    LEARNING_RATE = 0.1
    SUBSET_SIZE = 600
    N_ITER = 50

    X, y                                        = create_circle_dataset()
    training_history, activations, parameters   = create_neural_network(X, y, HIDDEN_LAYERS, LEARNING_RATE, N_ITER)
    converted_parameters, converted_activations = convert_to_fixed_point(parameters, activations)

    act_min, act_max = find_max_min_list_of_arrays(converted_activations)
    print("Minimum:", act_min)
    print("Maximum:", act_max)