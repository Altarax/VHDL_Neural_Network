# The following code is based on Machine Learnia videos : https://github.com/MachineLearnia/Deep-Learning-Youtube

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

DATASET_PATH = "py_src\A_Z Handwritten Data.csv"


# Initialisation
def init(layers_dimensions: list) -> dict:
    parameters = {}

    C = len(layers_dimensions)
    for c in range(1, C):
        parameters["W" + str(c)] = np.random.randn(
            layers_dimensions[c], layers_dimensions[c - 1]
        )
        parameters["b" + str(c)] = np.random.randn(layers_dimensions[c], 1)

    return parameters


# Forward propagation
def do_forward_propagation(X, parameters):
    activations = {"A0": X}

    C = len(parameters) // 2
    for c in range(1, C + 1):
        Z = (
            parameters["W" + str(c)].dot(activations["A" + str(c - 1)])
            + parameters["b" + str(c)]
        )
        activations["A" + str(c)] = 1 / (1 + np.exp(-Z))

    return activations


# Back propagation
def do_back_propagation(y, parameters, activations):
    m = y.shape[1]
    C = len(parameters) // 2

    dZ = activations["A" + str(C)] - y

    gradients = {}
    for c in reversed(range(1, C + 1)):
        gradients["dW" + str(c)] = 1 / m * np.dot(dZ, activations["A" + str(c - 1)].T)
        gradients["db" + str(c)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        if c > 1:
            dZ = (
                np.dot(parameters["W" + str(c)].T, dZ)
                * activations["A" + str(c - 1)]
                * 1
                - activations["A" + str(c - 1)]
            )

    return gradients


# Update
def update(gradients, parameters, learning_rate):
    C = len(parameters) // 2

    for c in range(1, C + 1):
        parameters["W" + str(c)] = (
            parameters["W" + str(c)] - learning_rate * gradients["dW" + str(c)]
        )
        parameters["b" + str(c)] = (
            parameters["b" + str(c)] - learning_rate * gradients["db" + str(c)]
        )

    return parameters


# Predict
def predict_y(X, parameters) -> bool:
    activations = do_forward_propagation(X, parameters)
    C = len(parameters) // 2
    Af = activations["A" + str(C)]

    return Af >= 0.5


# Deep neural network
def create_neural_network(X, y, hidden_layers: int, learning_rate: int, n_iter: int):
    dimensions = list(HIDDEN_LAYERS)
    dimensions.insert(0, X.shape[0])
    dimensions.append(y.shape[0])
    parameters = init(dimensions)
    C = len(parameters) // 2
    
    # Gradient descent
    training_history = np.zeros((int(n_iter), 2))
    for i in tqdm(range(n_iter)):
        activations = do_forward_propagation(X, parameters)
        gradients = do_back_propagation(y, parameters, activations)
        parameters = update(gradients, parameters, learning_rate)
        Af = activations["A" + str(C)]

        labels =[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], ["1","2"]]
        training_history[i, 0] = log_loss(y.flatten(), Af.flatten(), labels=labels)
        y_pred = predict_y(X, parameters)
        training_history[i, 1] = accuracy_score(y.flatten(), y_pred.flatten())

    # Plot courbe d'apprentissage
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_history[:, 0], label="train loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(training_history[:, 1], label="train acc")
    plt.legend()
    plt.show()

    return training_history


# Prepare CSV dataset
def prepare_csv_dataset(dataset_path: str) -> tuple:
    data = pd.read_csv("py_src\A_Z Handwritten Data.csv").astype("float32")

    X = data.drop("0", axis=1)
    X = X.T
    y = data["0"]
    y = y.values.reshape((1, y.shape[0]))

    return X, y


if __name__ == "__main__":
    HIDDEN_LAYERS = (100, 100, 100)
    LEARNING_RATE = 0.1
    N_ITER = 3000

    X, y = prepare_csv_dataset(DATASET_PATH)
    create_neural_network(X, y, HIDDEN_LAYERS, LEARNING_RATE, N_ITER)
