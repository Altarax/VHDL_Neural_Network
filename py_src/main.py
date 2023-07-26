# The following code is based on Machine Learnia videos : https://github.com/MachineLearnia/Deep-Learning-Youtube

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm

DATASET_PATH = "py_src\A_Z Handwritten Data.csv"

# Initialisation
def init(dataset_dimensions : int) -> dict:
    parameters = {}
    return parameters

# Forward propagation
def do_forward_propagation(X, parameters):
    pass

# Back propagation
def do_back_propagation(y, parametres, activations):
    pass

# Update
def update(gradients, parametres, learning_rate):
    return parametres

# Predict
def predict_y(X, parameters) -> bool:
    pass

# Deep neural network
def create_neural_network(X, y, hidden_layers : int, learning_rate : int, n_iter : int):
    pass

# Prepare CSV dataset3
def prepare_csv_dataset(dataset_path : str) -> tuple:
    return X, y

if __name__ == '__main__':
    HIDDEN_LAYERS = (16, 16 ,16)
    LEARNING_RATE = 0.1
    N_ITER = 3000

    X, y = prepare_csv_dataset(DATASET_PATH)
    create_neural_network(X, y, HIDDEN_LAYERS, LEARNING_RATE, N_ITER)