import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("py_src\A_Z Handwritten Data.csv").astype('float32')
print(data.head(10))

X = data.drop('0',axis = 1)
y = data['0']

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2)
train_x = np.reshape(train_x.values, (train_x.shape[0], 28,28))
test_x = np.reshape(test_x.values, (test_x.shape[0], 28,28))

print("Train data shape: ", train_x.shape)
print("Test data shape: ", test_x.shape)