import numpy as np
import torch

class ClassifierData:
    def __init__(self, path):
        # Loading in data
        MNIST_data = np.loadtxt(path)
        np.random.shuffle(MNIST_data)
        
        # Dimensions of data
        dimensions = MNIST_data.shape
        input_size = dimensions[1] - 1   

        # Splitting up the test data and the training data
        test_data, train_data = np.split(MNIST_data, [3000])
        
        # Resizing the training data and testing data for our inputs
        x_test = test_data[:, 0:input_size]
        x_test = np.array(x_test, dtype=np.float32)
        x_train = train_data[:, 0:input_size]
        x_train = np.array(x_train, dtype=np.float32)

        # Resizing the training data and testing data for our targets
        y_test = test_data[:, [-1]]
        y_test = np.array(y_test, dtype=np.float32)
        y_train = train_data[:, [-1]]
        y_train = np.array(y_train, dtype=np.float32)

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
