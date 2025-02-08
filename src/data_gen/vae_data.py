import numpy as np
import torch

class VAEData:
    def __init__(self, path):
        # Loading in data
        MNIST_data = np.loadtxt(path)
        np.random.shuffle(MNIST_data)
        
        # Dimensions of data
        dimensions = MNIST_data.shape
        input_size = dimensions[1] - 1   
        train_data = MNIST_data

        # Resizing the training data for our inputs
        x_train = train_data[:, 0:input_size]
        x_train = np.array(x_train, dtype=np.float32)
        x_train = x_train.reshape(len(x_train), 1, 14, 14)

        # Normalizing
        x_train = x_train / 255.0

        self.inputs = x_train
