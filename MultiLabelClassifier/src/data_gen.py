'''Author: Tanisha Ghosal, ID: 20783089'''

# Import
import numpy as np

'''Class to read in MNIST data and produce arrays of our testing and training datasets. '''
class Data():
    def __init__(self, path):
        
        # Loading in data
        MNIST_data = np.loadtxt(path)
        np.random.shuffle(MNIST_data)
        
        #Dimensions of data
        dimensions = MNIST_data.shape
        input_size = dimensions[1]-1   
        
        # Shuffling the data
        np.random.shuffle(MNIST_data)

        # Splitting up the test data and the training data (we use the first 26492 samples for training
        # and 3000 for testing)
        test_data, train_data = np.split(MNIST_data,[3000])
        
        # Resizing the training data and testing data for our inputs into arrays
        x_test=test_data[:,0:input_size]
        x_test= np.array(x_test, dtype= np.float32)
        x_train=train_data[:,0:input_size]
        x_train= np.array(x_train, dtype= np.float32)

        # Resizing the training data and testing data for our targets into arrays
        y_test=test_data[:,[-1]]
        y_test= np.array(y_test, dtype= np.float32)
        y_train=train_data[:,[-1]]
        y_train= np.array(y_train, dtype= np.float32)

        print(x_train)

        # Outputs of our class
        self.x_train= x_train
        self.y_train= y_train
        self.x_test= x_test
        self.y_test= y_test
