'''Author: Tanisha Ghosal, ID: 20783089'''

# Imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func

'''Neural network class.
    Architecture:
        Input layer fc1, four varying hidden layers of Relu and Linear activation functions, 
        and output layer fc2  
    '''
class Net(nn.Module):
    def __init__(self):

        # Inheriting class attributes
        super(Net, self).__init__()

        # Input dimension of data
        n_bits = 196
        # First hidden layer
        n_hidden1 = 100
        # Second hidden layer 
        n_hidden2 = 50
        #Third hidden layer
        n_hidden3 = 10
        # Fourth hidden layer - output is 5 due to 5 different classes of digits
        n_out = 5

        # Input
        self.fc1= nn.Linear(n_bits, n_hidden1)  

        # Activation functions in hidden layer - I used 
        # fully connected fc1 and fc2 layers at first but I chose to use hidden layers to improve the accuracy. 
        self.modelstack = nn.Sequential(nn.Linear(n_hidden1, n_hidden2),
                            nn.ReLU(),
                            nn.Linear(n_hidden2, n_hidden3),
                            nn.ReLU())
        self.fc2= nn.Linear(n_hidden3, n_out)


    # Feedforward function
    def forward(self, x):
        # Input
        h = self.fc1(x)
        # Hidden layers
        h = self.modelstack(h)
        # Output
        y = self.fc2(h)
        return y

    '''Reset function for the training weights since same network is trained multiple times.'''
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    '''Backpropagation function (following Workshop 2 code)'''
    def backprop(self, data, loss, optimizer):
        self.train()
        
        # Defining the inputs and targets as Torch tensors
        targets= torch.flatten(torch.from_numpy(data.y_train))
        inputs = torch.from_numpy(data.x_train)
        targets = targets/2
        targets = targets.type(torch.LongTensor)

        # Finding the loss of the training data and implementing the optimizer
        obj_val = loss(self.forward(inputs), targets)
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()

        # Calculating the accuracy
        predictedtrain = self.forward(inputs).tolist()
        actualtrain = targets.tolist()
        predictiontrain = np.argmax(predictedtrain, axis = 1)
        accuracytrain = (actualtrain==predictiontrain).astype(int)
        accuracytrain = sum(accuracytrain.astype(int))/len(accuracytrain)*100
        
        return obj_val.item(), accuracytrain

    # Test function. Avoids calculation of gradients.
    def test(self, data, loss):
        self.eval()
        with torch.no_grad():

            # Defining the inputs and targets as Torch tensors
            inputs= torch.from_numpy(data.x_test)
            targets= torch.flatten(torch.from_numpy(data.y_test))
            targets = targets/2
            targets = targets.type(torch.LongTensor)

             # Finding the loss of the test data
            cross_val= loss(self.forward(inputs), targets)
            targetstest = torch.from_numpy(data.y_test)
            inputstest = torch.from_numpy(data.x_test)
            targetstest = (targetstest/2)

            # Calculating the accuracy
            predictedtest = self.forward(inputs).tolist()
            actualtest = targets.tolist()
            predictiontest = np.argmax(predictedtest, axis = 1)
            accuracytest = (actualtest==predictiontest).astype(int)
            accuracytest = sum(accuracytest.astype(int))/len(accuracytest)*100

        return cross_val.item(), accuracytest
