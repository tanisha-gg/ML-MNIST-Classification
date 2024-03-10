# Write your assignment here
'''Author: Tanisha Ghosal, ID: 20783089'''

# Imports
import numpy as np
import json, argparse, torch, sys
import torch.optim as optim
import matplotlib.pyplot as plt
# Adding path with our custom classes
sys.path.append('src')
from data_gen import Data
from nn_gen import Net

# This code was constructed by referencing Workshop 2 shown in class.

'''Preparing the data to be analyzed by defining path to data file and calling our Data class which
sets up the training and test data. We then creating the model by calling our Net class which 
is the neural network that trains and tests on the data.'''

def prep_demo():
    # Construct dataset and model 
    path = 'data/even_mnist.csv'
    data = Data(path)
    model = Net()
    return model, data

'''The function that is run by optimizing our model parameters using Stochastic Gradient Descent
and calculating the loss via a Cross Entropy Loss (due to multi-label classification).
The loss values and accuracy values are also calculated from the model and printed as the code is run.'''
def run_demo(param, model, data):

    # Define an optimizer and the loss function
    optimizer = optim.SGD(model.parameters(), lr=param['learning_rate'])
    loss= torch.nn.CrossEntropyLoss(reduction= 'mean')

    # Lists to hold our loss and accuracy values
    obj_vals= []
    cross_vals= []
    accuracytrain = []
    accuracytest = []

    # Number of epochs from the .json file
    num_epochs= int(param['num_epochs'])

    # Training loop for our Neural Network, we loop through the number of epochs (times that we want
    # to train our dataset.)
    for epoch in range(1, num_epochs + 1):

        # Calling our model and appending the loss and accuracy values to our lists
        train_val, acctrain= model.backprop(data, loss, optimizer)
        obj_vals.append(train_val)
        accuracytrain.append(acctrain)
        
        # Calling our test and appending the loss and accuracy values to our lists
        test_val, acctest = model.test(data, loss)
        cross_vals.append(test_val)
        accuracytest.append(acctest)

        # Higher verbosity mode means that we print our loss and accuracy status at every display epoch.
        if int(args.verbosity) > 1:

            if (epoch+1) % param['display_epochs'] == 0:
                print('Epoch [{}/{}]'.format(epoch+1, num_epochs)+\
                        '\tTraining Loss: {:.4f}'.format(train_val)+\
                        '\tTest Loss: {:.4f}'.format(test_val)+\
                        '\tTraining Accuracy: {:.4f}'.format(acctrain)+\
                        '\tTest Accuracy: {:.4f}'.format(acctest))
    
    # Lower verbosity mode statement to let user know that the code has run.  
    if int(args.verbosity) <= 1:
        print('The neural network has finished running.')
    
    # Printing our final report of the loss and accuracy at the end for both verbosity modes.
    print("\u0332".join('Final Report'))
    print('Final Training Loss: {:.4f}'.format(obj_vals[-1]))
    print('Final Test Loss: {:.4f}'.format(cross_vals[-1]))
    print('Final Training Accuracy:{:.4f}'.format(accuracytrain[-1]) + ' %')
    print('Final Test Accuracy:{:.4f}'.format(accuracytest[-1]) + ' %')

    return obj_vals, cross_vals, accuracytrain, accuracytest

'''Function that plots our loss and accuracy'''
def plot_results(obj_vals, cross_vals, accuracytrain, accuracytest):
    assert len(obj_vals)==len(cross_vals), 'Length mismatch between the curves'
    num_epochs= len(obj_vals)

    # Loss and Accuracy plots saved in results folder
    plt.plot(range(num_epochs), obj_vals, label= "Training Loss", color="orchid")
    plt.plot(range(num_epochs), cross_vals, label= "Test Loss", color= "yellowgreen")
    plt.grid()
    plt.ylabel('Loss')
    plt.xlabel('Number of Epochs')
    plt.title('Loss vs. Number of Epochs')
    plt.legend()
    plt.savefig(args.res_path + '/loss.pdf')
    plt.close()

    plt.plot(range(num_epochs), accuracytrain, label= "Training Accuracy", color="lightskyblue")
    plt.plot(range(num_epochs), accuracytest, label= "Test Accuracy", color= "maroon")
    plt.grid()
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Number of Epochs")
    plt.legend()
    plt.savefig('results/accuracy.pdf')
    plt.close()

'''Taking in our command line arguments and calling our setting up our neural network run.'''
if __name__ == '__main__':

    # Command line arguments with defaults
    parser = argparse.ArgumentParser(description='ML with PyTorch')
    parser.add_argument('--param', help='Path to file with parameters (please include .json extension)', default = 'data/param.json')
    parser.add_argument('--res-path', help='Path to results folder', default = 'results/')
    parser.add_argument('--verbosity', help='Verbosity level. If =<1, the level of verbosity is simple. If >1, the level of verbosity is more detailed.', default=2)
    args = parser.parse_args()

    # Hyperparameters from json file
    with open(args.param) as paramfile:
        param = json.load(paramfile)

    # Using our allocated CPU and resetting the model parameters for the given run
    model = Net().to(torch.device("cpu"))
    model.reset()

    # Prepping dataset and model
    model, data = prep_demo()
    
    # Low verbosity mode statement so that user knows code has started running
    if int(args.verbosity) <= 1:
        print('The data has been read in. Neural network learning in progress...')
        
    # Running neural network
    obj_vals, cross_vals, accuracytrain, accuracytest = run_demo(param, model, data)

    # Plotting results
    plot_results(obj_vals, cross_vals, accuracytrain, accuracytest)
