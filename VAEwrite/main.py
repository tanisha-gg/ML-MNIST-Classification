# Write your assignment here
'''Author: Tanisha Ghosal, ID: 20783089'''

# Imports
import numpy as np
import json, argparse, torch, sys
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import rc
# Adding path with our custom classes
sys.path.append('src')
from src.data_gen import Data
from src.nn_gen import VAE

# Loss Function
def loss_func(inputs, mu, log_var, outputs, ns):

    # Reshaping inputs and outputs.
    inputs=inputs.reshape(ns,n_bits*n_bits)
    outputs=outputs.reshape(ns,n_bits*n_bits)
    
    # Finding the likelihood for the reconstruction (difference between inputs and outputs).
    rec_likehood=(torch.sum((inputs-outputs)**2, dim=1))

    # Calculating the KL divergence using a Normal distribution.
    kl_div = -0.5 * torch.sum(1 + log_var - torch.square(mu) - torch.exp(log_var))

    # The loss is the sum of the reconstruction term and the KL term.
    loss=torch.mean(rec_likehood+kl_div)

    return loss

# Training loop
def run_demo(lr, epochs, model, sorted_data, ns):

    # Instance of our data generation class.
    inputs = sorted_data.inputs

    #Transform the numpy array into a tensor
    inputs= torch.from_numpy(inputs)
    inputs=inputs.float()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr)  # We use gradient descent 

    #List that will contain the loss for each epoch
    loss_train= []
    num_epochs= int(epochs)

    if int(args.verbosity) <= 1:
        print('Training in progress...')

    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_val = model.backprop(inputs, loss_func, optimizer, ns)
        loss_train.append(train_val)
        if int(args.verbosity) > 1:
            if (epoch+1) % disp_epochs == 0:
                print('Epoch [{}/{}]'.format(epoch, num_epochs)+\
                          '\tTraining Loss: {:.4f}'.format(train_val))
 
    print('Final training loss: {:.4f}'.format(loss_train[-1]))
    print('Please see the loss plot in the results folder.')

    return loss_train

def plot_img(inputs, outputs, n_bits):
    # Plotting the reconstructed images
    if (int(args.verbosity) <= 1):
        print('Creating reconstructed images...')
    for n in range(len(inputs)):
        if (int(args.verbosity) > 1):
            print('Creating reconstructed image #{0}.'.format(n+1))
            
        rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        rc('text', usetex=True)

        fig, ax = plt.subplots(1,2)
        ax[0].imshow(np.reshape(inputs[n].detach().numpy(), (n_bits,n_bits)), cmap = 'plasma')
        ax[0].set_title('Data')
        ax[1].imshow(np.reshape(outputs[n].detach().numpy(), (n_bits,n_bits)), cmap = 'plasma')
        ax[1].set_title('Reconstructed')
        plt.savefig(args.res_path + '/'+str(n)+'.pdf')
        plt.close()

def plot_loss(train_values):
    # For plotting loss
    num_epochs = num_epochs= len(train_values)

    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    # Save in results folder
    plt.plot(range(num_epochs), train_values, label= "Training Loss", color="orchid")
    plt.grid()
    plt.ylabel('Loss')
    plt.xlabel('Number of Epochs')
    plt.title('Loss vs. Number of Epochs')
    plt.legend()
    plt.savefig(args.res_path + '/loss.pdf')
    plt.close()

if __name__ == '__main__':
    # Command line arguments
    parser= argparse.ArgumentParser(description='Inputs for VAE')
    parser.add_argument("-n", default="100" ,help="Number of samples to be drawn.")
    parser.add_argument("-param", default="data/param.json" ,help="Input path to .json file with parameter values.")
    parser.add_argument("-res_path", default="results_dir/" ,help="File path to results directory.")
    parser.add_argument("-verbosity" , default=2, help='Verbosity level. If =<1, the level of verbosity is simple. If >1, the level of verbosity is more detailed.')

    args = parser.parse_args()
    
    # Load data
    path = 'data/even_mnist.csv'
    data=np.loadtxt(path)

    # Parameter file
    with open(args.param) as paramfile:
        param = json.load(paramfile)

    # Defining parameters
    num_epochs = int(param['num_epochs'])
    lr = param['learning_rate']
    disp_epochs = param['display_epochs']
    
    # Number of samples to draw (and number of images to construct)  
    n=int(args.n)

    # Data dimensions
    dim=data.shape
    n_s=dim[0]    # Number of samples
    i_d=dim[1]-1  # Dimensions of data
    n_bits=int(np.sqrt(i_d))    # Number of bits

    # Create an instance of the model
    model = VAE()

    # Create an instance of the dataset
    path = 'data/even_mnist.csv'
    sorted_data = Data(path)
    
    # Run the model 
    loss_train = run_demo(lr, num_epochs, model, sorted_data, n_s)

    # Plot and save the loss
    plot_loss(loss_train)

    # Setting our training data from instance of data generation class
    inputs = sorted_data.inputs

    # Choose random index to sample from the inputs (to test and reconstruct)
    index = np.arange(0,n_s,1)
    sample_index = np.random.choice(index, n, replace=False)
    test_samples = np.take(inputs, sample_index, axis=0)
    test_samples = torch.Tensor(test_samples)
    
    # Generate an output on our testing data
    _,__,test_output = model.forward(test_samples,n)
    
    # Plotting and saving the n reconstructed images
    plot_img(test_samples, test_output, n_bits)

