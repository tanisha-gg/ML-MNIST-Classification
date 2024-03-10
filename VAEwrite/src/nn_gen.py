'''Author: Tanisha Ghosal, ID: 20783089'''

import torch 
import torch.nn as nn

'''
    VAE class with two neural networks - the encoder and decoder. 
    Architecture:
        1) The encoder includes one convolutional layers and two linear layers (to find the mean and variance).
        2) We do sampling to find a parameterization which calculates the normal distribution sample for the decoder.
        3) Decoder: One linear layer and two convolutional layers which invert the dimensions to match the encoder.
    '''
class VAE(nn.Module):
    
    def __init__(self):
        super(VAE, self).__init__()

        # Hidden layer 
        n_hidden = 7*7*10
        # Latent dimensions for decoder
        latent_dims = 50
        
        # Encoder layers
        self.encode1 = nn.Sequential(\
            nn.Conv2d(in_channels=1,out_channels=5,kernel_size=3,stride=1,padding=1),\
            nn.ReLU())
        
        self.encode2=nn.Sequential(\
            nn.Conv2d(in_channels=5,out_channels=10,kernel_size=4,stride=2,padding=1),\
            nn.ReLU())
        
        # Linear layers (for mean and variance)
        self.mu_fc = nn.Linear(n_hidden,latent_dims)
        self.var_fc = nn.Linear(n_hidden, latent_dims) 
        
        # Decoder layers
        self.decode_fc = nn.Linear(latent_dims, n_hidden)
        
        self.decode2 = nn.Sequential(\
            nn.ConvTranspose2d(in_channels=10, out_channels=5, kernel_size=4, stride=2, padding=1),\
            nn.ReLU())
        
        self.decode1 = nn.Sequential(\
            nn.ConvTranspose2d(in_channels=5, out_channels=1, kernel_size=3, stride=1, padding=1),\
            nn.ReLU())
    
    # Encoding
    def encoder(self, inputs):
        
        # Convolutional layer 
        h = self.encode1(inputs)     
        z = self.encode2(h)
        z_reshape = z.reshape(z.size(0),-1)
        
        # Calculate mu (mean) and the log of the variance 
        mu = self.mu_fc(z_reshape)                 
        log_var = self.var_fc(z_reshape)

        return mu, log_var

    # Decoding
    def decoder(self, sample, ns):

        # Decoding and reshaping to match input
        inverse = self.decode_fc(sample)
        inv_reshape = inverse.reshape([ns, 10, 7, 7])
        h = self.decode2(inv_reshape)
        z = self.decode1(h)
                
        return z
    
    # Feed forward
    def forward(self, inputs, ns):
        # Apply encoding
        mu, log_var = self.encoder(inputs)  

        # Sample from a Normal N(0,1) distribution, sample = mu+sigma*epsilon
        std = torch.exp((1/2)*log_var)
        epsilon = torch.randn_like(std)
        sample = mu+ (std * epsilon)   # Sample is mean + sigma*epsilon

        # Decoding 
        output = self.decoder(sample,ns)
        
        return mu, log_var, output
    
    # Backpropagation function
    def backprop(self, inputs, loss, optimizer, ns):
        self.train()

        # Run the VAE and calculate the loss
        mu, log_var, outputs = self.forward(inputs, ns)
        obj_val = loss(inputs, mu, log_var, outputs, ns)          
        
        # Backpropagate the gradient for the training data        
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        return obj_val.item()    
