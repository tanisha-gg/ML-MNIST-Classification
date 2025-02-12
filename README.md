# VAE and Multilabel Classifier

This repository contains implementations of a **Variational Autoencoder (VAE)** and a **Multilabel Classifier** for the MNIST dataset. The VAE is used for image generation, while the classifier is used for digit classification.

## Requirements
- Python 3.8+
- Install dependencies: Torch, NumPy

---

## Usage

### 1. **Train the VAE**
To train the Variational Autoencoder (VAE), run the following command:
```
python main.py --model vae --param data/param.json --res_path results/vae/ --verbosity 2
```
Output:

Training loss plot: results/vae/loss.pdf

Reconstructed images: results/vae/reconstructed_images/```

### 2. **Train the Multilabel Classifier**
```
python main.py --model classifier --param data/param.json --res_path results/classifier/ --verbosity 2
```
Output:

Training loss and accuracy plot: results/classifier/training_metrics.pdf

### Training Details

## VAE:

Defined in src/vae_model.py.

Architecture:
Encoder: Two fully connected layers to compute mean (mu) and log variance (logvar).
Decoder: Two fully connected layers to reconstruct the input.
Loss: Combines reconstruction loss (Binary Cross-Entropy) and KL divergence.

Optimizer: Adam.
Loss: Reconstruction + KL divergence.
Saves training loss plot.

## Classifier:

Defined in src/classifier_model.py.

Architecture:
Three fully connected layers with ReLU activation.
Loss: Cross-Entropy Loss.

Optimizer: SGD.
