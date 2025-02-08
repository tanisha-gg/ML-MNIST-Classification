import argparse
import json
import torch
import matplotlib.pyplot as plt
from src.data_gen.vae_data import VAEData
from src.data_gen.classifier_data import ClassifierData
from src.vae import VAE
from src.classifier import Net

def run_vae(param, res_path, verbosity):
    # Initialize data and model
    data = VAEData('data/even_mnist.csv')
    model = VAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=param['learning_rate'])

    # Training loop
    loss_train = []
    for epoch in range(param['num_epochs']):
        loss = model.backprop(data.inputs, optimizer, param['batch_size'])
        loss_train.append(loss)
        if verbosity > 1 and (epoch + 1) % param['display_epochs'] == 0:
            print(f'Epoch [{epoch+1}/{param["num_epochs"]}]\tLoss: {loss:.4f}')

    # Save results
    plt.plot(loss_train)
    plt.savefig(f'{res_path}/vae/loss.pdf')
    plt.close()

def run_classifier(param, res_path, verbosity):
    # Initialize data and model
    data = ClassifierData('data/even_mnist.csv')
    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=param['learning_rate'])
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop
    loss_train, accuracy_train = [], []
    for epoch in range(param['num_epochs']):
        loss, accuracy = model.backprop(data, loss_fn, optimizer)
        loss_train.append(loss)
        accuracy_train.append(accuracy)
        if verbosity > 1 and (epoch + 1) % param['display_epochs'] == 0:
            print(f'Epoch [{epoch+1}/{param["num_epochs"]}]\tLoss: {loss:.4f}\tAccuracy: {accuracy:.4f}')

    # Save results
    plt.plot(loss_train, label="Loss")
    plt.plot(accuracy_train, label="Accuracy")
    plt.legend()
    plt.savefig(f'{res_path}/classifier/training_metrics.pdf')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run VAE or Multilabel Classifier.")
    parser.add_argument("--model", type=str, required=True, choices=["vae", "classifier"], help="Model to run: 'vae' or 'classifier'.")
    parser.add_argument("--param", type=str, default="data/param.json", help="Path to parameter JSON file.")
    parser.add_argument("--res_path", type=str, default="results/", help="Path to results directory.")
    parser.add_argument("--verbosity", type=int, default=2, help="Verbosity level (1 for simple, 2 for detailed).")
    args = parser.parse_args()

    # Load parameters
    with open(args.param) as paramfile:
        param = json.load(paramfile)

    # Run the selected model
    if args.model == "vae":
        run_vae(param, args.res_path, args.verbosity)
    elif args.model == "classifier":
        run_classifier(param, args.res_path, args.verbosity)

if __name__ == "__main__":
    main()
