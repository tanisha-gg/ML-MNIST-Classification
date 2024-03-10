# PHYS449 Assignment 5

Designing a VAE to write even digits after being trained on MNIST data. 

## Dependencies

- json
- numpy
- argparse
- torch
- sys
- matplotlib

## Running `main.py`

To run `main.py` with the default arguments (using the default number of samples, default parameter file path, default results path, and highest verbosity mode), you would enter:

```sh
python main.py
```

If you wish to change the arguments, then please enter the full arguments. For example, for 50 samples, for a file path to the parameter .json file called "parameter_filepath", a file path to the MNIST data of "results_filepath", and a less verbose mode of 1, you would enter:

```sh
python main.py -n 50 -param parameter_filepath -res-path results_filepath -verbosity 1
```

Please see the reconstructed images and the loss in the results directory. 
