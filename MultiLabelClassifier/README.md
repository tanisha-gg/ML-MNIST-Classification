# PHYS449

Designing a neural network to implement a multi-label classifier on the even digits of MNIST data. 

## Dependencies

- json
- numpy
- argparse
- torch
- sys
- matplotlib

## Running `main.py`

To run `main.py` with the default arguments (using the default parameter file path, results path, and highest verbosity mode), you would enter:

```sh
python main.py
```

If you wish to change the arguments, then please enter the full arguments. For example, for a file path to the parameter .json file called "parameter_filepath", a file path to the MNIST data of "results_filepath", and a less verbose mode of 1, you would enter:

```sh
python main.py --param parameter_filepath --res-path results_filepath --verbosity 1
```
