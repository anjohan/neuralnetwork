# Neural Network

This repository contains a simple, fully connected, dense, deep neural network,
implemented in modern Fortran and parallelised using coarrays.

The implementation is highly object oriented for ease of reuse and extension.

## Usage

A neural network can be constructed using
``` Fortran
use mod_neural_network
class(neural_network), allocatable :: nn

nn = neural_network(number of inputs,
                    numbers of neurons per layer (including output layer),
                    activation function in hidden layers - e.g. relu() or sigmoid(),
                    activation function in output layer (optional),
                    L2 regularisation parameter (optional))
```
Example: A neural network with

- 2 inputs
- 3 hidden layers with RELU activation functions and 30, 20 and 10 neurons
- 1 output using a sigmoid function in the output layer (e.g. for binary classification)
- L2 regularisation parameter 0.01

can be constructed with
``` Fortran
nn = neural_network(2, [30, 20, 10, 1], relu(), sigmoid(), 0.01d0)
```
and trained with a learning rate of 0.001 for 100 epochs with a batch size of 32 with
``` Fortran
call nn%train(X, Y, 0.001d0, 100, 32)
```
where `X` has dimension `2 x N` and `Y` has dimension `1 x N` (where `N` is the number of training samples). Note that `Y` has to be a matrix even though the network only gives one output per input.

## Compilation and installation
Prerequisites:

- `gfortran`, tested with 8.2. It seems no other compilers support `co_sum` yet.
- `cmake`.

Compilation sequence is the usual,
```
git clone https://github.com/anjohan/neuralnetwork.git
cd neuralnetwork
mkdir build
cd build
cmake .. # or FC=caf cmake ..
make
```

This (hopefully) gives the library `libneuralnetwork.a`

For parallel execution, [OpenCoarrays](https://github.com/sourceryinstitute/OpenCoarrays) is required. Run `FC=caf cmake ..` if the `caf` wrapper is in your `PATH`. If the compiler does not contain `caf`, `-fcoarray=single` is used for serial execution.
