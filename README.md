# CPProb [![Build Status](https://travis-ci.com/probprog/CPProb.svg?token=p9LTU5yGsuwiT6ypq45J&branch=master)](https://travis-ci.com/probprog/CPProb)

> Library to perform probabilistic inference in C++14.

## Overview

This repository contains a library to allow probabilistic programming in C++14.

## Building CPProb
In order to build ALCP you will need [CMake][] and a C++14 compiler.

All the dependancies are automatically managed by CMake. If CMake does not find
a dependance, it will download and configure it automatically.

The dependancies are the following:
  * [Boost]
  * [FlatBuffers]
  * [ZeroMQ]

In general, you will need to specify a path to a modern C++14 complaint
compiler using the `-DCMAKE_CXX_COMPILER` variable if the default compiler
is too old.


The compile process is the same as usual, configure the project and compile
it.
```shell
mkdir build && cd build && cmake ..
cmake --build .
```

## Setting up a new probabilistic model
In order to set up a new probabilistic model, we just have to add a function that
recevies the observe values as inputs and returns nothing.

In this function, we have to use the functions `cpprob::sample` to sample from
distributions in boost, `cpprob::observe` to condition on a value and `cpprob::predict`
to be controlled by the program. All these statements are present in `include/cpprob/cpprob.hpp`.

If we see the probabilistic program as a distribution with probabilities p(X, Y), the 
distribution on which we will do inference is p(X | Y) where X = (x_1, ..., x_n) are the
values that appear in the predict statements and Y = (y_1, ..., y_m) are the input paramenters
for our model and hence the variables that appear in the observe statements.

Examples of models can be found in `src/models/models.cpp`.

Once we have the model defined, all we have to set the variable `f` in `src/main.cpp` 
to point to this function. 

## Compiling a Neural Network
In the compilation step, the neural network will ask the library for traces. Hence, cpprob
will run as a server and the neural network will run as a client.

If we are going to run the neural network and the main program in the same computer
we can run

```shell
./cpprob --mode compile
python -m infcomp.compile
```

If we are going to run the neural network and the main program in different computers, we have in
both programs the option `--tcp_addr` to set up the tcp address and port in which host the server.

## Performing Inference with a Traiend Neural Network
In the inference step, the library will ask the neural network for proposals.
In this case, cpprob will run as a client and the neural network will run as a server.

In this case we have to specify a file with the observed values on which we want to perform
inference with the `--observes_file` flag or we can input them with the `--observes` flag.
We always have to provie one of the two flags.

The file will have one value per line, while the if we use the `--observes` flag, the values will
be separated with spaces.

In any agregate type, the elements will be separated with spaces.

A vector or array will be enclosed in brackets. A pair or tuple will be enclosed in parenthesis.

When using the `--observes` flag, the list of parameters has to be enclosed between quotes.

If we are to perform inference on a function with signature
```C++
void model(const std::array<std::pair<double, double>, 6> &, int a, double b);
```
we can execute the inference engine as
```shell
./cpprob --mode infer --observes "[(1 2.1) (2 3.9) (3 5.3) (4 7.7) (5 10.2) (6 12.9)] 2 3.78"
python -m infcomp.infer
```
## License
Please see [LICENSE](LICENSE).

<!-- Links -->
[CMake]: http://www.cmake.org
[Boost]: http://www.boost.org/
[FlatBuffers]: https://google.github.io/flatbuffers/
[ZeroMQ]: http://zeromq.org/

