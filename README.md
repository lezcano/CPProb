# CPProb: Scalable Probabilistic Programming [![Build Status](https://travis-ci.org/Lezcano/CPProb.svg?branch=master)](https://travis-ci.org/Lezcano/CPProb)

## Overview

__CPProb__ is a probabilistic programming library specially designed to perform fast black-box Bayesian inference
on probabilistic models written in arbitrary C++14.

__CPProb__ provides with an easy way to perform Bayesian inference on pre-existing C++ codebases,
making almost no restrictions on the constructions within the language that can be used.
One of the main problems of modern probabilistic programming is the means by which it has to be performed.
Either one can choose to perform it using domain specific languages (DSL), or python-based variational libraries.
In both cases, one has to rewrite their model in the DSL of choice, or in Python. This is not a
satisfactory solution when the model that one has in hand is longer than a few thousand lines of code.

The main goals of __CPProb__ are
1. __Efficiency__:   You don't pay for what you don't use
2. __Scalability__:  It should be easy to use in existing C++ models
3. __Flexibility__:  You should be able to extend it according to your needs.

A model in __CPProb__ is nothing but a C++ function that takes the observations as arguments and
simulates the model via the use of `sample` and `observe` statements. __CPProb__ provides a third
basic statement called `predict, which is used to determine the latent variables from which one wants to
compute their posterior marginal distribution.

As a "hello world" in Bayesian inference, we provide here a conjugate-Gaussian model given by the equations

![Conjugate Gaussian model](https://i.imgur.com/TnMv4dC.png)

and we are interested in computing the posterior distribution

![Posterior distribution](https://imgur.com/LcJSqtQl.png)

These ideas are translated into the following model:
```c++
// File: "models/gaussian.cpp"
#include <boost/random/normal_distribution.hpp>
#include "cpprob/cpprob.hpp"

namespace models {
void gaussian_unknown_mean(const double x1, const double x2)
{
    constexpr double mu0 = 1, sigma0 = 1.5, sigma = 2;      // Hyperparameters

    boost::normal_distribution<> prior {mu0, sigma0};
    const double mu = cpprob::sample(prior, true);
    boost::normal_distribution<> likelihood {mu, sigma};

    cpprob::observe(likelihood, x1);
    cpprob::observe(likelihood, x2);
    cpprob::predict(mu, "Mean");
}
}
```

As an example of the ease of integration of __CPProb__, here we are using the custom distributions
present in the Boost libraries as the sampling and observation distributions. Extending the library in order to
use user-defined distributions doesn't take more than a few lines of code.

## Building CPProb
In order to build __CPProb__ you will need [CMake] and a C++14 compiler. It is being tested with the following compilers:

| Compiler    | Version  |
| ----------- |:--------:|
| GCC         | >= 6     |
| Clang       | >= 3.8   |

__CPProb__ depends on the following libraries:
  * [Boost]
  * [FlatBuffers]
  * [ZeroMQ]

Optionally, in order to perform Inference Compilation, we will also need
  * [Docker]

The [Flatbuffers] and [ZeroMQ] dependencies are handled automatically and can
be installed locally. To do so just run
```shell
(cd dependencies && ./install.sh)
```

To set-up and build the project, the process is standard:
```shell
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH="${PWD}/../dependencies/install"   # Configure the project
cmake --build .                                                 # Build
```

## Probabilistic models
As we saw in the introduction, a probabilistic model is nothing but a callable object such that
takes the observations as parameters and uses the statements `sample`, `observe` and `predict`.
The only requirement for the model is that the model (or at least the entry-point in the case that
there it is composed by several functions) it has to be in the namespace `models`.

Examples of models can be found in the file [include/models/models.hpp](include/models/models.hpp) with
some all-time classics like a Hidden Markov Models (HMM) or Gaussian Linear Models.

## Performing Sequential Importance Sampling (SIS)
The most simple inference algorithm to set up is importance sampling. This algorithm is most suitable for small models that could be approximated with a few million particles. If the model is not very computationally expensive, Importance Sampling can be used to generate lots of particles and get very quick approximate results.

Let's explain how to set-up a simple SIS inference engine on the model that we presented in the introduction. Suppose that we want to do inference on this model with the values `x1 = 3, x2 = 4` with '10.000' particles.
```c++
#include <iostream>
#include <string>
#include <tuple>
#include "models/gaussian.hpp"
#include "cpprob/cpprob.hpp"
#include "cpprob/postprocess/stats_printer.hpp"

int main () {
    const auto observes = std::make_tuple(3., 4.);
    const auto samples = 10'000;
    const std::string outfile = "posterior_sis";
    cpprob::inference(cpprob::StateType::sis, &models::gaussian_unknown_mean, observes, samples, outfile);
    std::cout << cpprob::StatsPrinter{outfile} << std::endl;
}
```

After execution, this script outputs on the console an estimation of the mean and the variance of the posterior distribution for Mu. In this example the true posterior has mean `2.32353` and variance `1.05882`. The post-processing module helps us parsing and computing the estimators, but we also have access to the generated particles ourselves which, in this case, have be been saved in the file `posterior_sis`. This file has the format `([(address, value)] log_weight)`. The file `posterior_sis.ids` contains the different ids of the different addresses. In this case we just have one address (address 0) with id "Mu".

## Performing Inference Compilation (CSIS)
This inference algorithm aims to provide escalable inference in large-scale models, like those where the prior distribution is given by a large simulator. Compilation is slow, given that it has to train a neural network, but after that we have a compiled neural network that can be used to perform fast inference on these models.

Inference Compilation is composed of two steps, as the name says, Inference and Compilation, although not in that order.

The first one, Compilation, consists on training a neural network, in order to perform fast inference afterwards.

To set-up everything up we need to make use of the neural network provided in the folder [infcomp](./infcomp). This neural network, writen in pytorch, depends on [ZeroMQ] and [Flatbuffers] to communicate with __CPProb__. We will build it with [Docker] using the provided [Dockerfile](./Dockerfile). To build it we just have to execute
```shell
docker build -t neuralnet .
```
Now, with a similar script as the first one, we are ready to train the neural network on our Gaussian model
```c++
// File: main_csis.cpp
#include <iostream>
#include <string>
#include <tuple>
#include "models/gaussian.hpp"
#include "cpprob/cpprob.hpp"
#include "cpprob/postprocess/stats_printer.hpp"

int main (int argc, char* argv[]) {
    if (argc != 2) { std::cout << "No arguments provided.\n"; return 1; }
    if (argv[1] == std::string("compile")) {
        cpprob::compile(&models::gaussian_unknown_mean);
    }
    else if (argv[1] == std::string("infer")) {
        const auto observes = std::make_tuple(3., 4.);
        const auto samples  = 100;
        const auto outfile  = std::string("posterior_csis");
        cpprob::inference(cpprob::StateType::csis, &models::gaussian_unknown_mean, observes, samples, outfile);
        std::cout << cpprob::StatsPrinter{outfile} << std::endl;
    }
}
```
We are just missing a folder, to save the trained neural network, suppose that it's called `workspace`, then we can start execute the neural net for compilation and inference respectively with
```shell
mkdir workspace
// Compilation. run first CPProb
./main_csis compile
docker run --rm -it -v $PWD/workspace:/workspace --net=host neuralnet python3 -m main --mode compile --dir /workspace

// Inference. Run first the Neural Network
docker run --rm -it -v $PWD/workspace:/workspace --net=host neuralnet python3 -m main --mode infer --dir /workspace
./main_csis infer
```

> Note: The neural network can be executed in one or several GPUs using `nvidia-docker`.

> Note: Since we have not specified on the __CPProb__ side the number of traces that we want to use for training, the way to finish the training is just by killing the _neuralnet_ job. It is of course possible to specify the number of training examples to use as an optional argument passed to `cpprob::compile`.

## Automatic Model Testing
Writing our own main file whenever we want to test a model might be instructive the first few times, but it becomes tedious rather quickly. For this reason, re provide a console interface in (./src/main.cpp) that helps automatising the process of using the diferent inference algorithms on different models.

The only requisites to use it are to include our model in the `main.cpp` file, link against our model using the `CMakeLists.txt` file, adding the option with a name and a pointer to our model into the `models` variable in the `main()` function and adding the corresponding `if/else` switch below.

After this, we will have access to many different configuration options by default. For example, compiling and executing CSIS on the __CPProb__ side (the neural net still needs to be executed manually) is as easy as calling
```shell
./main --compile --model my_model
./main -sis --csis --estimate --model my_model --n_samples 100 --observes "3 4"
```
where we are executing both SIS and CSIS inference with 100 particles each and observations `x1 = 3` and `x2 = 4`.


## References
An in-depth explanation of __CPProb__'s design can be found [here](./doc/compiled_inference.pdf):
```
@mastersthesis{lezcano2017cpprob,
    author    = "Mario Lezcano Casado",
    title     = "Compiled Inference with Probabilistic Programming for Large-Scale Scientific Simulations",
    school    = "University of Oxford",
    year      = "2017"
}
```

A large-scale application of __CPProb__:
```
@article{lezcano2017improvements,
  title={Improvements to Inference Compilation for Probabilistic Programming in Large-Scale Scientific Simulators},
  author={Lezcano Casado, Mario and Baydin, Atilim Gunes and Mart{\'\i}nez Rubio, David and Le, Tuan Anh and Wood, Frank and Heinrich, Lukas and Louppe, Gilles and Cranmer, Kyle and Ng, Karen and Bhimji, Wahid and Prabhat},
  journal={arXiv preprint arXiv:1712.07901},
  year={2017}
}
```

## License
Please see [LICENSE](LICENSE).

<!-- Links -->
[CMake]: http://www.cmake.org
[Boost]: http://www.boost.org/
[FlatBuffers]: https://google.github.io/flatbuffers/
[ZeroMQ]: http://zeromq.org/
[Docker]: https://www.docker.com/
[lezcano2017]: https://arxiv.org/pdf/1712.07901.pdf
