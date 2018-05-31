# CPProb [![Build Status](https://travis-ci.com/Lezcano/CPProb.svg?token=p9LTU5yGsuwiT6ypq45J&branch=master)](https://travis-ci.com/Lezcano/CPProb)

> Library to perform probabilistic inference in C++14.

## Overview

__CPProb__ is a CPProb library specially designed to perform fast black-box Bayesian inference
on probabilistic models written in arbitrary C++14.

__CPProb__ provides with an easy way to perform Bayesian inference on pre-existing C++ codebases,
making almost no restrictions on the constructions within the language that can be used.
One of the main problems of modern probabilistic programming is the means by which it has to be performed.
Either one can choose to perform it using domain specific languages (DSL), or python-based variational libraries.
In both cases, one has to rewrite their model in the DSL of choice, or in Python. This is not a
satisfactory solution when the model that one has in hand is longer than a few thousand lines of code.

The main goals of __CPProb__ are
  * Efficiency:   You don't pay for what you don't use
  * Scalability:  It should be easy to use in existing C++ models
  * Flexibility:  The user should be able to extend it according to her needs.

A model in __CPProb__ is nothing but a C++ function that takes the observations as arguments and
simulates the model via the use of `sample` and `observe` statements. __CPProb__ provides a third
basic statement called `predict, which is used to determine the latent variables from which one wants to
compute their posterior marginal distribution.

As a "hello world" in Bayesian inference, we provide here a conjugate-Gaussian model given by the equations
[Conjugate Gaussian model](https://i.imgur.com/TnMv4dC.png)
and we are interested in computing the posterior distribution
[Posterior distribution](https://imgur.com/LcJSqtQl.png)

These ideas are translated into the following model:
```c++
#include <boost/random/normal_distribution.hpp>
#include "cpprob/cpprob.hpp"

namespace models {
void gaussian_unknown_mean(const double x1, const double x2)
{
    constexpr double mu0 = 1, sigma0 = 1.5, sigma = 2;      // Hyperparameters

    boost::normal_distribution<RealType> prior {mu0, sigma0};
    const RealType mu = cpprob::sample(prior, true);
    boost::normal_distribution<RealType> likelihood {mu, sigma};

    cpprob::observe(likelihood, x1);
    cpprob::observe(likelihood, x2);
    cpprob::predict(mu, "Mean");
}
}
```

As an example of the ease of integration of __CPProb__, here we are using the custom distributions
present in the Boost libraries as the sampling and observation distributions. Extending the library in order to
use user-defined distributions is a matter of a few lines of code.

## Building CPProb
In order to build __CPProb__ you will need [CMake] and a C++14 compiler. It is being tested with the following compilers:

| Compiler    | Version  |
| ----------- |:--------:|
| GCC         | >= 6     |
| Clang       | >= 3.8   |

__CPProb__ has the following dependencies:
  * [Boost]
  * [FlatBuffers]
  * [ZeroMQ]

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

Examples of models can be found in the file [`include/models/models.hpp`](include/models/models.hpp) with
some all-time classics like a Hidden Markov Models (HMM) or Gaussian Linear Models.

## Performing Sequential Importance Sampling (SIS)
The most simple inference algorithm to set up is importance sampling. This algorithm is most suitable for small models that could be approximated with a few million particles. If the model is not very computationally expensive, Importance Sampling can be used to generate lots of particles and get very quick approximate results.

We can set-up inference using __CPProb__, if we assume that the model that we want to perform inference on is in the file `model.hpp`, the entry point function is called `mymodel`, the function takes two `double`s as argments, as it was the case in the introduction for `gaussian_unkown_mean`, and we wanted to peform inference with values `x1=3, x2=4`, we can write
```c++
#include <iostream>
#include <string>
#include <tuple>
#include "model.hpp"
int main () {
    const std::tuple<double, double> observes{3, 4};
    const std::string outfile = "posterior_is.txt"
    const std::size_t n_samples = 10'000;

    cpprob::generate_posterior(&models::mymodel, observes, "", outfile, n_samples, cpprob::StateType::sis);

    std::cout << cpprob::StatsPrinter{outfile} << std::endl;
}
```

This script generates a file named `posterior_is.txt` and postprocesses showing a few statistics associated
with the distribution of the variable(s) selected with the `predict` statement. The posterior distribution
is approximated here with 10.000 samples.


## Compiling a Neural Network
In the compilation step, the neural network will ask the library for traces. Hence, CPProb
will run as a server and the neural network will run as a client.

If we are going to run the neural network and the main program in the same computer
we can run

```shell
./cpprob --mode compile
python -m infcomp.compile
```

If we are going to run the neural network and the main program in different computers, we have in
both programs the option `--tcp_addr` to set up the tcp address and port in which host the server.

## Performing Inference with a Trained Neural Network
In the inference step, the library will ask the neural network for proposals.
In this case, CPProb will run as a client and the neural network will run as a server.

In this case we have to specify a file with the observed values on which we want to perform
inference with the `--observes_file` flag or we can input them with the `--observes` flag.
We always have to provide one of the two flags.

The file will have one value per line, while the if we use the `--observes` flag, the values will
be separated with spaces.

In any aggregate type, the elements will be separated with spaces.

A vector or array will be enclosed in brackets. A pair or tuple will be enclosed in parenthesis.

When using the `--observes` flag, the list of parameters has to be enclosed between quotes.

If we are to perform inference on a function with signature
```C++
void model(const std::array<std::pair<int, double>, 6> &, int a, double b);
```
we can execute the inference engine as
```shell
./cpprob --mode infer --observes "[(1 2.1) (2 3.9) (3 5.3) (4 7.7) (5 10.2) (6 12.9)] 2 3.78"
python -m infcomp.infer
```

## References
An in-depth explanation of CPProb's design can be found in:
```
@mastersthesis{lezcano2017cpprob,
    author    = "Mario Lezcano Casado",
    title     = "Compiled Inference with Probabilistic Programming for Large-Scale Scientific Simulations",
    school    = "University of Oxford",
    year      = "2017"
}
```

A large-scale application of __CPProb__ can be found in the [lezcano2017][NIPS workshop paper]:
```
@article{lezcano2017improvements,
  title={Improvements to Inference Compilation for Probabilistic Programming in Large-Scale Scientific Simulators},
  author={Lezcano Casado, Mario and Baydin, Atilim Gunes and Rubio, David Mart{\'\i}nez and Le, Tuan Anh and Wood, Frank and Heinrich, Lukas and Louppe, Gilles and Cranmer, Kyle and Ng, Karen and Bhimji, Wahid and others},
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
[lezcano2017]: https://arxiv.org/pdf/1712.07901.pdf
