# CPProb: Scalable Probabilistic Programming [![Build Status](https://travis-ci.com/Lezcano/CPProb.svg?token=p9LTU5yGsuwiT6ypq45J&branch=master)](https://travis-ci.com/Lezcano/CPProb)

> Library to perform probabilistic inference in C++14.

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
  * __Efficiency__:   You don't pay for what you don't use
  * __Scalability__:  It should be easy to use in existing C++ models
  * __Flexibility__:  The user should be able to extend it according to her needs.

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
#include "models/model.hpp"
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
#include <iostream>
#include <string>
#include <tuple>
#include "models/model.hpp"
#include "cpprob/cpprob.hpp"
#include "cpprob/postprocess/stats_printer.hpp"

int main (int argc, char* argv[]) {
    if (argc != 2) { std::cout << "No arguments provided.\n"; return 1; }
    if (argv[1] == "compile") {
        const std::string tcp_addr = "tcp://0.0.0.0:5555";
        const int batch_size = 128;
        cpprob::compile(&models::gaussian_unknown_mean, tcp_addr, "", batch_size);
    }
    else if (argv[1] == "infer") {
        const auto observes = std::make_tuple(3., 4.);
        const auto samples = 100;
        const std::string outfile = "posterior_csis";
        const std::string tcp_addr = "tcp://127.0.0.1:6666";
        cpprob::inference(cpprob::StateType::csis, &models::gaussian_unknown_mean, observes, samples, outfile, tcp_addr);
        std::cout << cpprob::StatsPrinter{outfile} << std::endl;
    }
}
```
We are just missing a folder, to save the trained neural network, suppose that it's called `workspace`, then we can start execute the neural net for compilation and inference respectively with
```shell
docker run --rm -it -v $PWD:/workspace --net=host neuralnet python3 -m main --mode compile
docker run --rm -it -v $PWD:/workspace --net=host neuralnet python3 -m main --mode infer
```

A few side-notes on this last part. The first one is that during inference, the neural network has to be executed first, and after that __CPProb__ should be executed. Otherwise both parties end up in a deadlock state. Another thing to note is that the neural network can be executed in a GPU (or several) using `nvidia-docker`. Finally, since we have not specified on the __CPProb__ side the number of traces that we want to use for training, the way to finish the training is just by killing the neuralnet job. It is of course possible to specify the number of training examples to use, as an optional argument passed to `cpprob::compile`.

## References
An in-depth explanation of __CPProb__'s design can be found [in here](./doc/icompiled_inference.pdf):
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
[Docker]: https://www.docker.com/
"[lezcano2017]: https://arxiv.org/pdf/1712.07901.pdf
