# CPProb

> Library to perform probabilistic inference in C++14.

## Disclaimer: The library is in a very early development stage! It may contain bugs

## Overview

This repository contains a proof of concept of a library to allow probabilistic programming in C++14.

## Building CPProb
In order to build ALCP you will need [CMake][], a C++14 compiler and
an updated version of the Boost libraries.

In general, you will need to specify a path to a modern C++14 complaint
compiler using the `-DCMAKE_CXX_COMPILER` variable if the default compiler
is too old.

The library also makes use of the Boost libraries for multiprecission
intgers support. In the case that boost libraries are not in the path
you will have to specify the path using the `-DBOOST_ROOT` variable.
This will not be needed in most systems.

The compile process is the same as usual, configure the project and compile
it.
```shell
mkdir build
cd build
cmake ..
cmake --build .
```

<!-- Links -->
[CMake]: http://www.cmake.org

