# From https://github.com/ldionne/dyno/blob/master/dependencies/CMakeLists.txt
# Copyright Louis Dionne 2017
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE.md or copy at http://boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.8)

include(ExternalProject)

ExternalProject_Add(googletest
  GIT_REPOSITORY    https://github.com/google/googletest.git
  GIT_TAG           master
  INSTALL_DIR "${CMAKE_INSTALL_PREFIX}"
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
  SOURCE_DIR        "${CMAKE_BINARY_DIR}/googletest-src"
  BINARY_DIR        "${CMAKE_BINARY_DIR}/googletest-build"
  TEST_COMMAND ""      # Disable test step
  UPDATE_COMMAND ""    # Disable source work-tree update
)

ExternalProject_Add(flatbuffers
  GIT_REPOSITORY    https://github.com/google/flatbuffers.git
  GIT_TAG           master
  INSTALL_DIR "${CMAKE_INSTALL_PREFIX}"
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
  BUILD_COMMAND  ""    # Disable build step
  TEST_COMMAND   ""    # Disable test step
  UPDATE_COMMAND ""    # Disable source work-tree update
)
