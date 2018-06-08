# From https://github.com/ldionne/dyno/blob/master/dependencies/CMakeLists.txt
# Copyright Louis Dionne 2017
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE.md or copy at http://boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.0)

# For ZeroMQ
find_package(Threads)
include(ExternalProject)

ExternalProject_Add(googletest
  URL            https://github.com/google/googletest/archive/release-1.8.0.zip
  INSTALL_DIR    "${CMAKE_INSTALL_PREFIX}"
  CMAKE_ARGS     -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
  TEST_COMMAND   ""    # Disable test step
  UPDATE_COMMAND ""    # Disable source work-tree update
)

ExternalProject_Add(ZeroMQ
  URL            https://github.com/zeromq/libZMQ/archive/v4.2.5.zip
  INSTALL_DIR    "${CMAKE_INSTALL_PREFIX}"
  CMAKE_ARGS     -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
  TEST_COMMAND   ""    # Disable test step
  UPDATE_COMMAND ""    # Disable source work-tree update
)

ExternalProject_Add(flatbuffers
  URL            https://github.com/google/flatbuffers/archive/v1.9.0.zip
  INSTALL_DIR    "${CMAKE_INSTALL_PREFIX}"
  CMAKE_ARGS     -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
  TEST_COMMAND   ""    # Disable test step
  UPDATE_COMMAND ""    # Disable source work-tree update
)

ExternalProject_add(
    boost
    URL http://sourceforge.net/projects/boost/files/boost/1.67.0/boost_1_67_0.tar.gz
    PREFIX boost
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ./bootstrap.sh --with-libraries=program_options,filesystem
    INSTALL_COMMAND ""
    BUILD_IN_SOURCE 1
    BUILD_COMMAND ./b2 install -j8 cxxflags=-std=c++1y --with-program_options --with-filesystem
)
