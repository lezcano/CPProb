# From https://github.com/ldionne/dyno/blob/master/dependencies/CMakeLists.txt
# Copyright Louis Dionne 2017
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE.md or copy at http://boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.0)

# For ZeroMQ
find_package(Threads)
include(ExternalProject)

ExternalProject_Add(googletest
  URL            https://github.com/google/googletest/archive/master.zip
  INSTALL_DIR    "${CMAKE_INSTALL_PREFIX}"
  CMAKE_ARGS     -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
  TEST_COMMAND   ""    # Disable test step
  UPDATE_COMMAND ""    # Disable source work-tree update
)

ExternalProject_Add(flatbuffers
  URL            https://github.com/google/flatbuffers/archive/master.zip
  INSTALL_DIR    "${CMAKE_INSTALL_PREFIX}"
  CMAKE_ARGS     -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
  TEST_COMMAND   ""    # Disable test step
  UPDATE_COMMAND ""    # Disable source work-tree update
)

ExternalProject_Add(ZeroMQ
  URL            https://github.com/zeromq/libZMQ/archive/master.zip
  INSTALL_DIR    "${CMAKE_INSTALL_PREFIX}"
  UPDATE_COMMAND ""
  CMAKE_ARGS     -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                 -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
                 -DBUILD_SHARED_LIBS=OFF
                 -DBUILD_STATIC_LIBS=ON
                 -DBUILD_PACKAGING=OFF
                 -DBUILD_TESTING=OFF
                 -DBUILD_NC_TESTS=OFF
                 -BUILD_CONFIG_TESTS=OFF
                 -DINSTALL_HEADERS=ON
  TEST_COMMAND   ""    # Disable test step
  UPDATE_COMMAND ""    # Disable source work-tree update
)

ExternalProject_Add(cppzmq
  URL            https://github.com/zeromq/cppzmq/archive/master.zip
  INSTALL_DIR    "${CMAKE_INSTALL_PREFIX}"
  UPDATE_COMMAND ""
  CMAKE_ARGS     -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
  TEST_COMMAND   ""    # Disable test step
  UPDATE_COMMAND ""    # Disable source work-tree update
)
