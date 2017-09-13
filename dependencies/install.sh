#!/usr/bin/env bash
# From https://github.com/ldionne/dyno/blob/master/dependencies/install.sh

CMAKE_BINARY_DIR="${PWD}/build"
CMAKE_INSTALL_PREFIX="${PWD}/install"
CMAKE_SOURCE_DIR="${PWD}"

rm -rf "${CMAKE_INSTALL_PREFIX}" "${CMAKE_BINARY_DIR}"
mkdir -p "${CMAKE_INSTALL_PREFIX}" "${CMAKE_BINARY_DIR}"
(cd "${CMAKE_BINARY_DIR}" &&
  cmake "${CMAKE_SOURCE_DIR}" -DCMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}" "${@}")
cmake --build "${CMAKE_BINARY_DIR}" -- -j4

echo "************************************************************************"
echo "When setting up CPProb, you should add '${CMAKE_INSTALL_PREFIX}' to 'CMAKE_PREFIX_PATH'"
echo "************************************************************************"
