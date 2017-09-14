# Many ideas taken from:
# https://github.com/ldionne/dyno/blob/master/CMakeLists.txt

cmake_minimum_required(VERSION 3.1)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(CPProb LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

##############################################################################
# Dependancies
##############################################################################
find_package(ZeroMQ REQUIRED)
find_package(GTest REQUIRED)
find_package(Backtrace REQUIRED)
find_package(FlatBuffers REQUIRED)

find_package(Boost COMPONENTS program_options filesystem REQUIRED)

##############################################################################
# Project Options
##############################################################################
option(BUILD_SHERPA "Build Sherpa" OFF)

add_compile_options(-Wall -Wextra)

include_directories(include)
add_subdirectory(src)

##############################################################################
# Setup testing
##############################################################################
enable_testing()
add_subdirectory(tests)
