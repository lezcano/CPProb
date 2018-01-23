# Many ideas taken from:
# https://github.com/ldionne/dyno/blob/master/CMakeLists.txt

cmake_minimum_required(VERSION 3.1)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

##############################################################################
# Project Options
##############################################################################
project(CPProb LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_SHERPA "Build Sherpa" OFF)

add_compile_options(-Wall -Wextra)

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # No -pedantic for g++ since it breaks the build on travis (!!)
    add_compile_options(-pedantic)
endif()

# Too pedantic with the generated code by flatbuffers and cpprob::any
# if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
#     add_compile_options(-Weverything)
#     add_compile_options(-Wno-c++98-compat)
# endif()

##############################################################################
# Dependencies
##############################################################################
find_package(ZeroMQ REQUIRED)
find_package(GTest REQUIRED)
find_package(Backtrace REQUIRED)
find_package(FlatBuffers REQUIRED)
find_package(Boost COMPONENTS program_options filesystem REQUIRED)


##############################################################################
# Setup Project
##############################################################################
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)

##############################################################################
# Setup Testing
##############################################################################
enable_testing()
add_subdirectory(tests)
