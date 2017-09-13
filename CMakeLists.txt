# Many ideas taken from:
# https://github.com/ldionne/dyno/blob/master/CMakeLists.txt

cmake_minimum_required(VERSION 3.0)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(CPProb LANGUAGES CXX)


##############################################################################
# Dependancies
##############################################################################
find_package(ZeroMQ REQUIRED)
find_package(GTest REQUIRED)
find_package(FlatBuffers REQUIRED)
# https://stackoverflow.com/questions/18583800/cmake-error-the-following-boost-libraries-could-not-be-found
set(Boost_USE_STATIC_LIBS ON)

find_package(Boost REQUIRED)

include_directories(${CMAKE_BINARY_DIR}/boost/include)
#find_package(Boost COMPONENTS program_options serialization filesystem REQUIRED)

##############################################################################
# Project Options
##############################################################################
option(BUILD_SHERPA "Build Sherpa" OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # http://stackoverflow.com/questions/25365160/boostmultiprecisionfloat128-and-c11
    add_compile_options(-fext-numeric-literals)
endif()

##############################################################################
# Look for Backtrace
##############################################################################
find_package(Backtrace REQUIRED)
include_directories(${Backtrace_INCLUDE_DIR})
add_compile_options(-rdynamic)

include_directories(include)
add_subdirectory(src)

##############################################################################
# Setup testing
##############################################################################
enable_testing()
add_subdirectory(tests)
