# Many ideas taken from:
# https://github.com/ldionne/dyno/blob/master/CMakeLists.txt

cmake_minimum_required(VERSION 2.8.2)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(CPProb LANGUAGES CXX)

# Setup the `cpprob` library target.
add_library(cpprob SHARED)
add_library(CPProb::cpprob ALIAS cpprob)
target_compile_features(cpprob PUBLIC cxx_std_14)
target_include_directories(cpprob PRIVATE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                             $<INSTALL_INTERFACE:include>)

##############################################################################
# Dependancies
##############################################################################
find_package(ZeroMQ REQUIRED)
target_link_libraries(cpprob PUBLIC ${ZeroMQ_LIBRARY})
target_include_directories(cpprob PUBLIC ${ZeroMQ_INCLUDE_DIR})
find_package(GTest REQUIRED)
find_package(FlatBuffers REQUIRED)

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
if (Backtrace_FOUND)
    include_directories(${Backtrace_INCLUDE_DIR})
    add_compile_options(-rdynamic)
endif()

if (BUILD_SHERPA)
    add_definitions(-DBUILD_SHERPA)
    # TODO(Lezcano) Change for FindPackage YODA & Rivet
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -l YODA -l Rivet")
endif()

include_directories(include)
add_subdirectory(src)

##############################################################################
# Setup testing
##############################################################################
enable_testing()
add_subdirectory(tests)
