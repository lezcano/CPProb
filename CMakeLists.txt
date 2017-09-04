cmake_minimum_required(VERSION 2.8.2)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

##############################################################################
# Project Name
##############################################################################
set(PROJECT_NAME cpprob)

project(${PROJECT_NAME})

##############################################################################
# Project Options
##############################################################################
option(BUILD_SHERPA "Build Sherpa" OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # http://stackoverflow.com/questions/25365160/boostmultiprecisionfloat128-and-c11
    add_compile_options(-fext-numeric-literals)
endif()

find_package(ZeroMQ)
find_package(GTest)
find_package(FlatBuffers)

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
