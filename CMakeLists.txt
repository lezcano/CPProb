cmake_minimum_required(VERSION 2.8.6)

##############################################################################
# Project Name
##############################################################################
set(PROJECT_NAME CPProb)
set(PROJECT_LIB ${PROJECT_NAME}_lib)

project(${PROJECT_NAME})

##############################################################################
# Project Options
##############################################################################
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)

if(CYGWIN)
    set(STD_CXX gnu++14)
else()
    set(STD_CXX c++14)
endif()

set(CMAKE_CXX_FLAGS "-std=${STD_CXX} -Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-O2 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -O3")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=undefined,memory")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # http://stackoverflow.com/questions/25365160/boostmultiprecisionfloat128-and-c11
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fext-numeric-literals")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()


message("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")
message("CMAKE CXX COMPILER is ${CMAKE_CXX_COMPILER_ID}")


##############################################################################
# Setup include paths. More include paths can be added in subdirectories.
##############################################################################
find_package(Boost 1.62)
MESSAGE(Boost_FOUND)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

include_directories(include)
add_subdirectory(src)
