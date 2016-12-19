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

set(CPP_GLOBAL_ALL_COMPILE_OPTIONS -std=${STD_CXX} -Wall -pedantic)
set(CPP_GLOBAL_DEBUG_COMPILE_OPTIONS -O0 -g3 -Wextra -Werror -fsanitize=undefined -fsanitize=address -fsanitize=thread -fsanitize=memory)
set(CPP_GLOABL_RELEASE_COMPILE_OPTIONS -O3 -g0)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()



##############################################################################
# Setup include paths. More include paths can be added in subdirectories.
##############################################################################
find_package(Boost 1.62)
MESSAGE(Boost_FOUND)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

add_subdirectory(src)
