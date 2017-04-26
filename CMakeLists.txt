cmake_minimum_required(VERSION 2.8.2)

##############################################################################
# Project Name
##############################################################################
set(PROJECT_NAME cpprob)

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

# -isystem
# http://stackoverflow.com/questions/25320100/suppress-warnings-from-boost-includes
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    #set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=undefined,memory")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # http://stackoverflow.com/questions/25365160/boostmultiprecisionfloat128-and-c11
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fext-numeric-literals")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

if (CMAKE_VERSION VERSION_LESS 3.2)
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "")
else()
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "UPDATE_DISCONNECTED 1")
endif()

##############################################################################
# Download and configure ZeroMQ
##############################################################################

find_package(ZeroMQ)

if(NOT ZeroMQ_FOUND)
    # Download and unpack googletest at configure time
    configure_file(cmake/ZeroMQ.cmake.in zeromq-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/zeromq-download )
    if(result)
      message(FATAL_ERROR "CMake step for zeromq failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/zeromq-download )
    if(result)
      message(FATAL_ERROR "Build step for zeromq failed: ${result}")
    endif()

    # Prevent overriding the parent project's compiler/linker
    # settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)


    set(ZeroMQ_INSTALL ${CMAKE_BINARY_DIR}/ZeroMQ-install)
    set(ZeroMQ_FOUND TRUE)
    set(ZeroMQ_INCLUDE_DIR ${ZeroMQ_INSTALL}/include)

    if(MSVC)
        FILE(GLOB_RECURSE ZeroMQ_LIBRARY "${ZeroMQ_INSTALL}/lib/libzmq-${CMAKE_VS_PLATFORM_TOOLSET}*.lib")
    else()
        FILE(GLOB_RECURSE ZeroMQ_LIBRARY "${ZeroMQ_INSTALL}/lib/libzmq-*.a")
    endif()
    set(ZeroMQ_LIBRARY_DIR ${ZeroMQ_INSTALL}/lib)
    set(ZeroMQ_EXTERNAL TRUE)
endif()

##############################################################################
# Download and configure GTest
##############################################################################

find_package(GTest)

if(NOT GTEST_FOUND)
    # Download and unpack googletest at configure time
    configure_file(cmake/gtest.cmake.in googletest-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
    if(result)
      message(FATAL_ERROR "CMake step for googletest failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
    if(result)
      message(FATAL_ERROR "Build step for googletest failed: ${result}")
    endif()

    # Prevent overriding the parent project's compiler/linker
    # settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    # Add googletest directly to our build. This defines
    # the gtest and gtest_main targets.
    add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
                     ${CMAKE_BINARY_DIR}/googletest-build)

    # The gtest/gtest_main targets carry header search path
    # dependencies automatically when using CMake 2.8.11 or
    # later. Otherwise we have to add them here ourselves.
    if (CMAKE_VERSION VERSION_LESS 2.8.11)
      include_directories("${gtest_SOURCE_DIR}/include")
    endif()
endif()


##############################################################################
# Setup include paths. More include paths can be added in subdirectories.
##############################################################################
find_package(Boost 1.62 COMPONENTS program_options REQUIRED)
MESSAGE(Boost_FOUND)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

include_directories(include)
add_subdirectory(src)

##############################################################################
# Setup testing
##############################################################################
enable_testing()
add_subdirectory(tests)
