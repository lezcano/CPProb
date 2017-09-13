file(GLOB SRC_FILES *.cpp)

# Setup the `cpprob` library target.
add_library(cpprob SHARED ${SRC_FILES})
add_library(CPProb::cpprob ALIAS cpprob)
target_compile_features(cpprob PUBLIC cxx_std_14)
target_include_directories(cpprob PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                          $<INSTALL_INTERFACE:include>)

target_link_libraries(cpprob PUBLIC ${ZeroMQ_LIBRARY})
target_include_directories(cpprob PUBLIC ${ZeroMQ_INCLUDE_DIR})
