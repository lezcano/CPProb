file(GLOB SRC_FILES *.cpp)

find_package(Threads)

add_executable(cpprob_tests ${SRC_FILES})
# I don't know why do we have to link with Boost_LIBRARIES to get rid of the boost::system::generic_category() linker error...
target_link_libraries(cpprob_tests PRIVATE cpprob ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${Boost_LIBRARIES})
target_include_directories(cpprob_tests PRIVATE ${GTEST_INCLUDE_DIRS})

add_test(Tests cpprob_tests)
