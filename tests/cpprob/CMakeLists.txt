file(GLOB SRC_FILES *.cpp)

add_executable(cpprob_tests ${SRC_FILES})
target_link_libraries(cpprob_tests cpprob ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(cpprob_tests PUBLIC ${GTEST_INCLUDE_DIRS})
set_property(TARGET cpprob_tests PROPERTY CXX_STANDARD 14)

add_test(Tests cpprob_tests)
