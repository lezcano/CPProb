file(GLOB SRC_FILES *.cpp)

add_executable(cpprob_tests ${SRC_FILES})
target_link_libraries(cpprob_tests PRIVATE cpprob ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(cpprob_tests PRIVATE ${GTEST_INCLUDE_DIRS})

add_test(Tests cpprob_tests)
