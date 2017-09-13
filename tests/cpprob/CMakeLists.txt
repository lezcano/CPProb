file(GLOB SRC_FILES *.cpp)

add_executable(cpprob_tests ${SRC_FILES})
target_link_libraries(cpprob_tests cpprob gtest gtest_main)
target_compile_features(cpprob_tests PUBLIC cxx_std_14)

add_test(Tests cpprob_tests)
