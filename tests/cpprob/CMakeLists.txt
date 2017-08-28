file(GLOB SRC_FILES *.cpp)

set(TESTS_NAME "${PROJECT_NAME}_tests")

add_executable(${TESTS_NAME} ${SRC_FILES})
set_property(TARGET ${TESTS_NAME} PROPERTY CXX_STANDARD 14)
target_link_libraries(${TESTS_NAME} ${PROJECT_NAME} gtest gtest_main)

add_test(Tests ${TESTS_NAME})
