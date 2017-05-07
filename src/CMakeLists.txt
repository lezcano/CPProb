add_subdirectory(cpprob)
add_subdirectory(models)

add_executable(main test_main.cpp)
# Link cpprob & models
target_link_libraries(main PUBLIC ${PROJECT_NAME} ${Boost_LIBRARIES} models)
