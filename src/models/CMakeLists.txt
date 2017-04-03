file(GLOB SRC_FILES *.cpp)

add_executable(main ${SRC_FILES})
# Link cpprob & models
target_link_libraries(main PUBLIC ${PROJECT_NAME} ${Boost_LIBRARIES})
## add the include directory to our compile directives
target_include_directories(main PUBLIC ${ZeroMQ_INCLUDE_DIR})
## at the 0mq library to our link directive
target_link_libraries(main PUBLIC ${ZeroMQ_LIBRARY})
