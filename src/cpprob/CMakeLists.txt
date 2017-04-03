file(GLOB SRC_FILES *.cpp)

add_library(${PROJECT_NAME} STATIC ${SRC_FILES})

## add the include directory to our compile directives
target_include_directories(${PROJECT_NAME} PUBLIC ${ZeroMQ_INCLUDE_DIR})
## at the 0mq library to our link directive
target_link_libraries(${PROJECT_NAME} PUBLIC ${ZeroMQ_LIBRARY})
