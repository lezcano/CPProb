file(GLOB SRC_FILES *.cpp)

#add_library(${PROJECT_LIB} ${SRC_FILES})
#set_target_properties(${PROJECT_LIB} PROPERTIES LINKER_LANGUAGE CXX)

add_executable(${PROJECT_NAME} ${SRC_FILES})
#target_link_libraries (cpprob ${PROJECT_LIB})

## add the include directory to our compile directives
target_include_directories(${PROJECT_NAME} PUBLIC ${ZeroMQ_INCLUDE_DIR})
## at the 0mq library to our link directive
target_link_libraries(${PROJECT_NAME} PUBLIC ${ZeroMQ_LIBRARY})
