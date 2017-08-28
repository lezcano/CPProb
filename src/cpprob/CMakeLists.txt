file(GLOB SRC_FILES *.cpp)

add_library(${PROJECT_NAME} SHARED ${SRC_FILES})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
target_include_directories(${PROJECT_NAME} PUBLIC ${ZeroMQ_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ${ZeroMQ_LIBRARY})
