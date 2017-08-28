file(GLOB SRC_FILES *.cpp)

add_library(${PROJECT_NAME} SHARED ${SRC_FILES})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

target_include_directories(${PROJECT_NAME} PUBLIC ${ZeroMQ_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ${ZeroMQ_LIBRARY})
