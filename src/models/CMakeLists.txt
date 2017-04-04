file(GLOB SRC_FILES *.cpp)

add_library(models STATIC ${SRC_FILES})
target_link_libraries (models PRIVATE ${PROJECT_NAME})
