file(GLOB SRC_FILES *.cpp *.hpp)

add_library(${PROJECT_LIB} ${SRC_FILES})
add_executable(alcp_main main.cpp)
target_link_libraries (alcp_main ${PROJECT_LIB})
