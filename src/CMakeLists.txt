file(GLOB SRC_FILES *.hpp)

add_library(${PROJECT_LIB} ${SRC_FILES})
set_target_properties(${PROJECT_LIB} PROPERTIES LINKER_LANGUAGE CXX)

add_executable(alcp_main main.cpp)
target_link_libraries (alcp_main ${PROJECT_LIB})
