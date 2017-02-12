file(GLOB SRC_FILES *.cpp)

#add_library(${PROJECT_LIB} ${SRC_FILES})
#set_target_properties(${PROJECT_LIB} PROPERTIES LINKER_LANGUAGE CXX)

add_executable(cpprob ${SRC_FILES})
#target_link_libraries (cpprob ${PROJECT_LIB})
