file(GLOB SRC_FILES_MODELS models.cpp sherpa_mini.cpp)

add_library(models SHARED ${SRC_FILES_MODELS})
target_link_libraries (models PUBLIC ${PROJECT_NAME})

add_executable(sherpa_mini_gen sherpa_mini_gen.cpp)
target_link_libraries(sherpa_mini_gen PUBLIC ${PROJECT_NAME} ${Boost_LIBRARIES} models)

#add_executable(sherpa_gen sherpa_gen.cpp)
#target_link_libraries(sherpa_gen PUBLIC ${PROJECT_NAME} ${Boost_LIBRARIES} models)
