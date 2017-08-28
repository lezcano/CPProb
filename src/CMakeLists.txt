add_subdirectory(cpprob)
add_subdirectory(models)

add_executable(main main.cpp)

target_link_libraries(main PUBLIC ${PROJECT_NAME} ${Boost_LIBRARIES})
set_property(TARGET main PROPERTY CXX_STANDARD 14)

find_package(Threads)
add_executable(models_main models_main.cpp)
set_property(TARGET models_main PROPERTY CXX_STANDARD 14)
target_link_libraries(models_main PUBLIC ${PROJECT_NAME} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

if (BUILD_SHERPA)
    target_link_libraries(main PUBLIC ${PROJECT_NAME} sherpa sherpa_mini)
    target_link_libraries(models_main PUBLIC ${PROJECT_NAME} sherpa sherpa_mini)
else()
    target_link_libraries(main PUBLIC ${PROJECT_NAME} models)
    target_link_libraries(models_main PUBLIC ${PROJECT_NAME} models)
endif()
