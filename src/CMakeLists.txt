add_subdirectory(cpprob)
add_subdirectory(models)

add_executable(main main.cpp)

target_link_libraries(main PUBLIC ${PROJECT_NAME} ${Boost_LIBRARIES})
target_compile_features(main PRIVATE cxx_std_14)

find_package(Threads)
add_executable(models_main models_main.cpp)
target_compile_features(models_main PRIVATE cxx_std_14)
target_link_libraries(models_main PUBLIC ${PROJECT_NAME} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

if (BUILD_SHERPA)
    target_link_libraries(main PUBLIC ${PROJECT_NAME} sherpa sherpa_mini)
    target_link_libraries(models_main PUBLIC ${PROJECT_NAME} sherpa sherpa_mini)
else()
    target_link_libraries(main PUBLIC ${PROJECT_NAME} models)
    target_link_libraries(models_main PUBLIC ${PROJECT_NAME} models)
endif()
