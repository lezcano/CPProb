add_subdirectory(cpprob)
add_subdirectory(models)

find_package(Threads)

add_executable(main main.cpp)
target_link_libraries(main PUBLIC cpprob ${Boost_LIBRARIES})
target_compile_features(main PUBLIC cxx_std_14)

add_executable(models_main models_main.cpp)
target_link_libraries(models_main PUBLIC cpprob ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(models_main PUBLIC cxx_std_14)

if (BUILD_SHERPA)
    target_link_libraries(main PUBLIC cpprob sherpa sherpa_mini)
    target_link_libraries(models_main PUBLIC cpprob sherpa sherpa_mini)
else()
    target_link_libraries(main PUBLIC cpprob models)
    target_link_libraries(models_main PUBLIC cpprob models)
endif()
