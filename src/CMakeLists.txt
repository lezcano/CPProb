add_subdirectory(cpprob)
add_subdirectory(models)

find_package(Threads)

add_executable(main main.cpp)
target_link_libraries(main PUBLIC cpprob ${Boost_LIBRARIES})

add_executable(models_main models_main.cpp)
target_link_libraries(models_main PUBLIC cpprob ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

if (BUILD_SHERPA)
    target_link_libraries(main PUBLIC cpprob sherpa sherpa_mini)
    target_compile_definitions(main PRIVATE BUILD_SHERPA)
    target_link_libraries(models_main PUBLIC cpprob sherpa sherpa_mini)
else()
    target_link_libraries(main PUBLIC cpprob models)
    target_link_libraries(models_main PUBLIC cpprob models)
endif()
