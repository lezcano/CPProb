add_subdirectory(cpprob)
add_subdirectory(models)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE cpprob models ${Boost_LIBRARIES})

if (BUILD_SHERPA)
    target_link_libraries(main PRIVATE sherpa sherpa_mini)
    target_compile_definitions(main PRIVATE BUILD_SHERPA)
endif()
