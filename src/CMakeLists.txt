add_subdirectory(cpprob)
add_subdirectory(models)

add_executable(main main.cpp)

# Link cpprob & models
if (BUILD_SHERPA)
    set(SHERPA_LIB sherpa)
else()
    set(SHERPA_LIB sherpa_mini)
endif()

target_link_libraries(main PUBLIC ${PROJECT_NAME} ${Boost_LIBRARIES} ${SHERPA_LIB})
