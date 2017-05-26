add_subdirectory(cpprob)
add_subdirectory(models)

add_executable(main main.cpp)

# Link cpprob & models
if (BUILD_SHERPA)
    set(SHERPA_WRAPPER_LIB sherpa sherpa_mini)
else()
    set(MODELS_LIB models)
endif()

target_link_libraries(main PUBLIC ${PROJECT_NAME} ${Boost_LIBRARIES} ${SHERPA_WRAPPER_LIB} ${MODELS_LIB})
