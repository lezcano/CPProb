add_library(models SHARED models.cpp hmm.cpp gaussian_unknown_mean.cpp linear_gaussian_1d.cpp)
target_link_libraries (models PUBLIC ${PROJECT_NAME})

find_library(SHERPA_LIBRARY SherpaMain HINTS /usr/local/lib/SHERPA-MC /usr/share/SHERPA-MC)


if(BUILD_SHERPA)
    MESSAGE(STATUS "Linking to SHERPA")
    add_library(sherpa SHARED sherpa.cpp)
    target_link_libraries(sherpa PUBLIC ${PROJECT_NAME} ${Boost_LIBRARIES} ${SHERPA_LIBRARY})
    target_compile_options(sherpa PRIVATE -I /usr/local/include/SHERPA-MC -L/usr/local/lib/SHERPA-MC  -rdynamic -Wl,--no-as-needed)

    add_executable(sherpa_gen sherpa_gen.cpp)
    target_link_libraries(sherpa_gen PUBLIC ${PROJECT_NAME} ${Boost_LIBRARIES} sherpa)
else()
    # Due to the ñapa, sherpa_mini does not work when using BUILD_SHERPA
    add_library(sherpa_mini SHARED sherpa_mini.cpp)

    add_executable(sherpa_mini_gen sherpa_mini_gen.cpp)
    target_link_libraries(sherpa_mini_gen PUBLIC ${PROJECT_NAME} ${Boost_LIBRARIES} sherpa_mini)
endif()
