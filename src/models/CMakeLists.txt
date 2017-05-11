add_library(models SHARED models.cpp)
target_link_libraries (models PUBLIC ${PROJECT_NAME})

find_library(SHERPA_LIBRARY SherpaMain HINTS /usr/local/lib/SHERPA-MC /usr/share/SHERPA-MC)

add_library(sherpa SHARED sherpa.cpp)

if(BUILD_SHERPA)
    target_link_libraries(sherpa PUBLIC ${PROJECT_NAME} ${Boost_LIBRARIES} ${SHERPA_LIBRARY})
    target_compile_options(sherpa PRIVATE -I /usr/local/include/SHERPA-MC -L/usr/local/lib/SHERPA-MC  -rdynamic -Wl,--no-as-needed)

    add_executable(sherpa_gen sherpa_gen.cpp)
    target_link_libraries(sherpa_gen PUBLIC ${PROJECT_NAME} ${Boost_LIBRARIES} sherpa)
endif()

# -L/usr/local/lib/SHERPA-MC -lSherpaMain -L/usr/local/lib -L/usr/local/lib64

add_executable(sherpa_mini_gen sherpa_mini_gen.cpp)
target_link_libraries(sherpa_mini_gen PUBLIC ${PROJECT_NAME} ${Boost_LIBRARIES} models)
