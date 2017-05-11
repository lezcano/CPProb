add_subdirectory(cpprob)
add_subdirectory(models)

add_executable(main main.cpp sherpa_details.cpp)


find_library(SHERPA_LIBRARY SherpaMain HINTS /usr/local/lib/SHERPA-MC)


# Link cpprob & models
target_link_libraries(main PUBLIC ${PROJECT_NAME} ${Boost_LIBRARIES} models ${SHERPA_LIBRARY})



target_compile_options(main PRIVATE -I /usr/local/include/SHERPA-MC -L/usr/local/lib/SHERPA-MC  -rdynamic -Wl,--no-as-needed)


# -L/usr/local/lib/SHERPA-MC -lSherpaMain -L/usr/local/lib -L/usr/local/lib64

