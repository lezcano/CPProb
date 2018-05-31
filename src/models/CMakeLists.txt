add_library(models SHARED models.cpp)
target_link_libraries(models PUBLIC cpprob ${Boost_LIBRARIES})
add_library(mod_prueba SHARED mod_prueba.cpp)
target_link_libraries(mod_prueba PUBLIC cpprob ${Boost_LIBRARIES})

if(BUILD_SHERPA)
    MESSAGE(STATUS "Linking to SHERPA")
    # TODO(Lezcano) Change for find_library YODA & Rivet
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -l YODA -l Rivet")

    find_library(SHERPA_LIBRARY SherpaMain HINTS /usr/local/lib/SHERPA-MC /usr/share/SHERPA-MC)
    add_library(sherpa SHARED sherpa.cpp calorimeter.cpp)

    target_link_libraries(sherpa PRIVATE cpprob ${Boost_LIBRARIES} ${SHERPA_LIBRARY})
    target_include_directories(sherpa PRIVATE /usr/local/include/SHERPA-MC)
    target_compile_options(sherpa PRIVATE -rdynamic -Wl,--no-as-needed)

    add_executable(sherpa_gen sherpa_gen.cpp)
    target_link_libraries(sherpa_gen PRIVATE cpprob sherpa)

    add_library(sherpa_mini SHARED sherpa_mini.cpp)
    target_link_libraries(sherpa_mini PRIVATE cpprob ${Boost_LIBRARIES})

    add_executable(sherpa_mini_gen sherpa_mini_gen.cpp)
    target_link_libraries(sherpa_mini_gen PRIVATE sherpa_mini cpprob ${Boost_LIBRARIES})

    add_executable(calo_variance calo_variance.cpp)
    target_link_libraries(calo_variance PRIVATE sherpa cpprob ${Boost_LIBRARIES})
endif()
