if(BUILD_SHERPA)
    add_definitions(-DBUILD_SHERPA)
    # TODO(Lezcano) Change for FindPackage YODA & Rivet
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -l YODA -l Rivet")

    find_library(SHERPA_LIBRARY SherpaMain HINTS /usr/local/lib/SHERPA-MC /usr/share/SHERPA-MC)
    MESSAGE(STATUS "Linking to SHERPA")
    add_library(sherpa SHARED sherpa.cpp)
    target_link_libraries(sherpa PUBLIC cpprob ${Boost_LIBRARIES} ${SHERPA_LIBRARY})
    target_include_directories(sherpa PRIVATE /usr/local/include/SHERPA-MC)
    target_compile_options(sherpa PRIVATE -rdynamic -Wl --no-as-needed)
    set_property(TARGET sherpa PROPERTY CXX_STANDARD 14)

    add_executable(sherpa_gen sherpa_gen.cpp)
    target_link_libraries(sherpa_gen PUBLIC cpprob sherpa)
    set_property(TARGET sherpa_gen PROPERTY CXX_STANDARD 14)

    add_library(sherpa_mini SHARED sherpa_mini.cpp)
    target_link_libraries(sherpa_mini PUBLIC cpprob ${Boost_LIBRARIES})
    set_property(TARGET sherpa_mini PROPERTY CXX_STANDARD 14)

    add_executable(sherpa_mini_gen sherpa_mini_gen.cpp)
    target_link_libraries(sherpa_mini_gen PUBLIC sherpa_mini cpprob ${Boost_LIBRARIES})
    set_property(TARGET sherpa_mini_gen PROPERTY CXX_STANDARD 14)
else()
    add_library(models SHARED models.cpp)
    target_link_libraries(models PUBLIC cpprob ${Boost_LIBRARIES})
    set_property(TARGET models PROPERTY CXX_STANDARD 14)
endif()
