file(GLOB SRC_FILES *.cpp)

add_library(cpprob SHARED ${SRC_FILES})
set_property(TARGET cpprob PROPERTY CXX_STANDARD 14)

target_link_libraries(cpprob PUBLIC ${ZeroMQ_LIBRARY} ${Boost_LIBRARY_DIR})
target_include_directories(cpprob PUBLIC ${ZeroMQ_INCLUDE_DIR} ${Boost_INCLUDE_DIR} ${Backtrace_INCLUDE_DIR} ${FLATBUFFERS_INCLUDE_DIR})
# For proper addresses
target_compile_options(cpprob PRIVATE -rdynamic)
