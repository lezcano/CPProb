cmake_minimum_required(VERSION 2.8.2)

project(flatbuffers-download NONE)

include(ExternalProject)
ExternalProject_Add(flatbuffers
  GIT_REPOSITORY    https://github.com/google/flatbuffers.git
  GIT_TAG           master
  SOURCE_DIR        "/home/lezcano/Documents/CPProb/build2/flatbuffers-src"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)
