cmake_minimum_required(VERSION 3.15)

execute_process(
   COMMAND python -m pylmcf --include
   OUTPUT_VARIABLE PYLMCF_INCLUDE_DIRS
   OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
   COMMAND python -m wnet --include
   OUTPUT_VARIABLE WNET_INCLUDE_DIRS
   OUTPUT_STRIP_TRAILING_WHITESPACE
)


project(wnetdeconv LANGUAGES CXX)
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind REQUIRED CONFIG)


set(CMAKE_CXX_STANDARD 20)

nanobind_add_module(wnetdeconv_cpp
    NB_STATIC NOMINSIZE
    src/wnetdeconv/cpp/wnetdeconv/wnetdeconv.cpp)

target_include_directories(wnetdeconv_cpp PRIVATE ${PYLMCF_INCLUDE_DIRS} ${WNET_INCLUDE_DIRS})
target_compile_definitions(wnetdeconv_cpp PRIVATE INCLUDE_NANOBIND_STUFF)

install(TARGETS wnetdeconv_cpp LIBRARY DESTINATION wnetdeconv)
