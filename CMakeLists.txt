cmake_minimum_required (VERSION 3.21)

project (MyBench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

include(cmake/hdf5.cmake)
include(cmake/highfive.cmake)
include(cmake/benchmark.cmake)
include(cmake/usearch.cmake)

add_executable(Main bench.cpp)

target_link_libraries(Main
    contrib::highfive
    contrib::benchmark
    contrib::usearch
)
target_compile_features(Main PRIVATE cxx_std_17)
