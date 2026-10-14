# ag: the tensor engine. no third-party code.
# the Makefile next to this file is the primary build (tests, wasm, bench);
# this exists for IDEs and for anyone who prefers CMake:
#   cmake -B build && cmake --build build && ctest --test-dir build

cmake_minimum_required(VERSION 3.20)
project(ag CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT MSVC)
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(ag_core STATIC
    src/core.cpp
    src/rng.cpp
    src/graph.cpp
    src/ops.cpp
    src/program.cpp
    src/exec.cpp
    src/optim.cpp
    src/data.cpp
    src/scalarize.cpp
    src/session.cpp
    src/capi.cpp
)
target_include_directories(ag_core PUBLIC include)

enable_testing()
foreach(t skeleton engine)
    add_executable(ag_test_${t} tests/test_${t}.cpp)
    target_link_libraries(ag_test_${t} PRIVATE ag_core)
    add_test(NAME ag_${t} COMMAND ag_test_${t})
endforeach()

add_executable(ag_bench bench/bench.cpp)
target_link_libraries(ag_bench PRIVATE ag_core)
