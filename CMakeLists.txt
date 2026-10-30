cmake_minimum_required(VERSION 3.20)
project(sage_sr VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(FetchContent)

option(SAGE_SR_ENABLE_OPENMP "Enable OpenMP expression evaluation" OFF)
option(SAGE_SR_ENABLE_TIMING "Enable backend timing counters" OFF)

find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(eigen)
endif()
if(NOT TARGET Eigen3::Eigen AND TARGET eigen)
    add_library(Eigen3::Eigen ALIAS eigen)
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 2.10 CONFIG QUIET)
if(NOT pybind11_FOUND)
    FetchContent_Declare(pybind11
        URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(pybind11)
endif()

add_library(sage_sr_core STATIC
    cpp/src/core/exp_tree.cpp
    cpp/src/core/exp_queue.cpp
    cpp/src/core/bridge.cpp
    cpp/src/core/tree.cpp
    cpp/src/eval/interpreter.cpp
    cpp/src/eval/optimizer.cpp
    cpp/src/eval/timing.cpp
    cpp/src/evaluator/evaluator.cpp
    cpp/src/gp/gp_manager.cpp
    cpp/src/mcts/mcts.cpp
    cpp/src/regressor.cpp)
set_property(TARGET sage_sr_core PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(sage_sr_core PUBLIC cpp/include)
target_link_libraries(sage_sr_core PUBLIC Eigen3::Eigen)
if(SAGE_SR_ENABLE_TIMING)
    target_compile_definitions(sage_sr_core PUBLIC SAGE_SR_ENABLE_TIMING=1)
endif()
if(SAGE_SR_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(sage_sr_core PUBLIC OpenMP::OpenMP_CXX)
endif()

pybind11_add_module(_core cpp/bindings.cpp)
target_link_libraries(_core PRIVATE sage_sr_core)
install(TARGETS _core LIBRARY DESTINATION sage_sr RUNTIME DESTINATION sage_sr)
