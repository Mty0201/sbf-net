cmake_minimum_required(VERSION 3.14)
project(bf_edge_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Release by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# pybind11
find_package(pybind11 REQUIRED)

# Eigen3: try system, then conda fallback
find_package(Eigen3 3.3 QUIET)
if(NOT Eigen3_FOUND)
  # Fallback: search conda environments
  set(EIGEN3_SEARCH_PATHS
    "$ENV{CONDA_PREFIX}/include/eigen3"
    "$ENV{CONDA_PREFIX}/include"
  )
  find_path(EIGEN3_INCLUDE_DIR
    NAMES Eigen/Core
    PATHS ${EIGEN3_SEARCH_PATHS}
    NO_DEFAULT_PATH
  )
  if(EIGEN3_INCLUDE_DIR)
    message(STATUS "Found Eigen3 at ${EIGEN3_INCLUDE_DIR}")
  else()
    message(FATAL_ERROR "Eigen3 not found. Install via: conda install eigen")
  endif()
else()
  set(EIGEN3_INCLUDE_DIR ${EIGEN3_INCLUDE_DIRS})
  message(STATUS "Found Eigen3 via find_package at ${EIGEN3_INCLUDE_DIR}")
endif()

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
else()
  message(WARNING "OpenMP not found; single-threaded build")
endif()

# nanoflann (vendored header-only)
set(NANOFLANN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nanoflann")

# Sources
set(SOURCES
  src/common.cpp
  src/stage1.cpp
  src/stage2.cpp
  src/stage3.cpp
  src/stage4.cpp
  src/bindings.cpp
)

pybind11_add_module(bf_edge_cpp ${SOURCES})

target_include_directories(bf_edge_cpp PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${NANOFLANN_DIR}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(bf_edge_cpp PRIVATE OpenMP::OpenMP_CXX)
endif()
