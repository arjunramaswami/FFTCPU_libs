# Author: Arjun Ramaswami
cmake_minimum_required(VERSION 3.14)
project(FFTW VERSION 3.0
            DESCRIPTION "Parallel and Distributed FFTW"
            LANGUAGES CXX C)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
set(examples hybrid_many openmp_many)

# Find FFTW
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(FFTW REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# External Dependencies
include(FetchContent)
FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts
  GIT_TAG        v2.2.1
)
FetchContent_MakeAvailable(cxxopts)

set(FFTW_PLAN "FFTW_ESTIMATE" CACHE STRING "FFTW Plan")
set_property(CACHE FFTW_PLAN PROPERTY STRINGS "FFTW_ESTIMATE" "FFTW_MEASURE" "FFTW_PATIENT" "FFTW_EXHAUSTIVE")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/config.h.in"
  "${CMAKE_BINARY_DIR}/include/config.h"
  ESCAPE_QUOTES
)

foreach(example ${examples})
  
  add_executable(${example} 
    ${PROJECT_SOURCE_DIR}/examples/${example}.cpp
    ${PROJECT_SOURCE_DIR}/src/fftw_many_sp.cpp
    ${PROJECT_SOURCE_DIR}/src/helper.cpp
  )
  target_link_libraries(${example} PRIVATE ${FFTW_FLOAT_LIB})

  target_link_libraries(${example} PRIVATE MPI::MPI_CXX ${FFTW_FLOAT_MPI_LIB})

  target_link_libraries(${example} PRIVATE OpenMP::OpenMP_CXX 
      ${FFTW_FLOAT_OPENMP_LIB})

  target_link_libraries(${example} PRIVATE cxxopts)

  target_include_directories(${example} PRIVATE ${PROJECT_SOURCE_DIR}/include 
      ${CMAKE_BINARY_DIR}/include ${FFTW_INCLUDE_DIRS})

endforeach()


