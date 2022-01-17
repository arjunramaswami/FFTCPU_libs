# Author: Arjun Ramaswami
cmake_minimum_required(VERSION 3.20)
project(MKL VERSION 1.0
            DESCRIPTION "SMP MKL"
            LANGUAGES CXX C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(examples openmp_many)

message("-- Building external dependencies")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/extDep.cmake)

message("-- Finding MPI and OpenMP")
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# separate executable for openmp_many
message("-- Building binary: openmp_many")
add_executable(openmp_many
  ${PROJECT_SOURCE_DIR}/examples/openmp_many.cpp
  ${PROJECT_SOURCE_DIR}/src/fftwf_openmp.cpp
  ${PROJECT_SOURCE_DIR}/src/helper.cpp
)

target_link_libraries(openmp_many PRIVATE 
  OpenMP::OpenMP_CXX
  ${FFTW_FLOAT_LIB}         # fftw3f
  ${FFTW_FLOAT_OPENMP_LIB}  # ffw3f_omp
  cxxopts
)

target_include_directories(openmp_many PRIVATE 
  ${PROJECT_SOURCE_DIR}/include 
  ${CMAKE_BINARY_DIR}/include 
  ${FFTW_INCLUDE_DIRS})

# separate executable for hybrid_many
message("-- Building binary: hybrid_many")
add_executable(hybrid_many
  ${PROJECT_SOURCE_DIR}/examples/hybrid_many.cpp
  ${PROJECT_SOURCE_DIR}/src/fftwf_hybrid.cpp
  ${PROJECT_SOURCE_DIR}/src/helper.cpp
)

target_link_libraries(hybrid_many PRIVATE 
  MPI::MPI_CXX 
  OpenMP::OpenMP_CXX 
  ${FFTW_FLOAT_LIB}         # fftw3f
  ${FFTW_FLOAT_MPI_LIB}     # fftw3f_mpi
  ${FFTW_FLOAT_OPENMP_LIB}  # fftw3f_omp
  cxxopts)

target_include_directories(hybrid_many PRIVATE 
  ${PROJECT_SOURCE_DIR}/include 
  ${CMAKE_BINARY_DIR}/include 
  ${FFTW_INCLUDE_DIRS})
