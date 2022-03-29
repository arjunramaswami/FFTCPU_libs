cmake_minimum_required(VERSION 3.20)
project(MKL VERSION 1.0
            DESCRIPTION "Parallel MKL"
            LANGUAGES CXX C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

set(examples openmp_many)

message("-- Building external dependencies")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/extDep.cmake)

message("-- Finding OpenMP")
find_package(OpenMP REQUIRED)

message("-- Finding MKL")
set(BLA_VENDOR Intel10_64lp)
find_package(BLAS)

# separate executable for openmp_many
message("-- Building binary: openmp_many")
add_executable(mkl_openmp_many
  ${PROJECT_SOURCE_DIR}/examples/openmp_many.cpp
  ${PROJECT_SOURCE_DIR}/src/mkl_openmp.cpp
  ${PROJECT_SOURCE_DIR}/src/helper.cpp
)

target_link_libraries(mkl_openmp_many PRIVATE 
  OpenMP::OpenMP_CXX
  ${FFTW_FLOAT_LIB}         # fftw3f
  ${FFTW_FLOAT_OPENMP_LIB}  # ffw3f_omp
  ${BLAS_LIBRARIES}        
  # libmkl_intel_lp64, libmkl_intel_thread, libmkl_core, libiomp5
  cxxopts
)

target_include_directories(mkl_openmp_many PRIVATE 
  ${PROJECT_SOURCE_DIR}/include 
  ${CMAKE_BINARY_DIR}/include 
  ${FFTW_INCLUDE_DIRS}
)