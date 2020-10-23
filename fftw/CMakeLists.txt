# Author: Arjun Ramaswami
cmake_minimum_required(VERSION 3.14)
project(FFTW VERSION 3.0
            DESCRIPTION "Parallel and Distributed FFTW"
            LANGUAGES CXX C)

add_executable(fftw
  ${PROJECT_SOURCE_DIR}/main.cpp
  ${PROJECT_SOURCE_DIR}/src/fftw_many_sp.cpp
  ${PROJECT_SOURCE_DIR}/src/helper.cpp
)

# Find FFTW
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/findFFTW")
find_package(FFTW REQUIRED)
target_link_libraries(fftw PRIVATE ${FFTW_DOUBLE_LIB} ${FFTW_FLOAT_LIB})
#target_link_libraries(fftw PRIVATE fftw3 fftw3f)

# Find MPI
find_package(MPI REQUIRED)
target_link_libraries(fftw PRIVATE MPI::MPI_CXX ${FFTW_FLOAT_MPI_LIB} ${FFTW_DOUBLE_MPI_LIB})

# Find OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(fftw PRIVATE OpenMP::OpenMP_CXX ${FFTW_FLOAT_OPENMP_LIB} ${FFTW_DOUBLE_OPENMP_LIB})

# External Dependencies
include(FetchContent)
FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts
  GIT_TAG        v2.2.1
)
FetchContent_MakeAvailable(cxxopts)

target_link_libraries(fftw PRIVATE cxxopts)
target_include_directories(fftw PRIVATE 
  ${PROJECT_SOURCE_DIR}/include 
  ${FFTW_INCLUDE_DIRS})
