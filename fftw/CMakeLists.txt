# Author: Arjun Ramaswami
cmake_minimum_required(VERSION 3.14)
project(FFTW VERSION 3.0
            DESCRIPTION "Parallel and Distributed FFTW"
            LANGUAGES CXX C)

set(examples hybrid_many openmp_many)

# Find FFTW
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/findFFTW")
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
                              ${FFTW_INCLUDE_DIRS})
endforeach()


