cmake_minimum_required(VERSION 3.20)

include(FetchContent)

message("-- Fetching Cxxopts")
## CxxOpts - for command line argument parsing
FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts
  GIT_TAG        v2.2.1
)
FetchContent_MakeAvailable(cxxopts)