//  Author: Arjun Ramaswami

#include <iostream>
using namespace std;

#include <omp.h>
#include "cxxopts.hpp"     // Cmd-Line Args parser
#include "fftwf_many.hpp"  // Single precision Batch Hybrid
#include "helper.hpp"

void print_config(int N1, int N2, int N3, int iter, int inverse, int nthreads, int sp);

int main(int argc, char **argv){

  // Cmd line argument declarations
  unsigned N = 64, iter = 1, nthreads = 1, batch = 1;
  bool inverse = false, dp = false;

  cxxopts::Options options("FFTW", "Parse FFTW input params");

  options.add_options()
      ("n, num", "Size of FFT dim", cxxopts::value<unsigned>()->default_value("64"))
      ("t, threads", "Number of threads", cxxopts::value<unsigned>()->default_value("1"))
      ("c, batch", "Number of batch", cxxopts::value<unsigned>()->default_value("1"))
      ("i, iter", "Number of iterations", cxxopts::value<unsigned>()->default_value("1"))
      ("b, inverse", "Backward FFT", cxxopts::value<bool>()->default_value("false"))
      ("h,help", "Print usage")
  ;

  auto result = options.parse(argc, argv);

  if (result.count("help")){
    cout << options.help() << endl;
    return EXIT_SUCCESS;
  }

  N = result["num"].as<unsigned>();
  nthreads = result["threads"].as<unsigned>();
  batch = result["batch"].as<unsigned>();
  iter = result["iter"].as<unsigned>();
  inverse = result["inverse"].as<bool>();
    
  // Initialize: set default number of threads to be used
  omp_set_num_threads(nthreads);

  try{
    fftwf_openmp_many(N, batch, nthreads, inverse, iter);
  }
  catch(const char* msg){
    cerr << msg << endl;
  }

  return EXIT_SUCCESS;
}
