//  Author: Arjun Ramaswami

#include <iostream>
using namespace std;

#include <omp.h>
#include "cxxopts.hpp"     // Cmd-Line Args parser
#include "fftwf_many.hpp"  // Single precision Batch Hybrid
#include "helper.hpp"

void print_config(int N1, int N2, int N3, int iter, int inverse, int nthreads, int sp);

int main(int argc, char **argv){

  cxxopts::Options options("FFTW", "Parse FFTW input params");

  options.add_options()
      ("n, num", "Size of FFT dim", cxxopts::value<unsigned>()->default_value("64"))
      ("d, dim", "Number of dim",cxxopts::value<unsigned>()->default_value("3"))
      ("t, threads", "Number of threads", cxxopts::value<unsigned>()->default_value("1"))
      ("c, batch", "Number of batch", cxxopts::value<unsigned>()->default_value("1"))
      ("i, iter", "Number of iterations", cxxopts::value<unsigned>()->default_value("1"))
      ("b, inverse", "Backward FFT", cxxopts::value<bool>()->default_value("false"))
      ("w, wisdomfile", "File to wisdom", cxxopts::value<string>()->default_value("test.wisdom"))
      ("e, expm", "Expm number", cxxopts::value<unsigned>()->default_value("1"))
      ("h,help", "Print usage")
  ;

  auto result = options.parse(argc, argv);

  if (result.count("help")){
    cout << options.help() << endl;
    return EXIT_SUCCESS;
  }

  unsigned N = result["num"].as<unsigned>();
  unsigned dim = result["dim"].as<unsigned>();
  unsigned nthreads = result["threads"].as<unsigned>();
  unsigned batch = result["batch"].as<unsigned>();
  unsigned iter = result["iter"].as<unsigned>();
  bool inverse = result["inverse"].as<bool>(); 
  string wisfile = result["wisdomfile"].as<string>();
  unsigned expm = result["expm"].as<unsigned>();
    
  // Initialize: set default number of threads to be used
  omp_set_num_threads(nthreads);

  try{
    switch(expm){
      case 1:{
        cout << "Expm 1: Only FFTW\n";
        fftwf_openmp_many(N, dim, batch, nthreads, inverse, iter, wisfile);
        break;
      }
      case 2:{
        cout << "Expm 2: Streaming FFTW\n";
        fftwf_openmp_many_streamappln(N, batch, nthreads, inverse, iter, wisfile);
        break;
      }
      case 3:{
        cout << "Expm 3: Conv FFTW\n";
        fftwf_openmp_many_conv(N, batch, nthreads, inverse, iter, wisfile);
        break;
      }
      default:{
        cout << "No experiment chosen\n";
        break;
      }
    }
  }
  catch(const char* msg){
    cerr << msg << endl;
  }

  return EXIT_SUCCESS;
}
