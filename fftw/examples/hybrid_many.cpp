//  Author: Arjun Ramaswami

#include <iostream>
using namespace std;

#include <mpi.h>
#include <omp.h>

#include "cxxopts.hpp" // Cmd-Line Args parser
#include "fftwf_many.hpp"  // Single precision Batch Hybrid
#include "helper.hpp"

int main(int argc, char **argv){

  // Cmd line argument declarations

  cxxopts::Options options("FFTW", "Parse FFTW input params");

  options.add_options()
      ("n, num", "Size of FFT dim", cxxopts::value<unsigned>()->default_value("64"))
      ("t, threads", "Number of threads", cxxopts::value<unsigned>()->default_value("1"))
      ("c, batch", "Number of batch", cxxopts::value<unsigned>()->default_value("1"))
      ("i, iter", "Number of iterations", cxxopts::value<unsigned>()->default_value("1"))
      ("b, inverse", "Backward FFT", cxxopts::value<bool>()->default_value("false"))
      ("w, wisdomfile", "File to wisdom", cxxopts::value<string>()->default_value("test.wisdom"))
      ("y, noverify", "No verification", cxxopts::value<bool>()->default_value("false") )
      ("h,help", "Print usage")
  ;

  auto result = options.parse(argc, argv);

  if (result.count("help")){
    cout << options.help() << endl;
    return EXIT_SUCCESS;
  }

  unsigned N = result["num"].as<unsigned>();
  unsigned nthreads = result["threads"].as<unsigned>();
  unsigned batch = result["batch"].as<unsigned>();
  unsigned iter = result["iter"].as<unsigned>();
  bool inverse = result["inverse"].as<bool>();
  string wisfile = result["wisdomfile"].as<string>();
  bool noverify = result["noverify"].as<bool>();
    
  // Initialize: set default number of threads to be used
  omp_set_num_threads(nthreads);

  // Initialize hybrid implementation
  int provided, threads_ok;
  MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided); // instead of MPI_Init
  threads_ok = provided >= MPI_THREAD_FUNNELED;

  if(!threads_ok){
    cerr << "Cannot initialize hybrid execution!" << endl;
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  int world_size, myrank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size); 
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  if(( (N * N * N) % world_size) != 0){
    if(myrank == 0){
      cerr << "Number of processes should divide the 3D FFT equally" << endl;
    }
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  try{
    fftwf_hybrid_many(N, batch, nthreads, inverse, iter, wisfile, noverify);
  }
  catch(const char* msg){
    cerr << msg << endl;
  }

  MPI_Finalize();
  
  return EXIT_SUCCESS;
}

