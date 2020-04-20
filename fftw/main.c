/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#include "include/argparse.h"    // Cmd-line Args to set some global vars
#include "include/fft_dp_ref.h"  // Double precision
#include "include/fft_sp_ref.h"  // Single precision 

static const char *const usage[] = {
    "./host [options]",
    NULL,
};

void print_config(int N1, int N2, int N3, int iter, int inverse, int nthreads, int sp);

int main(int argc, const char **argv){

  // Cmd line argument declarations
  int N1 = 64, N2 = 64, N3 = 64;
  int iter = 1, nthreads = 1, inverse = 0, sp = 0, status;

  struct argparse_option options[] = {
    OPT_HELP(),
    OPT_GROUP("Basic Options"),
    OPT_INTEGER('m',"n1", &N1, "FFT 1st Dim Size"),
    OPT_INTEGER('n',"n2", &N2, "FFT 2nd Dim Size"),
    OPT_INTEGER('p',"n3", &N3, "FFT 3rd Dim Size"),
    OPT_BOOLEAN('s',"sp", &sp, "Single Precision"),
    OPT_INTEGER('i',"iter", &iter, "Number of iterations"),
    OPT_INTEGER('t',"threads", &nthreads, "Num Threads"),
    OPT_BOOLEAN('b',"inverse", &inverse, "Backward FFT"),
    OPT_END(),
  };

  struct argparse argparse;
  argparse_init(&argparse, options, usage, 0);
  argparse_describe(&argparse, "Computing FFT3d using FFTW", "FFT size is mandatory, default number of iterations is 1");
  argc = argparse_parse(&argparse, argc, argv);

  // Initialize
  // Set the default number of threads to be used
  omp_set_num_threads(nthreads);

  // initialize hybrid implementation
  int provided, threads_ok;
  MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided); // instead of MPI_Init
  threads_ok = provided >= MPI_THREAD_FUNNELED;

  if(!threads_ok){
    fprintf(stderr, "Cannot initialize hybrid execution.\n");
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  int world_size, myrank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size); 
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  if((N1 % world_size != 0) && (N2 % world_size !=0) && (N3 % world_size != 0)){
    if(myrank == 0){
      fprintf(stderr, "\nNumber of processes should divide the 3d FFT equally\n");
    }
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  if(sp == 1){
    status = fftwf_mpi(N1, N2, N3, nthreads, inverse, iter);
  }
  else{
    status = fftw_mpi(N1, N2, N3, nthreads, inverse, iter);
  }

  MPI_Finalize();

  if(status){
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

