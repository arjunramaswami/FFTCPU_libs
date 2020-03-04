/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "include/argparse.h"    // Cmd-line Args to set some global vars
#include "include/fft_dp_ref.h"  // Double precision
#include "include/fft_sp_ref.h"  // Single precision 

static const char *const usage[] = {
    "./host [options]",
    NULL,
};

void print_config(int N[3], int iter, int inverse, int nthreads, int sp);

int main(int argc, const char **argv){

  // Cmd line argument declarations
  int N[3] = {64, 64, 64};
  int iter = 1, nthreads = 1, inverse = 0, sp = 0;

  struct argparse_option options[] = {
    OPT_HELP(),
    OPT_GROUP("Basic Options"),
    OPT_INTEGER('m',"n1", &N[0], "FFT 1st Dim Size"),
    OPT_INTEGER('n',"n2", &N[1], "FFT 2nd Dim Size"),
    OPT_INTEGER('p',"n3", &N[2], "FFT 3rd Dim Size"),
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

  // Print to console the configuration chosen to execute during runtime
  print_config(N, iter, inverse, nthreads, sp);

  // Set the default number of threads to be used
  omp_set_num_threads(nthreads);

  if(sp == 1){
    fftw_sp(N, nthreads, inverse, iter);
  }
  else{
    fftw_dp(N, nthreads, inverse, iter);
  }

  return 0;
}



void print_config(int N[3], int iter, int inverse, int nthreads, int sp){
  printf("\n------------------------------------------\n");
  printf("FFTW Configuration: \n");
  printf("--------------------------------------------\n");
  printf("Type               = Complex to Complex\n");
  printf("Points             = {%i, %i, %i} \n", N[0], N[1], N[2]);
  printf("Precision          = %s \n",  sp==1 ? "Single": "Double");
  printf("Direction          = %s \n", inverse ? "BACKWARD ":"FORWARD");
  printf("Placement          = In Place    \n");
  #ifdef MEASURE
  printf("Plan               = Measure     \n");
  #elif PATIENT
  printf("Plan               = Patient     \n");
  #elif EXHAUSTIVE
  printf("Plan               = Exhaustive  \n");
  #else
  printf("Plan               = Estimate    \n");
  #endif
  printf("Threads            = %i \n", nthreads);
  printf("Iterations         = %d \n", iter);
  printf("--------------------------------------------\n\n");
}