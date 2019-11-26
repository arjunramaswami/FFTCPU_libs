/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

// global dependencies
#include <stdio.h>
#include <stdlib.h>

#ifdef OMP
#include <omp.h>
#endif

#include <fftw3.h>

// local dependencies
#include "common/fft_api.h"   // cmplex typedef
#include "common/argparse.h"  // Cmd-line Args to set some global vars
#include "common/helper.h"  // Cmd-line Args to set some global vars

static const char *const usage[] = {
    "bin/host [options]",
    NULL,
};

void main(int argc, const char **argv) {
  unsigned i = 0;
  double fftw_runtime = 0.0;
  const int inp_filename_len = 50;
  char inp_fname[inp_filename_len];
  int status;

  cmplx *fft_data;
  // Need distinct data for sp and dp FFTW for separate function calls
  fftwf_complex *fftw_sp_data;
  fftw_complex *fftw_dp_data;

  // Cmd line argument declarations
  int N[3] = {64, 64, 64};
  unsigned iter = 1, inverse = 0;

  struct argparse_option options[] = {
    OPT_HELP(),
    OPT_GROUP("Basic Options"),
    OPT_INTEGER('m',"n1", &N[0], "FFT 1st Dim Size"),
    OPT_INTEGER('n',"n2", &N[1], "FFT 2nd Dim Size"),
    OPT_INTEGER('p',"n3", &N[2], "FFT 3rd Dim Size"),
    OPT_INTEGER('i',"iter", &iter, "Number of iterations"),
    OPT_BOOLEAN('b',"back", &inverse, "Backward/inverse FFT"),
    OPT_END(),
  };

  struct argparse argparse;
  argparse_init(&argparse, options, usage, 0);
  argparse_describe(&argparse, "Computing FFT3d using FFTW", "FFT size is mandatory, default number of iterations is 1");
  argc = argparse_parse(&argparse, argc, argv);

  printf("------------------------------\n");
  printf("%s FFT3d Size : %d %d %d\n", inverse ? "Backward":"Forward", N[0], N[1], N[2]);
  printf("Number of Iterations %d \n", iter);
#ifdef __FFT_SP
  printf("Single Precision Complex Floating Points\n");
#else
  printf("Double Precision Complex Floating Points\n");
#endif
  printf("------------------------------\n\n");

  // Allocate mem for input buffer and fftw buffer
  fft_data = (cmplx *)malloc(sizeof(cmplx) * N[0] * N[1] * N[2]);

#ifdef OMP
   status =  fftw_init_threads(); 
   if(status == 0){
     printf("Something went wrong with Multithreaded FFTW! Exiting... \n");
     exit(EXIT_FAILURE);
   }
   int nthreads = omp_get_max_threads();
   printf("Using %d threads \n", nthreads);
   fftw_plan_with_nthreads(nthreads);
#endif

#ifdef __FFT_SP
  printf("Obtaining Single Precision Data\n");
  int num_bytes = snprintf(inp_fname, inp_filename_len, "inputfiles/input_f%d_%d_%d_%d.inp", N[0], N[1], N[2], iter);
  if(num_bytes > inp_filename_len){
    printf("Insufficient buffer size to store path to inputfile\n");
    exit(EXIT_FAILURE);
  }
  fftw_sp_data = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * N[0] * N[1] * N[2]);
  get_sp_input_data(fft_data, fftw_sp_data, N, inp_fname);
#else
  printf("Obtaining Double Precision Data\n");
  int num_bytes = snprintf(inp_fname, inp_filename_len, "inputfiles/input_d%d_%d_%d.inp", N[0], N[1], N[2]);
  if(num_bytes > inp_filename_len){
    printf("Insufficient buffer size to store path to inputfile\n");
    exit(EXIT_FAILURE);
  }
  fftw_dp_data = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N[0] * N[1] * N[2]);
  get_dp_input_data(fft_data, fftw_dp_data, N, inp_fname);
#endif

#ifdef __FFT_SP
  fftwf_plan plan;

  if(inverse){
    plan = fftwf_plan_dft_3d( N[0], N[1], N[2], &fftw_sp_data[0], &fftw_sp_data[0], FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  else{
    plan = fftwf_plan_dft_3d( N[0], N[1], N[2], &fftw_sp_data[0], &fftw_sp_data[0], FFTW_FORWARD, FFTW_ESTIMATE);
  }
#else
  fftw_plan plan;

  if(inverse){
    plan = fftw_plan_dft_3d( N[0], N[1], N[2], &fftw_dp_data[0], &fftw_dp_data[0], FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  else{
    plan = fftw_plan_dft_3d( N[0], N[1], N[2], &fftw_dp_data[0], &fftw_dp_data[0], FFTW_FORWARD, FFTW_ESTIMATE);
  }
#endif

  double start = getTimeinMilliSec();
  // execute FFT3d iter number of times
  for( i = 0; i < iter; i++){
#ifdef __FFT_SP
    fftwf_execute(plan);
    //fftw_runtime += compute_sp_fftw(fftw_sp_data, N, inverse);
#else
    fftw_execute(plan);
    //fftw_runtime += compute_dp_fftw(fftw_dp_data, N, inverse);
#endif
  }
  double stop = getTimeinMilliSec();

  fftw_runtime = stop - start;

  // Print performance metrics
  compute_metrics(fftw_runtime, iter, N);

  // Free the resources allocated
  printf("\nCleaning up\n\n");

  if(fft_data)
    free(fft_data);
#ifdef __FFT_SP
    fftwf_free(fftw_sp_data);
    fftwf_destroy_plan(plan);
#else
    fftw_free(fftw_dp_data);
#endif 

#ifdef OMP
    fftw_cleanup_threads();
#endif

}
