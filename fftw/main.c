/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

// global dependencies
#include <stdio.h>
#include <stdlib.h>

#ifdef OMP
#include <omp.h>
#ifndef __FFT_DP
#error "Only DP FFT can be multithreaded"
#endif
#endif

#include <fftw3.h>

// local dependencies
#include "common/fft_api.h"   // cmplex typedef
#include "common/argparse.h"  // Cmd-line Args to set some global vars
#include "common/helper.h"  // Cmd-line Args to set some global vars

static const char *const usage[] = {
    "./host [options]",
    NULL,
};

void main(int argc, const char **argv) {
  unsigned i = 0;
  double fftw_runtime = 0.0;
  int status;

  cmplx *fft_data;
  // Need distinct data for sp and dp FFTW for separate function calls
  fftwf_complex *fftw_sp_data;
  fftw_complex *fftw_dp_data;

  // Cmd line argument declarations
  int N[3] = {64, 64, 64};
  unsigned iter = 1, inverse = 0;
  int nthreads = 1;

  struct argparse_option options[] = {
    OPT_HELP(),
    OPT_GROUP("Basic Options"),
    OPT_INTEGER('m',"n1", &N[0], "FFT 1st Dim Size"),
    OPT_INTEGER('n',"n2", &N[1], "FFT 2nd Dim Size"),
    OPT_INTEGER('p',"n3", &N[2], "FFT 3rd Dim Size"),
    OPT_INTEGER('i',"iter", &iter, "Number of iterations"),
    OPT_BOOLEAN('b',"back", &inverse, "Backward/inverse FFT"),
    OPT_INTEGER('t',"threads", &nthreads, "Num Threads"),
    OPT_END(),
  };

  struct argparse argparse;
  argparse_init(&argparse, options, usage, 0);
  argparse_describe(&argparse, "Computing FFT3d using FFTW", "FFT size is mandatory, default number of iterations is 1");
  argc = argparse_parse(&argparse, argc, argv);

  /**********************************************
  * Print configuration chosen by user
  **********************************************/
  printf("------------------------------\n");
  printf("Configuration: \n\n");
  printf("FFT3d Size : %d %d %d\n", N[0], N[1], N[2]);
  printf("Number of Iterations %d \n", iter);
  printf("Number of Threads %d \n", nthreads);
#ifdef __FFT_SP
  printf("Single Precision Complex Floating Points\n");
#else
  printf("Double Precision Complex Floating Points\n");
#endif
  printf("------------------------------\n\n");

  /**********************************************
  * Allocate memory for input buffers
  **********************************************/
  fft_data = (cmplx *)malloc(sizeof(cmplx) * N[0] * N[1] * N[2]);
#ifdef __FFT_SP
  printf("Obtaining SP Data\n");
  // Allocate memory for fftw data
  // fftw_malloc 16-byte aligns to take advantage of SIMD instructions such as AVX, SSE
  fftw_sp_data = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * N[0] * N[1] * N[2]);

  // Fill allocated memory
  get_sp_input_data(fft_data, fftw_sp_data, N);
#else
  // Allocate memory for fftw data
  // fftw_malloc 16-byte aligns to take advantage of SIMD instructions such as AVX, SSE
  fftw_dp_data = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N[0] * N[1] * N[2]);

  // Fill allocated memory
  get_dp_input_data(fft_data, fftw_dp_data, N);
#endif

  /*********************************************************************
  * Create plan - distinct forward and backward plans for omp, sp and dp
  *********************************************************************/
#ifdef __FFT_SP
  printf("Creating in-place plan for SP FFT\n");
  fftwf_plan plan, plan_inverse;

  plan_inverse = fftwf_plan_dft_3d( N[0], N[1], N[2], &fftw_sp_data[0],
            &fftw_sp_data[0], FFTW_BACKWARD, FFTW_ESTIMATE);
  plan = fftwf_plan_dft_3d( N[0], N[1], N[2], &fftw_sp_data[0], &fftw_sp_data[0], FFTW_FORWARD, FFTW_ESTIMATE);

#elif __FFT_DP

#ifdef OMP
  printf("Configuring plan for Multithreaded FFT\n");
   status = fftw_init_threads(); 
   if(status == 0){
     printf("Something went wrong with Multithreaded FFTW! Exiting... \n");
     exit(EXIT_FAILURE);
   }
   printf("Using OMP with %d threads \n", nthreads);
   fftw_plan_with_nthreads(nthreads);
#endif

  printf("Creating in-place plan for DP FFT\n");
  fftw_plan plan, plan_inverse;

  plan_inverse = fftw_plan_dft_3d( N[0], N[1], N[2], &fftw_dp_data[0],
          &fftw_dp_data[0], FFTW_BACKWARD, FFTW_ESTIMATE);
  plan = fftw_plan_dft_3d( N[0], N[1], N[2], &fftw_dp_data[0], &fftw_dp_data[0], FFTW_FORWARD, FFTW_ESTIMATE);
#endif

  printf("Executing %d number of FFTW\n", iter);

  /**********************************************
  * Execute Plan
  **********************************************/
  double start = getTimeinMilliSec();
  // execute FFT3d iter number of times
  for( i = 0; i < iter; i++){
#ifdef __FFT_SP
    fftwf_execute(plan);
    fftwf_execute(plan_inverse);
    //fftw_runtime += compute_sp_fftw(fftw_sp_data, N, inverse);
#else
    fftw_execute(plan);
    fftw_execute(plan_inverse);
    //fftw_runtime += compute_dp_fftw(fftw_dp_data, N, inverse);
#endif
  }
  double stop = getTimeinMilliSec();

  fftw_runtime = stop - start;

  /**********************************************
  * Print performance metrics
  **********************************************/
  compute_metrics(fftw_runtime, iter, N);

  /**********************************************
  * Cleanup
  **********************************************/
  printf("\nCleaning up\n\n");
  if(fft_data)
    free(fft_data);

#ifdef OMP
    fftw_cleanup_threads();
#endif
#if !defined(OMP) && defined(__FFT_SP)
    fftwf_cleanup();
#elif !defined(OMP) && defined(__FFT_DP)
    fftw_cleanup();
#endif

#ifdef __FFT_SP
    fftwf_free(fftw_sp_data);
    fftwf_destroy_plan(plan);
    fftwf_destroy_plan(plan_inverse);
#else
    fftw_free(fftw_dp_data);
    fftw_destroy_plan(plan);
    fftw_destroy_plan(plan_inverse);
#endif
}
