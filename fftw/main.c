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
  int i = 0, status;
  double fftw_runtime = 0.0, start = 0.0, stop = 0.0, diff = 0.0;

  // Need distinct data for sp and dp FFTW for separate function calls
  fftwf_complex *fftw_sp_data;
  fftw_complex *fftw_dp_data;

  // Cmd line argument declarations
  int N[3] = {64, 64, 64};
  unsigned iter = 1;
  int nthreads = 1;
  int inverse = 0;
  int H1 = 1, H2 = 1, H3 = 1;

  struct argparse_option options[] = {
    OPT_HELP(),
    OPT_GROUP("Basic Options"),
    OPT_INTEGER('m',"n1", &N[0], "FFT 1st Dim Size"),
    OPT_INTEGER('n',"n2", &N[1], "FFT 2nd Dim Size"),
    OPT_INTEGER('p',"n3", &N[2], "FFT 3rd Dim Size"),
    OPT_INTEGER('i',"iter", &iter, "Number of iterations"),
    OPT_INTEGER('t',"threads", &nthreads, "Num Threads"),
    OPT_BOOLEAN('b',"inverse", &inverse, "Backward FFT"),
    OPT_END(),
  };

  struct argparse argparse;
  argparse_init(&argparse, options, usage, 0);
  argparse_describe(&argparse, "Computing FFT3d using FFTW", "FFT size is mandatory, default number of iterations is 1");
  argc = argparse_parse(&argparse, argc, argv);

  /**********************************************
  * Print configuration chosen by user
  **********************************************/
  printf("\n------------------------------\n");
  printf("FFTW Configuration: \n");
  printf("------------------------------\n");
#ifdef __FFT_SP
  printf("\n%sSINGLE PRECISION COMPLEX 3d FFT\n\n", inverse?"BACKWARD ":"FORWARD ");
#else
  printf("\n%sDOUBLE PRECISION COMPLEX 3d FFT\n\n", inverse?"BACKWARD ":"FORWARD ");
#endif
  printf("Parameters: \n");
  printf("FFT_DIMENSION      =  3\n");
  printf("FFT_LENGTHS        = {%i, %i, %i} \n", N[0], N[1], N[2]);
  printf("FFT_FORWARD_DOMAIN = DFTI_COMPLEX\n");
  printf("FFT_PLACEMENT      = DFTI_INPLACE\n");
  printf("THREADS            = %i \n", nthreads);
  printf("Iterations         = %d \n", iter);
  printf("------------------------------------\n\n");

  /**********************************************
  * Allocate memory for input buffers
  **********************************************/
#ifdef __FFT_SP
  // fftw_malloc 16-byte aligns to take advantage of SIMD instructions such as AVX, SSE
  fftw_sp_data = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * N[0] * N[1] * N[2]);
#else
  // fftw_malloc 16-byte aligns to take advantage of SIMD instructions such as AVX, SSE
  fftw_dp_data = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N[0] * N[1] * N[2]);
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
  // execute FFT3d iter number of times
  for( i = 0; i < iter; i++){
#ifdef __FFT_SP
    if(inverse){
      get_sp_input_data(fftw_sp_data, N, -H1, -H2, -H3);

      start = getTimeinMilliSec();
      fftwf_execute(plan_inverse);
      stop = getTimeinMilliSec();

      diff += stop - start;
      status = verify_sp(fftw_sp_data, N, H1, H2, H3);
      if(status == 1){
        printf("Error in transformation \n");
        exit(0);
      }
    }
    else{
      get_sp_input_data(fftw_sp_data, N, H1, H2, H3);

      start = getTimeinMilliSec();
      fftwf_execute(plan);
      stop = getTimeinMilliSec();

      diff += stop - start;
      status = verify_sp(fftw_sp_data, N, H1, H2, H3);
      if(status == 1){
        printf("Error in transformation \n");
        exit(0);
      }
    }

#else
    if(inverse){
      get_dp_input_data(fftw_dp_data, N, -H1, -H2, -H3);

      start = getTimeinMilliSec();
      fftw_execute(plan_inverse);
      stop = getTimeinMilliSec();

      diff += stop - start;
      status = verify_dp(fftw_dp_data, N, H1, H2, H3);
      if(status == 1){
        printf("Error in transformation \n");
        exit(0);
      }
    }
    else{
      get_dp_input_data(fftw_dp_data, N, H1, H2, H3);

      start = getTimeinMilliSec();
      fftw_execute(plan);
      stop = getTimeinMilliSec();

      diff += stop - start;
      printf(" Time : %lf %lf - %lf \n", start, stop, diff);
      status = verify_dp(fftw_dp_data, N, H1, H2, H3);
      if(status == 1){
        printf("Error in transformation \n");
        exit(0);
      }
    }
#endif
  }

  fftw_runtime = diff;

  /**********************************************
  * Print performance metrics
  **********************************************/
  compute_metrics(fftw_runtime, iter, N);

  /**********************************************
  * Cleanup
  **********************************************/
  printf("\nCleaning up\n\n");
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
