/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

// global dependencies
#define _POSIX_C_SOURCE 199309L  
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#define _USE_MATH_DEFINES
#include <fftw3.h>

// common dependencies
#include "fft_api.h"

// function definitions
unsigned coord(unsigned N[3], unsigned i, unsigned j, unsigned k);

// --- CODE ------------------------------------------------------------------

/******************************************************************************
 * \brief  create random single precision floating point values for FFT 
 *         computation or read existing ones if already saved in a file
 * \param  fft_data  : pointer to fft3d sized allocation of sp complex data 
 * \param  fftw_data : pointer to fft3d sized allocation of sp complex data for fftw cpu computation
 * \param  N : 3 element integer array containing the size of FFT3d  
 *****************************************************************************/
void get_sp_input_data(float2 *fft_data, fftwf_complex *fftw_data, unsigned N[3]){
  unsigned i = 0, j = 0, k = 0, where = 0;
  float a, b;

  for (i = 0; i < N[0]; i++) {
    for (j = 0; j < N[1]; j++) {
      for ( k = 0; k < N[2]; k++) {
        where = coord(N, i, j, k);

        fft_data[where].x = (float)where;
        fft_data[where].y = (float)where;

        fftw_data[where][0] = fft_data[where].x;
        fftw_data[where][1] = fft_data[where].y;
#ifdef DEBUG
        printf(" %d %d %d : fft[%d] = (%f, %f) fftw[%d] = (%f, %f) \n", i, j, k, where, fft_data[where].x, fft_data[where].y, where, fftw_data[where][0], fftw_data[where][1]);
#endif
      }
    }
  }
}
/******************************************************************************
 * \brief  create random double precision floating point values for FFT 
 *         computation or read existing ones if already saved in a file
 * \param  fft_data  : pointer to fft3d sized allocation of dp complex data
 * \param  fftw_data : pointer to fft3d sized allocation of dp complex data for fftw cpu computation
 * \param  N : 3 element integer array containing the size of FFT3d  
 *****************************************************************************/
void get_dp_input_data(double2 *fft_data, fftw_complex *fftw_data, unsigned N[3]){
  unsigned i = 0, j = 0, k = 0, where = 0;

  for (i = 0; i < N[0]; i++) {
    for (j = 0; j < N[1]; j++) {
      for ( k = 0; k < N[2]; k++) {
        where = coord(N, i, j, k);

        fft_data[where].x = (double)where;
        fft_data[where].y = (double)where;

        fftw_data[where][0] = fft_data[where].x;
        fftw_data[where][1] = fft_data[where].y;
#ifdef DEBUG          
          printf(" %d %d %d : fft[%d] = (%lf, %lf) fftw[%d] = (%lf, %lf) \n", i, j, k, where, fft_data[where].x, fft_data[where].y, where, fftw_data[where][0], fftw_data[where][1]);
#endif
      }
    }
  }
}
/******************************************************************************
 * \brief  compute walltime in milliseconds
 * \retval time in milliseconds
 *****************************************************************************/
double getTimeinMilliSec(){
   struct timespec a;
   clock_gettime(CLOCK_MONOTONIC, &a);
   return (double)(a.tv_nsec) * 1.0e-6 + (double)(a.tv_sec) * 1.0E3;
}

/******************************************************************************
 * \brief  compute the offset in the matrix based on the indices of dim given
 * \param  i, j, k : indices of different dimensions used to find the 
 *         coordinate in the matrix 
 * \param  N : fft size
 * \retval linear offset in the flattened 3d matrix
 *****************************************************************************/
unsigned coord(unsigned N[3], unsigned i, unsigned j, unsigned k) {
  // TODO : works only for uniform dims
  return i * N[0] * N[1] + j * N[2] + k;
}

/******************************************************************************
 * \brief  compute single precision fft3d using FFTW - single process CPU
 * \param  fftw_data : pointer to fft3d sized allocation of sp complex data for fftw cpu computation
 * \param  N[3] : fft size
 * \param  inverse : 1 for backward fft3d
 * \retval walltime of fftw execution measured in double precision
 *****************************************************************************/
double compute_sp_fftw(fftwf_complex *fftw_data, int N[3], int inverse){
  fftwf_plan plan;

  printf("-> Planning %sSingle precision FFTW ... \n", inverse ? "inverse ":"");
  if(inverse){
    plan = fftwf_plan_dft_3d( N[0], N[1], N[2], &fftw_data[0], &fftw_data[0], FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  else{
    plan = fftwf_plan_dft_3d( N[0], N[1], N[2], &fftw_data[0], &fftw_data[0], FFTW_FORWARD, FFTW_ESTIMATE);
  }
  printf("-> Computing Single Precision FFTW\n");
  double start = getTimeinMilliSec();
  fftwf_execute(plan);
  double stop = getTimeinMilliSec();

  fftwf_destroy_plan(plan);
  return (stop - start);
}
/******************************************************************************
 * \brief  compute double precision fft3d using FFTW - single process CPU
 * \param  fftw_data : pointer to fft3d sized allocation of dp complex data for fftw cpu computation
 * \param  N[3] : fft size
 * \param  inverse : 1 for backward fft3d
 * \retval walltime of fftw execution measured in double precision
 *****************************************************************************/
double compute_dp_fftw(fftw_complex *fftw_data, int N[3], int inverse){
  fftw_plan plan;

  printf("-> Planning %sDouble precision FFTW ... \n", inverse ? "inverse ":"");
  if(inverse){
    plan = fftw_plan_dft_3d( N[0], N[1], N[2], &fftw_data[0], &fftw_data[0], FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  else{
    plan = fftw_plan_dft_3d( N[0], N[1], N[2], &fftw_data[0], &fftw_data[0], FFTW_FORWARD, FFTW_ESTIMATE);
  }

  printf("-> Computing Double Precision FFTW\n");
  double start = getTimeinMilliSec();
  fftw_execute(plan);
  double stop = getTimeinMilliSec();

  fftw_destroy_plan(plan);
  return (stop - start);
}


/******************************************************************************
 * \brief  print time taken for fftw runs to a file
 * \param  fftw_time: double
 * \param  iter - number of iterations of each
 * \param  fname - filename given through cmd line arg
 * \param  N - fft size
 *****************************************************************************/
void compute_metrics( double fftw_runtime, unsigned iter, int N[3]){
  double avg_fftw_runtime = 0.0;

  printf("\nNumber of runs: %d\n\n", iter);
  printf("\tFFT Size\tTotal Runtime(ms)\tAvg Runtime(ms)\tThroughput(GFLOPS/sec)\t\n");

  printf("fftw:"); 
  if(fftw_runtime != 0.0){
    avg_fftw_runtime = fftw_runtime / iter;
    double gpoints_per_sec = ( N[0] * N[1] * N[2] / (fftw_runtime * 1E-3)) * 1E-9;
    double gflops = 3 * 5 * N[0] * N[1] * N[2]* (log((double)N[0])/log((double)2))/(fftw_runtime * 1E-3 * 1E9);
    printf("\t  %d³ \t\t%lf \t\t %lf \t  %.4f \t\n", N[0], fftw_runtime, avg_fftw_runtime, gflops);
  }
  else{
    printf("ERROR in FFT3d\n");
  }
}

