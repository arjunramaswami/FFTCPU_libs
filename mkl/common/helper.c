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

// function definitions
unsigned coord(unsigned N[3], unsigned i, unsigned j, unsigned k);

// --- CODE ------------------------------------------------------------------
/******************************************************************************
 * \brief  compute walltime in milliseconds
 * \retval time in milliseconds
 *****************************************************************************/
double getTimeinMilliSec(){
   struct timespec a;
   clock_gettime(CLOCK_MONOTONIC, &a);
   //printf("%lf %lf\n", a.tv_nsec, a.tv_sec);
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
 * \brief  print time taken for fftw runs to a file
 * \param  fftw_time: double
 * \param  iter - number of iterations of each
 * \param  fname - filename given through cmd line arg
 * \param  N - fft size
 *****************************************************************************/
void compute_metrics( double mkl_fft_runtime, int iter, int N1, int N2, int N3){
  double avg_fft_runtime = 0.0;

  printf("      FFTSize  TotalRuntime(ms)  AvgRuntime(ms)  Throughput(GFLOPS)    \n");

  printf("mkl:"); 
  if(mkl_fft_runtime != 0.0){
    avg_fft_runtime = mkl_fft_runtime / (iter);  // * 2 to remove inverse
    double gpoints_per_sec = ( N1 * N2 * N3 / (mkl_fft_runtime * 1E-3)) * 1E-9;
    double gflops = 3 * 5 * N1 * N2 * N3 * (log((double)N1)/log((double)2))/(avg_fft_runtime * 1E-3 * 1E9);
    printf("%5d³       %.4f            %.4f          %.4f    \n", N1, mkl_fft_runtime, avg_fft_runtime, gflops);
  }
  else{
    printf("ERROR in FFT3d\n");
  }
}

