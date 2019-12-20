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
void compute_metrics( double fftw_runtime, unsigned iter, int N[3]){
  double avg_fftw_runtime = 0.0;

  printf("\nNumber of runs: %d\n\n", iter);
  printf("      FFTSize  TotalRuntime(ms)  AvgRuntime(ms)  Throughput(GFLOPS)    \n");

  printf("fftw:"); 
  if(fftw_runtime != 0.0){
    avg_fftw_runtime = fftw_runtime / (iter * 2);  // * 2 to remove inverse
    double gpoints_per_sec = ( N[0] * N[1] * N[2] / (fftw_runtime * 1E-3)) * 1E-9;
    double gflops = 3 * 5 * N[0] * N[1] * N[2]* (log((double)N[0])/log((double)2))/(avg_fftw_runtime * 1E-3 * 1E9);
    printf("%5d³       %.4f            %.4f          %.4f    \n", N[0], fftw_runtime, avg_fftw_runtime, gflops);
  }
  else{
    printf("ERROR in FFT3d\n");
  }
}

