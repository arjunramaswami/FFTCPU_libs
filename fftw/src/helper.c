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
#include <float.h>
#define _USE_MATH_DEFINES

/* Compute (K*L)%M accurately */
double moda(int K, int L, int M){
    return (double)(((long long)K * L) % M);
}

// --- CODE ------------------------------------------------------------------

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
 * \brief  print time taken for fftw runs to a file
 * \param  fftw_time: milliseconds
 * \param  iter - number of iterations of each
 * \param  N - fft size
 * \param  flops - fftw_flops 
 *****************************************************************************/
void print_results(double fftw_runtime, int iter, int N[3], int nthreads, double flops){
  double avg_fftw_runtime = 0.0;

  printf("\n");
  printf("       Threads  FFTSize  AvgRuntime(ms)  Throughput(GFLOPS)  \n");

  printf("fftw:"); 
  if(fftw_runtime != 0.0){
    avg_fftw_runtime = fftw_runtime / iter;  
    //double gpoints_per_sec = ( N[0] * N[1] * N[2] / (fftw_runtime * 1E-3)) * 1E-9;
    double gflops = flops / (avg_fftw_runtime * 1E6);
    //double gflops = 3 * 5 * N[0] * N[1] * N[2]* (log((double)N[0])/log((double)2))/(avg_fftw_runtime * 1E-3 * 1E9);
    printf("%6d %8d³ %12.4lf %15.2lf \n\n", nthreads, N[0], avg_fftw_runtime, gflops);
  }
  else{
    printf("ERROR in FFT3d\n");
  }
}