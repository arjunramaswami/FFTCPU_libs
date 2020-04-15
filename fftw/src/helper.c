/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

#define _POSIX_C_SOURCE 199309L  
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>
#include <mpi.h>
#define _USE_MATH_DEFINES

/* Compute (K*L)%M accurately */
double moda(int K, int L, int M){
    return (double)(((long long)K * L) % M);
}

/**
 * \brief  compute walltime in milliseconds
 * \retval time in milliseconds
 */
double getTimeinMilliSec(){
   struct timespec a;
   clock_gettime(CLOCK_MONOTONIC, &a);
   return (double)(a.tv_nsec) * 1.0e-6 + (double)(a.tv_sec) * 1.0E3;
}

/**
 * \brief  print time taken for 3d fft execution and data transfer
 * \param  exec_time    : time in seconds to execute iter number of parallel 3d FFT
 * \param  gather_time  : time in seconds to gather results to the master node after transformation
 * \param  flops        : fftw_flops 
 * \param  N1, N2, N3   : fft size
 * \param  nprocs       : number of processes used
 * \param  nthreads     : number of threads used
 * \param  iter         : number of iterations
 * \return  0 if successful, 1 otherwise
 */
int print_results(double exec_time, double gather_time, double flops, int N1, int N2, int N3, int nprocs, int nthreads, int iter){
  double avg_fftw_runtime = 0.0, avg_transfer_time = 0.0;

  printf("\n");
  printf("       Processes  Threads  FFTSize  AvgRuntime(ms)  Throughput(GFLOPS) AvgTimetoTransfer(ms)  \n");

  printf("fftw:"); 
  if(exec_time != 0.0){
    avg_fftw_runtime = (exec_time / iter) * 1E3;  
    avg_transfer_time = (gather_time / iter) * 1E3;  
    //double gpoints_per_sec = ( N[0] * N[1] * N[2] / (fftw_runtime * 1E-3)) * 1E-9;
    double gflops = (flops / avg_fftw_runtime) * 1E-6;
    //double gflops = 3 * 5 * N[0] * N[1] * N[2]* (log((double)N[0])/log((double)2))/(avg_fftw_runtime * 1E-3 * 1E9);
    printf("%8d %8d %8d³ %12.4lf %17.4lf %19.4lf \n\n", nprocs, nthreads, N1, avg_fftw_runtime, gflops, avg_transfer_time);
  }
  else{
    fprintf(stderr, "ERROR in FFT3d\n");
    return 1;
  }
  return 0;
}

/*
 * \brief Check return value of MPI function calls
 * \param status: value returned by the call
 * \return 0 if successful, 1 on error
 */
int checkStatus(int status){
  switch (status){
    case MPI_SUCCESS:
      return 0;
    case MPI_ERR_COMM:
      fprintf(stderr, "Invalid communicator\n");
      return 1;
    case MPI_ERR_COUNT:
      fprintf(stderr, "Invalid count arg\n");
      return 1;
    case MPI_ERR_TYPE:
      fprintf(stderr, "Invalid datatype arg\n");
      return 1;
    case MPI_ERR_BUFFER:
      fprintf(stderr, "Invalid buffer pointer\n");
      return 1;
    default:
      fprintf(stderr, "Unknown Error\n");
      return 1;
  }
}

/*
 * \brief Print configuration of execution 
 * \param N1, N2, N3 : number of points in each dimension
 * \param sp         : 1 if single precision
 * \param nprocs     : number of processes used
 * \param nthreads   : number of threads used
 * \param inverse    : 1 if backward transformation
 * \param iter       : number of iterations of execution
 */
void print_config(int N1, int N2, int N3, int sp, int nprocs, int nthreads, int inverse, int iter){
  printf("\n------------------------------------------\n");
  printf("FFTW Configuration: \n");
  printf("--------------------------------------------\n");
  printf("Type               = Complex to Complex\n");
  printf("Points             = {%i, %i, %i} \n", N1, N2, N3);
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