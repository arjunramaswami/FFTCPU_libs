//  Author: Arjun Ramaswami

#include <iostream>
#include <iomanip>
#include <mpi.h>
#include "config.h"
#include "helper.hpp"
#include <fftw3.h>

using std::cout;
using std::cerr;
using std::endl;
using std::setw;

using std::setprecision;
using std::fixed;

/* Compute (K*L)%M */
double moda(unsigned K, unsigned L, unsigned M){
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
 * \param  exec_time    : average time in seconds to execute a parallel 3d FFT
 * \param  gather_time  : average time in seconds to gather results to the master node after transformation
 * \param  flops        : fftw_flops 
 * \param  N1, N2, N3   : fft size
 * \param  nprocs       : number of processes used
 * \param  nthreads     : number of threads used
 * \param  iter         : number of iterations
 * \return true if successful, false otherwise
 */
bool print_results(double exec_time, double gather_time, double flops, double sd, unsigned N, unsigned nprocs, unsigned nthreads, unsigned iter, unsigned how_many){

  if(exec_time == 0.0)
    throw "Error in Run\n";
  
  cout << "\nMeasurements\n" << "--------------------------\n";
  cout << "Processes           : " << nprocs << endl;
  cout << "Threads per proc    : " << nthreads << endl;
  cout << "FFT Size            : " << N << "^3\n";
  cout << "Batch               : " << how_many << endl;
  cout << "Iterations          : " << iter << endl;
  cout << "Avg Tot Runtime     : " << fixed << (exec_time * 1e3)<< " ms\n";
  cout << "Runtime per batch   : " << ((exec_time / how_many) * 1e3) << " ms\n";
  cout << "SD                  : " << sd * 1e3 << " ms\n";
  cout << "Throughput          : " << (flops * 1e-9) << " GFLOPs\n";
  cout << "Time to Transfer    : " << gather_time << "ms\n";
  cout << "--------------------------\n";

  return true;
}

/*
 * \brief Check return value of MPI function calls
 * \param status: value returned by the call
 * \return true if successful, false on error
 */
bool checkStatus(int status){
  switch (status){
    case MPI_SUCCESS:
      return true;
    case MPI_ERR_COMM:
      fprintf(stderr, "Invalid communicator\n");
      return false;
    case MPI_ERR_COUNT:
      fprintf(stderr, "Invalid count arg\n");
      return false;
    case MPI_ERR_TYPE:
      fprintf(stderr, "Invalid datatype arg\n");
      return false;
    case MPI_ERR_BUFFER:
      fprintf(stderr, "Invalid buffer pointer\n");
      return false;
    default:
      fprintf(stderr, "Unknown Error\n");
      return false;
  }
}

/*
 * \brief Print configuration of execution 
 * \param N .      : number of points in each dimension of 3D FFT
 * \param dp       : true if double precision (not supported)
 * \param nprocs   : number of processes used
 * \param nthreads : number of threads used
 * \param how_many : number of batched FFTs executed
 * \param inverse  : true if backward transformation
 * \param iter     : number of iterations of execution
 */
void print_config(unsigned N, bool dp, unsigned nprocs, unsigned nthreads, unsigned how_many, bool inverse, unsigned iter){

  cout << "\n------------------------------------------\n";
  cout << "FFTW Configuration: \n";
  cout << "--------------------------------------------\n";
  cout << "Type               = Complex to Complex\n";
  cout << "Points             = {"<< N << ", " << N << ", " << N << "}" << endl;
  cout << "Precision          = "<< (dp ? "Double":"Single") << endl;
  cout << "Direction          = "<< (inverse ? "BACKWARD ":"FORWARD") << endl;
  cout << "Placement          = In Place    \n";
  unsigned fftw_plan = FFTW_PLAN;
  switch(fftw_plan){
    case FFTW_MEASURE:  
      cout << "Plan               = Measure     \n";                        break;
    case FFTW_ESTIMATE: 
      cout << "Plan               = Estimate  \n"; 
      break;
    case FFTW_PATIENT: 
      cout << "Plan               = Patient     \n";
      break;
    case FFTW_EXHAUSTIVE: 
      cout << "Plan               = Exhaustive  \n";
      break;
    default: 
      throw "Incorrect plan\n";
      break;    
  }
  cout << "Threads            = "<< nthreads << endl;
  cout << "Iterations         = "<< iter << endl;
  cout << "--------------------------------------------\n\n";
}
