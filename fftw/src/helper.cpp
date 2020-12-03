//  Author: Arjun Ramaswami

#include <iostream>
#include <iomanip>
#include <mpi.h>
#include "helper.hpp"

using std::cout;
using std::cerr;
using std::endl;
using std::setw;
using std::setprecision;

/* Compute (K*L)%M accurately */
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
bool print_results(double exec_time, double gather_time, double flops, unsigned N, unsigned nprocs, unsigned nthreads, unsigned iter, unsigned how_many){

  if(exec_time == 0.0)
    throw "Error in Run\n";
  
  double avg_exec = exec_time / iter;

  cout << "\nMeasurements\n" << "--------------------------\n";
  cout << "Processes           : " << nprocs << endl;
  cout << "Threads             : " << nthreads << endl;
  cout << "FFT Size            : " << N << "^3\n";
  cout << "Batch               : " << how_many << endl;
  cout << "Iterations          : " << iter << endl;
  cout << "Avg Tot Runtime     : " << setprecision(4) << avg_exec << " ms\n";
  cout << "Runtime per batch   : " << (avg_exec / how_many) << " ms\n";
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
 * \param N1, N2, N3 : number of points in each dimension
 * \param sp         : 1 if single precision
 * \param nprocs     : number of processes used
 * \param nthreads   : number of threads used
 * \param inverse    : 1 if backward transformation
 * \param iter       : number of iterations of execution
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
  #ifdef MEASURE
  cout << "Plan               = Measure     \n";
  #elif PATIENT
  cout << "Plan               = Patient     \n";
  #elif EXHAUSTIVE
  cout << "Plan               = Exhaustive  \n";
  #else
  cout << "Plan               = Estimate    \n";
  #endif
  cout << "Threads            = "<< nthreads << endl;
  cout << "Iterations         = "<< iter << endl;
  cout << "--------------------------------------------\n\n";
}
