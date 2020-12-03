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

  cout << "\nMeasurements\n" << "--------------------------\n";
  cout << "Processes           : " << nprocs << endl;
  cout << "Threads             : " << nthreads << endl;
  cout << "FFT Size            : " << N << "^3\n";
  cout << "Batch               : " << how_many << endl;
  cout << "Total Runtime       : " << setprecision(4) << exec_time << " ms\n";
  cout << "Runtime per batch   : " << (exec_time / how_many) << " ms\n";
  cout << "Throughput          : " << flops * 1e-9 << " GFLOPs\n";
  cout << "Time to Transfer    : " << gather_time << "ms\n";
  cout << "--------------------------\n";

  /*
  cout << "\n       Processes  Threads  FFTSize  Batch  AvgRuntime(ms)  Total Flops TimetoTransfer(ms)  " << endl;
  cout << "fftw:";
  if(exec_time != 0.0){
    //double gpoints_per_sec = ( N[0] * N[1] * N[2] / (fftw_runtime * 1E-3)) * 1E-9;
    double gflops = (flops * 1e-9) / (exec_time * 1e-3);
    //double gflops = 3 * 5 * N[0] * N[1] * N[2]* (log((double)N[0])/log((double)2))/(avg_fftw_runtime * 1E-3 * 1E9);
    cout << setw(6) << nprocs << setw(11) << nthreads << setw(8) << N << "^3";
    cout << setw(6) << how_many;
    cout << setw(13) << exec_time << setw(15) << gflops << setw(15) << gather_time << endl;
  }
  else{
    cerr << "Error in FFT3D" << endl;
    return false;
  }
  */
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
