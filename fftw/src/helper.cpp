//  Author: Arjun Ramaswami

#include <iostream>
#include <iomanip>
#include <mpi.h>
#include "config.h"
#include "helper.hpp"
#include <fftw3.h>
#include <cmath>

using std::cout;
using std::cerr;
using std::endl;
using std::setw;

using std::setprecision;
using std::fixed;

/**
 * \brief  Verify single precision batched FFT3d computation using FFTW
 * \param  fftw_data   : pointer to 3D number of sp points after FFTW
 * \param  verify_data : pointer to 3D number of sp points for verification
 * \param  N1, N2, N3  : fft size
 * \param  H1, H2, H3  : harmonic to modify frequency of discrete time signal
 * \param  how_many    : number of batched implementations of FFTW
 * \return true if successful, false otherwise
 */
bool verify_fftw(fftwf_complex *fftw_data, fftwf_complex *verify_data, unsigned N, unsigned how_many){

  double magnitude = 0.0, noise = 0.0, mag_sum = 0.0, noise_sum = 0.0;

  for(size_t i = 0; i < how_many * N * N * N; i++){

    // FFT -> iFFT is scaled by dimensions (N*N*N)
    verify_data[i][0] = verify_data[i][0] * N * N * N;
    verify_data[i][1] = verify_data[i][1] * N * N * N;

    magnitude = verify_data[i][0] * verify_data[i][0] + \
                      verify_data[i][1] * verify_data[i][1];
    noise = (verify_data[i][0] - fftw_data[i][0]) \
        * (verify_data[i][0] - fftw_data[i][0]) + 
        (verify_data[i][1] - fftw_data[i][1]) * (verify_data[i][1] - fftw_data[i][1]);

    mag_sum += magnitude;
    noise_sum += noise;

#ifndef NDEBUG
    cout << i << ": fftw_out[" << i << "] = (" << fftw_data[i][0] << ", " << fftw_data[i][1] << ")";
    cout << " = (" << verify_data[i][0] << ", " << verify_data[i][1] << ")";
    cout << endl;
#endif
  }

  float db = 10 * log(mag_sum / noise_sum) / log(10.0);

    // if SNR greater than 120, verification passes
  if(db > 120){
    return true;
  }
  else{ 
    cout << "Signal to noise ratio on output sample: " << db << " --> FAILED \n\n";
    return false;
  }
}

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
      cout << "Plan               = Patient    \n";
      break;
    case FFTW_EXHAUSTIVE: 
      cout << "Plan               = Exhaustive  \n";
      break;
    case FFTW_WISDOM_ONLY:
      cout << "Plan               = Wisdom Only \n";
      break;
    default: 
      throw "Incorrect plan\n";
      break;    
  }
  cout << "Threads            = "<< nthreads << endl;
  cout << "Iterations         = "<< iter << endl;
  cout << "--------------------------------------------\n\n";
}
