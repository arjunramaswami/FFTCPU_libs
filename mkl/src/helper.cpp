#include <iostream>
#include <iomanip>
#include "helper.hpp"
#include <cmath>
#include "mkl.h"

using std::cout;
using std::cerr;
using std::endl;
using std::setw;

using std::setprecision;
using std::fixed;

/**
 * \brief  Verify single precision batched FFT3d computation using FFTW
 * \param  fft_data    : pointer to 3D number of sp points after FFTW
 * \param  verify_data : pointer to 3D number of sp points for verification
 * \param  N           : fft size
 * \param  how_many    : number of batched implementations of FFTW
 * \return true if successful, false otherwise
 */
bool verify_mkl(MKL_Complex8 *fft_data, MKL_Complex8 *verify_data, unsigned N, unsigned how_many){

  double magnitude = 0.0, noise = 0.0, mag_sum = 0.0, noise_sum = 0.0;

  for(size_t i = 0; i < how_many * N * N * N; i++){

    // FFT -> iFFT is scaled by dimensions (N*N*N)
    verify_data[i].real = verify_data[i].real * N * N * N;
    verify_data[i].imag = verify_data[i].imag * N * N * N;

    magnitude = verify_data[i].real * verify_data[i].real + \
                      verify_data[i].imag * verify_data[i].imag;
    noise = (verify_data[i].real - fft_data[i].real) \
        * (verify_data[i].real - fft_data[i].real) + 
        (verify_data[i].imag - fft_data[i].imag) * (verify_data[i].imag - fft_data[i].imag);

    mag_sum += magnitude;
    noise_sum += noise;

#ifndef NDEBUG
  cout << i << ": fft_out[" << i << "] = (" << fft_data[i].real << ", " << fft_data[i].imag << ")";
  cout << " = (" << verify_data[i].real << ", " << verify_data[i].imag << ")";
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

/*
 * \brief Print configuration of execution 
 * \param N        : number of points in each dimension of FFT3D
 * \param nthreads : number of threads used
 * \param how_many : number of batched FFTs executed
 * \param inverse  : true if backward transformation
 * \param iter     : number of iterations of execution
 */
void print_config(unsigned N, unsigned nthreads, unsigned how_many, bool inverse, unsigned iter){

  cout << "\n------------------------------------------\n";
  cout << "MKL Configuration: \n";
  cout << "--------------------------------------------\n";
  cout << "Type               = Complex to Complex\n";
  cout << "Points             = {"<< N << ", " << N << ", " << N << "}" << endl;
  cout << "Precision          = "<< "Double" << endl;
  cout << "Direction          = "<< (inverse ? "BACKWARD ":"FORWARD") << endl;
  cout << "Placement          = In Place    \n";
  cout << "Threads            = "<< nthreads << endl;
  cout << "Iterations         = "<< iter << endl;
  cout << "--------------------------------------------\n\n";
}

/**
 * \brief  print time taken for 3d fft execution and data transfer
 * \param  exec_time    : average time in milliseconds to execute a FFT3D
 * \param  flops        : throughput 
 * \param  sd           : standard deviation of runtimes
 * \param  N            : fft size
 * \param  nthreads     : number of threads used
 * \param  iter         : number of iterations
 * \return true if successful, false otherwise
 */
bool print_results(double exec_time, double flops, double sd, unsigned N,  unsigned nthreads, unsigned iter, unsigned how_many){

  if(exec_time == 0.0)
    throw "Error in Run\n";
  
  cout << "\nMeasurements\n" << "--------------------------\n";
  cout << "Threads per proc    : " << nthreads << endl;
  cout << "FFT Size            : " << N << "^3\n";
  cout << "Batch               : " << how_many << endl;
  cout << "Iterations          : " << iter << endl;
  cout << "Avg Tot Runtime     : " << fixed << (exec_time * 1e3)<< " ms\n";
  cout << "Runtime per batch   : " << ((exec_time / how_many) * 1e3) << " ms\n";
  cout << "SD                  : " << sd * 1e3 << " ms\n";
  cout << "Throughput          : " << (flops * 1e-9) << " GFLOPs\n";
  cout << "--------------------------\n";

  return true;
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