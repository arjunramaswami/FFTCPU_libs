#ifndef HELPER_HPP
#define HELPER_HPP

#include <mkl.h>
/*
 * \brief Print configuration of execution 
 * \param N        : number of points in each dimension of 3D FFT
 * \param nthreads : number of threads used
 * \param how_many : number of batched FFTs executed
 * \param inverse  : true if backward transformation
 * \param iter     : number of iterations of execution
 */
void print_config(unsigned N, unsigned nthreads, unsigned how_many, bool inverse, unsigned iter);

/**
 * \brief  print time taken for 3d fft execution and data transfer
 * \param  exec_time    : average time in seconds to execute a parallel 3d FFT
 * \param  flops        : fftw_flops 
 * \param  sd           : standard deviation of runtimes
 * \param  N            : fft size
 * \param  nthreads     : number of threads used
 * \param  iter         : number of iterations
 * \return true if successful, false otherwise
 */
bool print_results(double exec_time, double flops, double sd, unsigned N, unsigned nthreads, unsigned iter, unsigned how_many);

/**
 * \brief  compute walltime in milliseconds
 * \retval time in milliseconds
 */
double getTimeinMilliSec();

bool verify_mkl(MKL_Complex8 *fft_data, MKL_Complex8 *verify_data, unsigned N, unsigned how_many);

#endif // HELPER_HPP
