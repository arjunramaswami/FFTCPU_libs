// Author: Arjun Ramaswami

#ifndef HELPER_HPP
#define HELPER_HPP

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
void print_config(unsigned N, bool dp, unsigned nprocs, unsigned nthreads, unsigned how_many, bool inverse, unsigned iter);

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
bool print_results(double exec_time, double gather_time, double flops, double sd, unsigned N, unsigned nprocs, unsigned nthreads, unsigned iter, unsigned how_many);

/**
 * \brief  compute walltime in milliseconds
 * \retval time in milliseconds
 */
double getTimeinMilliSec();

/**
 * \brief Compute (K*L)%M 
 * \return output of the computation
 */
double moda(unsigned K, unsigned L, unsigned M);

/*
 * \brief Check return value of MPI function calls
 * \param status: value returned by the call
 * \return true if successful, false on error
 */
bool checkStatus(int status);

#endif // HELPER_HPP
