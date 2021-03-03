// Author: Arjun Ramaswami

#ifndef FFT_MANY_SP_HPP
#define FFT_MANY_SP_HPP

/**
 * \brief  Hybrid Single precision FFTW execution
 * \param  dim        - number of dimensions of FFT (supports only 3)
 * \param  N          - Size of one dimension of FFT
 * \param  how_many   - number of batches
 * \param  nthreads   - number of threads
 * \param  inverse    - true if backward transform
 * \param  iter       - number of iterations of execution
 */
void fftwf_mpi_many_sp(unsigned dim, unsigned N, unsigned how_many, unsigned nthreads, bool inverse, unsigned iter);

/**
 * \brief  OpenMP Multithreaded Single precision FFTW execution
 * \param  dim        - number of dimensions of FFT (supports only 3)
 * \param  N          - Size of one dimension of FFT
 * \param  how_many   - number of batches
 * \param  nthreads   - number of threads
 * \param  inverse    - true if backward transform
 * \param  iter       - number of iterations of execution
 */
void fftwf_openmp_many_sp(unsigned dim, unsigned N, unsigned how_many, unsigned nthreads, bool inverse, unsigned iter, std::string wisfile);

#endif // FFT_MANY_SP_HPP