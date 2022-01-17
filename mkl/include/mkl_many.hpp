#ifndef MKL_MANY_HPP
#define MKL_MANY_HPP

/**
 * \brief  OpenMP Multithreaded Single Precision MKL execution
 * \param  N          - Size of one dimension of FFT
 * \param  how_many   - number of batches
 * \param  nthreads   - number of threads
 * \param  inverse    - true if backward transform
 * \param  iter       - number of iterations of execution
 */
void mkl_openmp_many(unsigned N, unsigned how_many, unsigned nthreads, bool inverse, unsigned iter);

#endif // MKL_MANY_HPP