// Author: Arjun Ramaswami

#ifndef FFT_MANY_SP_HPP
#define FFT_MANY_SP_HPP

/**
 * \brief  Hybrid Single precision FFTW execution
 * \param  N          - Size of one dimension of FFT
 * \param  how_many   - number of batches
 * \param  nthreads   - number of threads
 * \param  inverse    - true if backward transform
 * \param  iter       - number of iterations of execution
 * \param  wisfile    - path to wisdom file
 * \param  noverify   - disable verification
 */
void fftwf_hybrid_many(unsigned N, unsigned how_many, unsigned nthreads, bool inverse, unsigned iter, std::string wisfile, bool noverify);

/**
 * \brief  OpenMP Multithreaded Single precision FFTW 3D execution
 * \param  N          - Size of one dimension of FFT
 * \param  dim        - number of dimensions
 * \param  how_many   - number of batches
 * \param  nthreads   - number of threads
 * \param  inverse    - true if backward transform
 * \param  iter       - number of iterations of execution
 * \param  wisfile    - path to wisdom file
 */
void fftwf_openmp_many(unsigned N, unsigned dim, unsigned how_many, unsigned nthreads, bool inverse, unsigned iter, std::string wisfile);

/**
 * \brief  OpenMP Multithreaded Single precision FFTW 3D execution
 *         Experiment where FFTW deals with different data on the every 
 *         iteration but computes a result at the end of every iteration 
 * \param  N          - Size of one dimension of FFT
 * \param  how_many   - number of batches
 * \param  nthreads   - number of threads
 * \param  inverse    - true if backward transform
 * \param  iter       - number of iterations of execution
 * \param  wisfile    - path to wisdom file
 */
void fftwf_openmp_many_streamappln(unsigned N, unsigned how_many, unsigned nthreads, bool inverse, unsigned iter, std::string wisfile);

/**
 * \brief  OpenMP Multithreaded Single precision FFTW 3D execution 
 *         Experiment to simulate iterative computation using
 *         3D Convolution. The data that is calculated
 *         at the end of each iteration is reused with the next
 * \param  N          - Size of one dimension of FFT
 * \param  how_many   - number of batches
 * \param  nthreads   - number of threads
 * \param  inverse    - true if backward transform
 * \param  iter       - number of iterations of execution
 * \param  wisfile    - path to wisdom file
 */
void fftwf_openmp_many_conv(unsigned N, unsigned how_many, unsigned nthreads, bool inverse, unsigned iter, std::string wisfile);

#endif // FFT_MANY_SP_HPP