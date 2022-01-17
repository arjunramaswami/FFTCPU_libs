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
 * \brief  OpenMP Multithreaded Single precision FFTW execution
 * \param  dim        - number of dimensions of FFT (supports only 3)
 * \param  N          - Size of one dimension of FFT
 * \param  how_many   - number of batches
 * \param  nthreads   - number of threads
 * \param  inverse    - true if backward transform
 * \param  iter       - number of iterations of execution
 */
void fftwf_openmp_many(unsigned N, unsigned how_many, unsigned nthreads, bool inverse, unsigned iter, std::string wisfile);

void fftwf_openmp_many_streamappln(unsigned N, unsigned how_many, unsigned nthreads, bool inverse, unsigned iter, std::string wisfile);

void fftwf_openmp_many_conv(unsigned N, unsigned how_many, unsigned nthreads, bool inverse, unsigned iter, std::string wisfile);

void fftwf_openmp_many_nowisnoinv(unsigned N, unsigned how_many, unsigned nthreads, bool inverse, unsigned iter);

void fftwf_openmp_many_waveinp(unsigned N, unsigned how_many, unsigned nthreads, bool inverse, unsigned iter, std::string wisfile);

#endif // FFT_MANY_SP_HPP