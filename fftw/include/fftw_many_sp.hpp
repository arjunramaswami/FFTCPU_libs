// Author: Arjun Ramaswami

#ifndef FFT_MANY_SP_HPP
#define FFT_MANY_SP_HPP

void fftwf_mpi_many_sp(unsigned dim, unsigned N, unsigned how_many, unsigned nthreads, bool inverse, unsigned iter);

void fftwf_openmp_many_sp(unsigned dim, unsigned N, unsigned how_many, unsigned nthreads, bool inverse, unsigned iter);

#endif // FFT_MANY_SP_HPP