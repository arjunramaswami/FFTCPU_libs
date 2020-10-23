// Author: Arjun Ramaswami

#ifndef FFT_MANY_SP_H
#define FFT_MANY_SP_H

void fftwf_many_sp(unsigned dim, unsigned N, unsigned how_many, unsigned nthreads, bool inverse, unsigned iter);

#endif // FFT_MANY_SP_H