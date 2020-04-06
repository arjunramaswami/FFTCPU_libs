/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

#ifndef FFT_SP_REF_H
#define FFT_SP_REF_H

void fftwf_mpi(int N1, int N2, int N3, int nthreads, int inverse, int iter);

#endif // FFT_SP_REF_H