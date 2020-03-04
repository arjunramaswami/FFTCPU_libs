/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

#ifndef HELPER_H
#define HELPER_H

void print_results(double fftw_runtime, int iter, int N[3], int nthreads, double flops);

double getTimeinMilliSec();

double moda(int K, int L, int M);

#endif // HELPER_H
