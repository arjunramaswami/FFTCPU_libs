/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

#ifndef HELPER_H
#define HELPER_H

double getTimeinMilliSec();

unsigned coord(unsigned N[3], unsigned i, unsigned j, unsigned k);

void compute_metrics(double mkl_fft_runtime, int iter, int N1, int N2, int N3);

#endif // HELPER_H
