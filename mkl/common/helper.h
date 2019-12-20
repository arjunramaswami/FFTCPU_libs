/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

#ifndef HELPER_H
#define HELPER_H

double getTimeinMilliSec();

unsigned coord(unsigned N[3], unsigned i, unsigned j, unsigned k);

void compute_metrics(double fftw_runtime, unsigned iter, int N[3]);

#endif // HELPER_H
