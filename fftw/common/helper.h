/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

#ifndef HELPER_H
#define HELPER_H

void get_sp_input_data(fftwf_complex *fftw_data, int N[3], int H1, int H2, int H3);

void get_dp_input_data(fftw_complex *fftw_data, int N[3], int H1, int H2, int H3);

double getTimeinMilliSec();

void compute_metrics(double fftw_runtime, unsigned iter, int N[3]);

int verify_dp(fftw_complex *fftw_data, int N[3], int H1, int H2, int H3);

int verify_sp(fftwf_complex *fftwf_data, int N[3], int H1, int H2, int H3);

#endif // HELPER_H
