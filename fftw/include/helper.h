/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

#ifndef HELPER_H
#define HELPER_H

void print_config(int N1, int N2, int N3, int sp, int nprocs, int nthreads, int inverse, int iter);

int print_results(double exec_time, double gather_time, double flops, int N1, int N2, int N3, int nprocs, int nthreads, int iter);

double getTimeinMilliSec();

double moda(int K, int L, int M);

int checkStatus(int status);

#endif // HELPER_H
