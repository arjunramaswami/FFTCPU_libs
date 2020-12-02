// Author: Arjun Ramaswami

#ifndef HELPER_HPP
#define HELPER_HPP

void print_config(unsigned N, bool dp, unsigned nprocs, unsigned nthreads, unsigned how_many, bool inverse, unsigned iter);

bool print_results(double exec_time, double gather_time, double flops, unsigned N, unsigned nprocs, unsigned nthreads, unsigned iter);

double getTimeinMilliSec();

double moda(unsigned K, unsigned L, unsigned M);

bool checkStatus(int status);

#endif // HELPER_HPP
