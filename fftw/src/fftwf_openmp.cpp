// Author: Arjun Ramaswami
#include <iostream>
#include <cmath>
#include <omp.h>
#include <fftw3.h>
#include <mkl.h>
#include "config.h"
#include <algorithm> // nth_element used in calculating median, quartiles
#include <vector>

#include "cxxopts.hpp" // Cmd-Line Args parser
#include "fftwf_many.hpp"
#include "helper.hpp"
using namespace std;

static fftwf_plan plan, plan_verify;

static void cleanup_openmp(fftwf_complex *fftw_data, fftwf_complex *verify_data){

  // Cleanup : fftw data, plans and threads
  fftwf_free(fftw_data);
  fftwf_free(verify_data);
  fftwf_destroy_plan(plan);
  fftwf_cleanup_threads();
}

/**
 * \brief create single precision floating points values for FFT computation for each process level block
 * \param fftw_data   : pointer to 3d number of sp points for FFTW
 * \param verify_data : pointer to 3d number of sp points for verification
 * \param N           : number of points in each dimension
 * \param how_many    : number of batched implementations of FFTW
 */
void get_data(fftwf_complex *fftw_data, fftwf_complex *verify_data, size_t N, unsigned how_many){

  float re_val = 0.0f, img_val = 0.0f;

  for(unsigned i = 0; i < how_many * N * N * N; i++){
    re_val = ((float) rand() / (RAND_MAX));
    img_val = ((float) rand() / (RAND_MAX));

    verify_data[i][0] = fftw_data[i][0] = re_val;
    verify_data[i][1] = fftw_data[i][1] = img_val;

#ifndef NDEBUG          
  printf("fftw[%u]: (%f, %f)\n", i, fftw_data[i][0], fftw_data[i][1]);
#endif
  }
}

/**
 * \brief create single precision floating points values for FFT computation for each process level block
 * \param fftw_data   : pointer to 3d number of sp points for FFTW
 * \param verify_data : pointer to 3d number of sp points for verification
 * \param N           : number of points in each dimension
 * \param H1, H2, H3  : harmonic to modify frequency of discrete time signal
 * \param how_many    : number of batched implementations of FFTW
 */
void get_data_wave(fftwf_complex *fftw_data, fftwf_complex *verify_data,size_t N, unsigned how_many){

  unsigned index; 
  float TWOPI = 6.2831853071795864769;
  float phase, phase1, phase2, phase3;
  unsigned H1 = 1, H2 = 1, H3 = 1;
  double re_val = 0.0, img_val = 0.0;
  unsigned S1 = N*N, S2 = N, S3 = 1;

  for(size_t many = 0; many < how_many; many++){
    for(size_t i = 0; i < N; i++) {
      for(size_t j = 0; j < N; j++) {
        for(size_t k = 0; k < N; k++) {
          phase1 = moda(i, H1, N) / N;
          phase2 = moda(j, H2, N) / N;
          phase3 = moda(k, H3, N) / N;
          phase = phase1 + phase2 + phase3;

          index = (many * S1 * S2) + (i * S1) + (j * S2) + k;

          re_val = cosf( TWOPI * phase ) / (N * N * N);
          img_val = sinf( TWOPI * phase ) / (N * N * N);

          verify_data[index][0] = fftw_data[index][0] = re_val;
          verify_data[index][1] = fftw_data[index][1] = img_val;
        }
      }
    }
  }
}

/**
 * \brief  OpenMP Multithreaded Single precision FFTW 3D execution
 * \param  N          - Size of one dimension of FFT
 * \param  how_many   - number of batches
 * \param  nthreads   - number of threads
 * \param  inverse    - true if backward transform
 * \param  iter       - number of iterations of execution
 * \param  wisfile    - path to wisdom file
 */
void fftwf_openmp_many(unsigned N, unsigned how_many, unsigned nthreads, bool inverse, unsigned iter, std::string wisfile){
    
  if ( (how_many == 0) || (nthreads == 0) || (iter == 0) )
    throw "Invalid value, should be >=1!";

  // Initialising Threads
  int threads_ok = fftwf_init_threads(); 
  if(threads_ok == 0)
    throw "Something went wrong with Multithreaded FFTW! Exiting... \n";

  // All subsequent plans will now use nthreads
  fftwf_plan_with_nthreads((int)nthreads);

  // Allocating input and verification arrays
  size_t data_sz = how_many * N * N * N;
  fftwf_complex *fftw_data = fftwf_alloc_complex(data_sz);
  fftwf_complex *verify_data = fftwf_alloc_complex(data_sz);

  // Setting direction of FFT for the plan
  int direction = FFTW_FORWARD;
  int direction_inv = FFTW_BACKWARD;
  if(inverse){
    direction = FFTW_BACKWARD;
    direction_inv = FFTW_FORWARD;
  }

  // Parameters for planning
  const int n[3] = {(int)N, (int)N, (int)N};
  int idist = N * N * N, odist = N * N * N;
  int istride = 1, ostride = 1;
  const int *inembed = n, *onembed = n;
  const unsigned fftw_plan = FFTW_PLAN;
  switch(fftw_plan){
    case FFTW_MEASURE:  cout << "FFTW Plan: Measure\n";
                        break;
    case FFTW_ESTIMATE: cout << "FFTW Plan: Estimate\n";
                        break;
    case FFTW_PATIENT:  cout << "FFTW Plan: Patient\n";
                        break;
    case FFTW_EXHAUSTIVE: cout << "FFTW Plan: Exhaustive\n";
                        break;
    case FFTW_WISDOM_ONLY: cout << "FFTW Plan: Wisdom Only \n";
                        break;
    default: throw "Incorrect plan\n";
            break;
  }

  plan_verify = fftwf_plan_many_dft(3, n, how_many, fftw_data, inembed, istride, idist, fftw_data, onembed, ostride, odist, direction_inv, FFTW_ESTIMATE);
  
  // Import wisdom from filename
  int wis_status = 0;
  fftwf_forget_wisdom();

  if(fftw_plan != FFTW_ESTIMATE){
    wis_status = fftwf_import_wisdom_from_filename(wisfile.c_str());
    if(wis_status == 0) // could not import wisdom
      cout << "-- Cannot import wisdom from " << wisfile << endl;
    else if((wis_status == 0) && (fftw_plan == FFTW_WISDOM_ONLY)){
      cleanup_openmp(fftw_data, verify_data);
      throw "Plan should use imported wisdom. Cannot import. Quitting\n";
    }
    else                 
      cout << "-- Importing wisdom from " << wisfile << endl;
  }
  else
    cout << "Estimate Plan: Not using wisdom" << endl;

  // Make Plan
  double plan_start = getTimeinMilliSec();
  plan = fftwf_plan_many_dft(3, n, how_many, fftw_data, inembed, istride, idist, fftw_data, onembed, ostride, odist, direction, fftw_plan);
  double plan_time = getTimeinMilliSec() - plan_start;
  cout << "Planning Completed\n";

  if(wis_status == 0 && (fftw_plan != FFTW_WISDOM_ONLY) && (fftw_plan != FFTW_ESTIMATE)){
    // i.e., wisdom is not imported
    int exp_stat = fftwf_export_wisdom_to_filename(wisfile.c_str()); 
    if(exp_stat == 0)
      cout << "-- Could not export wisdom file to " << wisfile.c_str() << endl;
    else
      cout << "-- Exporting wisdom file to " << wisfile.c_str() << endl;
  }
  else
    cout << "Not exporting any wisdom\n";

  /* every iteration: FFT followed by inverse for verification */
  double start = 0.0, stop = 0.0, exec_diff = 0.0;
  vector<double> exec_t;
#ifndef NDEBUG
  cout << "Iteration: ";
#endif
  double test_res = 0.0;
  for(size_t it = 0; it < iter; it++){
#ifndef NDEBUG
    cout << it << ", ";
#endif
    // Get new data every iteration on the same allocation 
    get_data(fftw_data, verify_data, N, how_many);

    start = getTimeinMilliSec();
    fftwf_execute(plan);
    stop = getTimeinMilliSec();

    fftwf_execute(plan_verify);

    bool status = verify_fftw(fftw_data, verify_data, N, how_many);
    if(!status){
      cleanup_openmp(fftw_data, verify_data);
      throw "Error in Transformation\n";
    }

    double diff = stop - start;
    exec_t.push_back(diff);
    exec_diff += diff;
  }
  cout << endl;

#ifndef NDEBUG
  cout << "Printing individual runtimes:\n";
  for(unsigned i = 0; i < iter; i++)
    printf(" %u: %lfms\n", i, exec_t[i]);
  cout << endl;
#endif

  double Q1_val = 0.0, median = 0.0, Q3_val = 0.0;
  if(iter > 2){
    // using rounding up algorithm to simplify
    const unsigned Q1 = ceil(exec_t.size() / 4.0);
    const unsigned Q2 = exec_t.size() / 2;
    const unsigned Q3 = ceil(3 * exec_t.size() / 4.0);

    cout << "\n-- Real : Q1: " << Q1 << " Q2: " << Q2 << " Q3: " << Q3 << endl;
    cout << "-- Array: Q1: " << Q1-1 <<" Q2: "<<Q2-1<< " Q3: "<< Q3-1 << endl;

    nth_element(exec_t.begin(),          exec_t.begin() + Q1, exec_t.end());
    nth_element(exec_t.begin() + Q1 + 1, exec_t.begin() + Q2, exec_t.end());
    nth_element(exec_t.begin() + Q2 + 1, exec_t.begin() + Q3, exec_t.end());

  #ifndef NDEBUG
    cout << "Printing sorted runtimes:\n";
    for(unsigned i = 0; i < iter; i++)
      printf(" %u: %lfms\n", i, exec_t[i]);
    cout << endl;
  #endif

    // Subtract by 1 to get the right index into the array because of [0..]
    Q1_val = exec_t[Q1 - 1];
    median = exec_t[Q2 - 1];
    if((iter % 2 == 0) && (iter > 2)){
      nth_element(exec_t.begin() + Q1 + 1, exec_t.begin() + Q2-1, exec_t.end());
      median = (median + exec_t[Q2]) / 2.0;
    }
    Q3_val = exec_t[Q3-1];
  }
  double mean = exec_diff / iter;
  double variance = 0.0;
  for(unsigned i = 0; i < iter; i++){
    variance += pow(exec_t[i] - mean, 2);
  }

  double sq_sd = variance / iter;
  double sd = sqrt(variance / iter);

  double add, mul, fma, flops;
  fftwf_flops(plan, &add, &mul, &fma);
  flops = add + mul + fma;

  cout << "\nMeasurements\n" << "--------------------------\n";
  cout << "FFT Size            : " << N << "^3\n";
  cout << "Threads             : " << nthreads << endl;
  cout << "Batch               : " << how_many << endl;
  cout << "Iterations          : " << iter << endl;
  cout << "Avg Tot Runtime     : " << std::fixed << mean<< " ms\n";
  cout << "Runtime per batch   : " << (mean / how_many) << " ms\n";
  cout << "SD                  : " << sd << " ms\n";
  cout << "Throughput          : " << (flops * 1e-9) << " GFLOPs\n";
  if(iter > 2){
    cout << "Q1                  : " << std::fixed << Q1_val << " ms\n";
    cout << "Median              : " << std::fixed << median << " ms\n";
    cout << "Q3                  : " << std::fixed << Q3_val << " ms\n";
  }
  cout << "Plan Time           : " << plan_time * 1e-3<< " sec\n";

  cleanup_openmp(fftw_data, verify_data);
}

/**
 * \brief  OpenMP Multithreaded Single precision FFTW 3D execution
 *         Experiment where FFTW deals with different data on the every 
 *         iteration but computes a result at the end of every iteration 
 * \param  N          - Size of one dimension of FFT
 * \param  how_many   - number of batches
 * \param  nthreads   - number of threads
 * \param  inverse    - true if backward transform
 * \param  iter       - number of iterations of execution
 * \param  wisfile    - path to wisdom file
 */
void fftwf_openmp_many_streamappln(unsigned N, unsigned how_many, unsigned nthreads, bool inverse, unsigned iter, std::string wisfile){
    
  if ( (how_many == 0) || (nthreads == 0) || (iter == 0) )
    throw "Invalid value, should be >=1!";

  // Initialising Threads
  int threads_ok = fftwf_init_threads(); 
  if(threads_ok == 0)
    throw "Something went wrong with Multithreaded FFTW! Exiting... \n";

  // All subsequent plans will now use nthreads
  fftwf_plan_with_nthreads((int)nthreads);

  // Allocating input and verification arrays
  size_t data_sz = how_many * N * N * N;
  fftwf_complex *fftw_data = fftwf_alloc_complex(data_sz);
  fftwf_complex *verify_data = fftwf_alloc_complex(data_sz);

  // Setting direction of FFT for the plan
  int direction = FFTW_FORWARD;
  int direction_inv = FFTW_BACKWARD;
  if(inverse){
    direction = FFTW_BACKWARD;
    direction_inv = FFTW_FORWARD;
  }

  // Parameters for planning
  const int n[3] = {(int)N, (int)N, (int)N};
  int idist = N * N * N, odist = N * N * N;
  int istride = 1, ostride = 1;
  const int *inembed = n, *onembed = n;
  const unsigned fftw_plan = FFTW_PLAN;
  switch(fftw_plan){
    case FFTW_MEASURE:  cout << "FFTW Plan: Measure\n";
                        break;
    case FFTW_ESTIMATE: cout << "FFTW Plan: Estimate\n";
                        break;
    case FFTW_PATIENT:  cout << "FFTW Plan: Patient\n";
                        break;
    case FFTW_EXHAUSTIVE: cout << "FFTW Plan: Exhaustive\n";
                        break;
    case FFTW_WISDOM_ONLY: cout << "FFTW Plan: Wisdom Only \n";
                        break;
    default: throw "Incorrect plan\n";
            break;
  }

  // Import wisdom from filename
  int wis_status = 0;
  fftwf_forget_wisdom();

  if(fftw_plan != FFTW_ESTIMATE){
    wis_status = fftwf_import_wisdom_from_filename(wisfile.c_str());
    if(wis_status == 0) // could not import wisdom
      cout << "-- Cannot import wisdom from " << wisfile << endl;
    else if((wis_status == 0) && (fftw_plan == FFTW_WISDOM_ONLY)){
      cleanup_openmp(fftw_data, verify_data);
      throw "Plan should use imported wisdom. Cannot import. Quitting\n";
    }
    else                 
      cout << "-- Importing wisdom from " << wisfile << endl;
  }
  else
    cout << "Estimate Plan: Not using wisdom" << endl;

  // Make Plan
  double plan_start = getTimeinMilliSec();
  plan = fftwf_plan_many_dft(3, n, how_many, fftw_data, inembed, istride, idist, fftw_data, onembed, ostride, odist, direction, FFTW_ESTIMATE);
  double plan_time = getTimeinMilliSec() - plan_start;
  cout << "Planning Completed\n";

  if(wis_status == 0 && (fftw_plan != FFTW_WISDOM_ONLY) && (fftw_plan != FFTW_ESTIMATE)){
    // i.e., wisdom is not imported
    int exp_stat = fftwf_export_wisdom_to_filename(wisfile.c_str()); 
    if(exp_stat == 0)
      cout << "-- Could not export wisdom file to " << wisfile.c_str() << endl;
    else
      cout << "-- Exporting wisdom file to " << wisfile.c_str() << endl;
  }
  else
    cout << "Not exporting any wisdom\n";

  /* every iteration: FFT followed by inverse for verification */
  double start = 0.0, stop = 0.0, exec_diff = 0.0;
  vector<double> exec_t;
#ifndef NDEBUG
  cout << "Iteration: ";
#endif
  const size_t tot_sz = N*N*N;
  const size_t num = 256*256*256;
  float *temp1, *temp2;
  temp1 = new float [num];
  temp2 = new float [num];
  float test_res = 0.0f;

  for(size_t it = 0; it < iter; it++){
#ifndef NDEBUG
    cout << it << ", ";
#endif

    // Get new data every iteration on the same allocation 
    get_data(fftw_data, verify_data, N, how_many);

    for(unsigned i = 0; i < num; i++){
      temp1[i] = ((float) rand() / (RAND_MAX)); 
      temp2[i] = ((float) rand() / (RAND_MAX)); 
    }

    // omp_set_num_threads in the main call, also influences this
    // dot product
    test_res = cblas_sdot(num, temp1, 1, temp2, 1);

    start = getTimeinMilliSec();
    fftwf_execute(plan);
    stop = getTimeinMilliSec();

    // vector scalar product
    cblas_csscal(tot_sz, test_res, fftw_data, 1);

    double diff = stop - start;
    exec_t.push_back(diff);
    exec_diff += diff;
  }
  cout << endl;
  delete[] temp1;
  delete[] temp2;

#ifndef NDEBUG
  cout << "Printing individual runtimes:\n";
  for(unsigned i = 0; i < iter; i++)
    printf(" %u: %lfms\n", i, exec_t[i]);
  cout << endl;
#endif

  double Q1_val = 0.0, median = 0.0, Q3_val = 0.0;
  if(iter > 2){
    // using rounding up algorithm to simplify
    const unsigned Q1 = ceil(exec_t.size() / 4.0);
    const unsigned Q2 = ceil(exec_t.size() / 2.0);
    const unsigned Q3 = ceil(3 * exec_t.size() / 4.0);

    cout << "\n-- Real : Q1: " << Q1 << " Q2: " << Q2 << " Q3: " << Q3 << endl;
    cout << "-- Array: Q1: " << Q1-1 <<" Q2: "<<Q2-1<< " Q3: "<< Q3-1 << endl;

    nth_element(exec_t.begin(),          exec_t.begin() + Q1, exec_t.end());
    nth_element(exec_t.begin() + Q1 + 1, exec_t.begin() + Q2, exec_t.end());
    nth_element(exec_t.begin() + Q2 + 1, exec_t.begin() + Q3, exec_t.end());

  #ifndef NDEBUG
    cout << "Printing sorted runtimes:\n";
    for(unsigned i = 0; i < iter; i++)
      printf(" %u: %lfms\n", i, exec_t[i]);
    cout << endl;
  #endif

    // Subtract by 1 to get the right index into the array because of [0..]
    Q1_val = exec_t[Q1 - 1];
    median = exec_t[Q2 - 1];
    if((iter % 2 == 0) && (iter > 2)){
      nth_element(exec_t.begin() + Q1 + 1, exec_t.begin() + Q2-1, exec_t.end());
      median = (median + exec_t[Q2]) / 2.0;
    }
    Q3_val = exec_t[Q3-1];
  }

  double mean = exec_diff / iter;
  double variance = 0.0;
  for(unsigned i = 0; i < iter; i++){
    variance += pow(exec_t[i] - mean, 2);
  }

  double sq_sd = variance / iter;
  double sd = sqrt(variance / iter);

  double add, mul, fma, flops;
  fftwf_flops(plan, &add, &mul, &fma);
  flops = add + mul + fma;

  cout << "\nMeasurements\n" << "--------------------------\n";
  cout << "FFT Size            : " << N << "^3\n";
  cout << "Threads             : " << nthreads << endl;
  cout << "Batch               : " << how_many << endl;
  cout << "Iterations          : " << iter << endl;
  cout << "Avg Tot Runtime     : " << std::fixed << mean<< " ms\n";
  cout << "Runtime per batch   : " << (mean / how_many) << " ms\n";
  cout << "SD                  : " << sd << " ms\n";
  cout << "Throughput          : " << (flops * 1e-9) << " GFLOPs\n";
  if(iter > 2){
    cout << "Q1                  : " << std::fixed << Q1_val << " ms\n";
    cout << "Median              : " << std::fixed << median << " ms\n";
    cout << "Q3                  : " << std::fixed << Q3_val << " ms\n";
  }
  cout << "Plan Time           : " << plan_time * 1e-3<< " sec\n";

  cleanup_openmp(fftw_data, verify_data);
}
/**
 * \brief  OpenMP Multithreaded Single precision FFTW 3D execution 
 *         Experiment to simulate iterative computation using
 *         3D Convolution. The data that is calculated
 *         at the end of each iteration is reused with the next
 * \param  N          - Size of one dimension of FFT
 * \param  how_many   - number of batches
 * \param  nthreads   - number of threads
 * \param  inverse    - true if backward transform
 * \param  iter       - number of iterations of execution
 * \param  wisfile    - path to wisdom file
 */
void fftwf_openmp_many_conv(unsigned N, unsigned how_many, unsigned nthreads, bool inverse, unsigned iter, std::string wisfile){
    
  if ( (how_many == 0) || (nthreads == 0) || (iter == 0) )
    throw "Invalid value, should be >=1!";

  // Initialising Threads
  int threads_ok = fftwf_init_threads(); 
  if(threads_ok == 0)
    throw "Something went wrong with Multithreaded FFTW! Exiting... \n";

  // All subsequent plans will now use nthreads
  fftwf_plan_with_nthreads((int)nthreads);

  // Allocating input and verification arrays
  size_t data_sz = how_many * N * N * N;
  fftwf_complex *fftw_data = fftwf_alloc_complex(data_sz);
  fftwf_complex *verify_data = fftwf_alloc_complex(data_sz);

  // Setting direction of FFT for the plan
  int direction = FFTW_FORWARD;
  int direction_inv = FFTW_BACKWARD;
  if(inverse){
    direction = FFTW_BACKWARD;
    direction_inv = FFTW_FORWARD;
  }

  // Parameters for planning
  const int n[3] = {(int)N, (int)N, (int)N};
  int idist = N * N * N, odist = N * N * N;
  int istride = 1, ostride = 1;
  const int *inembed = n, *onembed = n;
  const unsigned fftw_plan = FFTW_PLAN;
  switch(fftw_plan){
    case FFTW_MEASURE:  cout << "FFTW Plan: Measure\n";
                        break;
    case FFTW_ESTIMATE: cout << "FFTW Plan: Estimate\n";
                        break;
    case FFTW_PATIENT:  cout << "FFTW Plan: Patient\n";
                        break;
    case FFTW_EXHAUSTIVE: cout << "FFTW Plan: Exhaustive\n";
                        break;
    default: throw "Incorrect plan\n";
            break;
  }

  // Import wisdom from filename
  int wis_status = 0;
  fftwf_forget_wisdom();

  if(fftw_plan != FFTW_ESTIMATE){
    wis_status = fftwf_import_wisdom_from_filename(wisfile.c_str());
    if(wis_status == 0) // could not import wisdom
      cout << "-- Cannot import wisdom from " << wisfile << endl;
    else if((wis_status == 0) && (fftw_plan == FFTW_WISDOM_ONLY)){
      cleanup_openmp(fftw_data, verify_data);
      throw "Plan should use imported wisdom. Cannot import. Quitting\n";
    }
    else                 
      cout << "-- Importing wisdom from " << wisfile << endl;
  }
  else
    cout << "Estimate Plan: Not using wisdom" << endl;

  // Make Plan
  double plan_start = getTimeinMilliSec();
  plan = fftwf_plan_many_dft(3, n, how_many, fftw_data, inembed, istride, idist, fftw_data, onembed, ostride, odist, direction, fftw_plan);
  double plan_time = getTimeinMilliSec() - plan_start;

  if(wis_status == 0 && (fftw_plan != FFTW_WISDOM_ONLY) && (fftw_plan != FFTW_ESTIMATE)){
    // i.e., wisdom is not imported
    int exp_stat = fftwf_export_wisdom_to_filename(wisfile.c_str()); 
    if(exp_stat == 0)
      cout << "-- Could not export wisdom file to " << wisfile.c_str() << endl;
    else
      cout << "-- Exporting wisdom file to " << wisfile.c_str() << endl;
  }
  else
    cout << "Not exporting any wisdom\n";

  // Make plan for verification
  plan_verify = fftwf_plan_many_dft(3, n, how_many, fftw_data, inembed, istride, idist, fftw_data, onembed, ostride, odist, direction_inv, FFTW_ESTIMATE);

  const size_t tot_sz = N*N*N;
  const float inv_tot = 1.0f / tot_sz;
  MKL_Complex8 *filter;
  filter = new MKL_Complex8 [tot_sz];

  for(unsigned i = 0; i < tot_sz; i++){
    filter[i].real = ((float) rand() / (RAND_MAX)); 
    filter[i].imag = ((float) rand() / (RAND_MAX)); 
  }

  get_data(fftw_data, verify_data, N, how_many);

  /* every iteration: FFT followed by inverse for verification */
  double start = 0.0, stop = 0.0, exec_diff = 0.0;
  vector<double> exec_t;
#ifndef NDEBUG
  cout << "Iteration: ";
#endif
  for(unsigned it = 0; it < iter; it++){
#ifndef NDEBUG
    cout << it << ", ";
#endif

    start = getTimeinMilliSec();
    fftwf_execute(plan);
    stop = getTimeinMilliSec();

    // omp_set_num_threads in the main call, also influences this
    // element-wise multiplication 
    vcMul(tot_sz, reinterpret_cast<MKL_Complex8*>(fftw_data), filter, reinterpret_cast<MKL_Complex8*>(fftw_data));

    fftwf_execute(plan_verify);

    // vector scalar product
    cblas_csscal(tot_sz, inv_tot, fftw_data, 1);

    double diff = stop - start;
    exec_t.push_back(diff);
    exec_diff += diff;
  }
  cout << endl;
  delete[] filter;

#ifndef NDEBUG
  cout << "Printing individual runtimes:\n";
  for(unsigned i = 0; i < iter; i++)
    printf(" %u: %lfms\n", i, exec_t[i]);
  cout << endl;
#endif

  double Q1_val = 0.0, median = 0.0, Q3_val = 0.0;
  if(iter > 2){
    // using rounding up algorithm to simplify
    const unsigned Q1 = ceil(exec_t.size() / 4.0);
    const unsigned Q2 = ceil(exec_t.size() / 2.0);
    const unsigned Q3 = ceil(3 * exec_t.size() / 4.0);

    cout << "\n-- Real : Q1: " << Q1 << " Q2: " << Q2 << " Q3: " << Q3 << endl;
    cout << "-- Array: Q1: " << Q1-1 <<" Q2: "<<Q2-1<< " Q3: "<< Q3-1 << endl;

    nth_element(exec_t.begin(),          exec_t.begin() + Q1, exec_t.end());
    nth_element(exec_t.begin() + Q1 + 1, exec_t.begin() + Q2, exec_t.end());
    nth_element(exec_t.begin() + Q2 + 1, exec_t.begin() + Q3, exec_t.end());

  #ifndef NDEBUG
    cout << "Printing sorted runtimes:\n";
    for(unsigned i = 0; i < iter; i++)
      printf(" %u: %lfms\n", i, exec_t[i]);
    cout << endl;
  #endif

    // Subtract by 1 to get the right index into the array because of [0..]
    Q1_val = exec_t[Q1 - 1];
    median = exec_t[Q2 - 1];
    if((iter % 2 == 0) && (iter > 2)){
      nth_element(exec_t.begin() + Q1 + 1, exec_t.begin() + Q2-1, exec_t.end());
      median = (median + exec_t[Q2]) / 2.0;
    }
    Q3_val = exec_t[Q3-1];
  }

  double mean = exec_diff / iter;
  double variance = 0.0;
  for(unsigned i = 0; i < iter; i++){
    variance += pow(exec_t[i] - mean, 2);
  }

  double sq_sd = variance / iter;
  double sd = sqrt(variance / iter);

  double add, mul, fma, flops;
  fftwf_flops(plan, &add, &mul, &fma);
  flops = add + mul + fma;

  cout << "\nMeasurements\n" << "--------------------------\n";
  cout << "FFT Size            : " << N << "^3\n";
  cout << "Threads             : " << nthreads << endl;
  cout << "Batch               : " << how_many << endl;
  cout << "Iterations          : " << iter << endl;
  cout << "Avg Tot Runtime     : " << std::fixed << mean<< " ms\n";
  cout << "Runtime per batch   : " << (mean / how_many) << " ms\n";
  cout << "SD                  : " << sd << " ms\n";
  cout << "Throughput          : " << (flops * 1e-9) << " GFLOPs\n";
  if(iter > 2){
    cout << "Q1                  : " << std::fixed << Q1_val << " ms\n";
    cout << "Median              : " << std::fixed << median << " ms\n";
    cout << "Q3                  : " << std::fixed << Q3_val << " ms\n";
  }
  cout << "Plan Time           : " << plan_time * 1e-3<< " sec\n";

  cleanup_openmp(fftw_data, verify_data);
}