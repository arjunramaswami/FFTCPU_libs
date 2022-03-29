#include <iostream>
#include <cmath>
#include <omp.h>
#include "mkl.h"  

#include "cxxopts.hpp" // Cmd-Line Args parser
#include "mkl_many.hpp"
#include "helper.hpp"
using namespace std;

/**
 * \brief random sp floating points
 * \param fft_data    : pointer to fft array
 * \param verify_data : pointer to verification array
 * \param N           : number of total points
 * \param how_many    : number of batched implementations of FFTW
 */
void get_data(MKL_Complex8 *fft_data, MKL_Complex8 *verify_data, unsigned N, unsigned how_many){

  float re_val = 0.0f, img_val = 0.0f;

  for(unsigned i = 0; i < how_many * N; i++){
    re_val = ((float) rand() / (RAND_MAX));
    img_val = ((float) rand() / (RAND_MAX));

    verify_data[i].real = fft_data[i].real = re_val;
    verify_data[i].imag = fft_data[i].imag = img_val;

#ifndef NDEBUG          
  printf("fft[%u]: (%f, %f)\n", i, fft_data[i].real, fft_data[i].imag);
#endif
  }
}

/**
 * \brief Output specific error message
 */
static void error_msg(MKL_LONG status){
  if(status != DFTI_NO_ERROR){
    char *error_message = DftiErrorMessage(status);
    printf("Failed with message %s \n", error_message);
  }
}

/**
 * \brief  OpenMP Multithreaded Single precision MKL
 * \param  N          - Size of one dimension of FFT
 * \param  dim        - number of dimensions
 * \param  how_many   - number of batches
 * \param  nthreads   - number of threads
 * \param  inverse    - true if backward transform
 * \param  iter       - number of iterations of execution
 * \param  wisfile    - path to wisdom file
 */
void mkl_openmp_many(unsigned N, unsigned dims, unsigned how_many, unsigned nthreads, bool inverse, unsigned iter){

  if ( (how_many == 0) || (nthreads == 0) || (iter == 0) | (dims > 3))
    throw "Invalid value, should be >=1!";

  const unsigned num = pow(N, dims);

  MKL_Complex8 *fft_data = (MKL_Complex8*)mkl_malloc(num * sizeof(MKL_Complex8), 64);
  if (fft_data == NULL)
    throw "FFT data allocation failed";

  MKL_Complex8 *verify_data = (MKL_Complex8*)mkl_malloc(num * sizeof(MKL_Complex8), 64);
  if (verify_data == NULL)
    throw "FFT verification data allocation failed";

  const MKL_LONG dim = dims;
  MKL_LONG *n = (MKL_LONG*)mkl_calloc(dims, sizeof(MKL_LONG), 64);
  for(unsigned i = 0; i < dims; i++)
    n[i] = (MKL_LONG)N;

  MKL_LONG status;
  DFTI_DESCRIPTOR_HANDLE fft_desc_handle = NULL;
  status = DftiCreateDescriptor(&fft_desc_handle, DFTI_SINGLE, DFTI_COMPLEX, dim, n);
  error_msg(status);
  status = DftiSetValue(fft_desc_handle, DFTI_PLACEMENT, DFTI_INPLACE);
  error_msg(status);
  status = DftiSetValue(fft_desc_handle, DFTI_THREAD_LIMIT, nthreads);
  error_msg(status);
  status = DftiSetValue(fft_desc_handle, DFTI_NUMBER_OF_TRANSFORMS, 1);
  error_msg(status);
  status = DftiCommitDescriptor(fft_desc_handle);
  error_msg(status);

  double start = 0.0, stop = 0.0, exec_diff = 0.0;
  vector<double> exec_t;
#ifndef NDEBUG
  cout << "Iteration: ";
#endif
  for(unsigned it = 0; it < iter; it++){
#ifndef NDEBUG
    cout << it << ", ";
#endif
    get_data(fft_data, verify_data, num, how_many);

    start = getTimeinMilliSec();
    DftiComputeForward(fft_desc_handle, fft_data);
    stop = getTimeinMilliSec();

    DftiComputeBackward(fft_desc_handle, fft_data);
    bool status_out = verify_mkl(fft_data, verify_data, num, how_many);
    if(!status_out){
      mkl_free(fft_data);
      mkl_free(verify_data);
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

  cout << "\nMeasurements\n" << "--------------------------\n";
  cout << "FFT Size            : " << N << "^"<< dims <<"\n";
  cout << "Threads             : " << nthreads << endl;
  cout << "Batch               : " << how_many << endl;
  cout << "Iterations          : " << iter << endl;
  cout << "Avg Tot Runtime     : " << std::fixed << mean<< " ms\n";
  cout << "Runtime per batch   : " << (mean / how_many) << " ms\n";
  cout << "SD                  : " << sd << " ms\n";
  if(iter > 2){
    cout << "Q1                  : " << std::fixed << Q1_val << " ms\n";
    cout << "Median              : " << std::fixed << median << " ms\n";
    cout << "Q3                  : " << std::fixed << Q3_val << " ms\n";
  }

  DftiFreeDescriptor(&fft_desc_handle);

  mkl_free(n);
  mkl_free(fft_data);
  mkl_free(verify_data);
}

/**
 * \brief  OpenMP Multithreaded Single precision MKL FFT
 *         Experiment where MKL deals with different data on the every 
 *         iteration but computes a result at the end of every iteration 
 * \param  N          - Size of one dimension of FFT
 * \param  how_many   - number of batches
 * \param  nthreads   - number of threads
 * \param  inverse    - true if backward transform
 * \param  iter       - number of iterations of execution
 * \param  wisfile    - path to wisdom file
 */
void mkl_openmp_stream(unsigned N, unsigned how_many, unsigned nthreads, bool inverse, unsigned iter){

  if ( (how_many == 0) || (nthreads == 0) || (iter == 0) )
    throw "Invalid value, should be >=1!";

  MKL_Complex8 *fft_data = (MKL_Complex8*)mkl_malloc(N * N * N * sizeof(MKL_Complex8), 64);
  if (fft_data == NULL)
    throw "FFT data allocation failed";

  MKL_Complex8 *verify_data = (MKL_Complex8*)mkl_malloc(N * N * N * sizeof(MKL_Complex8), 64);
  if (verify_data == NULL)
    throw "FFT verification data failed";


  const MKL_LONG dim = 3;
  const MKL_LONG n[3] = {N, N, N};
  MKL_LONG status;

  DFTI_DESCRIPTOR_HANDLE fft_desc_handle = NULL;
  status = DftiCreateDescriptor(&fft_desc_handle, DFTI_SINGLE, DFTI_COMPLEX, dim, n);
  error_msg(status);
  status = DftiSetValue(fft_desc_handle, DFTI_PLACEMENT, DFTI_INPLACE);
  error_msg(status);
  status = DftiSetValue(fft_desc_handle, DFTI_THREAD_LIMIT, nthreads);
  error_msg(status);
  status = DftiCommitDescriptor(fft_desc_handle);
  error_msg(status);

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

  for(unsigned it = 0; it < iter; it++){
#ifndef NDEBUG
    cout << it << ", ";
#endif

    get_data(fft_data, verify_data, tot_sz, how_many);

    for(unsigned i = 0; i < num; i++){
      temp1[i] = ((float) rand() / (RAND_MAX)); 
      temp2[i] = ((float) rand() / (RAND_MAX)); 
    }

    // omp_set_num_threads in the main call, also influences this
    // dot product
    test_res = cblas_sdot(num, temp1, 1, temp2, 1);

    start = getTimeinMilliSec();
    DftiComputeForward(fft_desc_handle, fft_data);
    stop = getTimeinMilliSec();

    // vector scalar product
    cblas_csscal(tot_sz, test_res, fft_data, 1);

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

  cout << "\nMeasurements\n" << "--------------------------\n";
  cout << "FFT Size            : " << N << "^3\n";
  cout << "Threads             : " << nthreads << endl;
  cout << "Batch               : " << how_many << endl;
  cout << "Iterations          : " << iter << endl;
  cout << "Avg Tot Runtime     : " << std::fixed << mean<< " ms\n";
  cout << "Runtime per batch   : " << (mean / how_many) << " ms\n";
  cout << "SD                  : " << sd << " ms\n";
  if(iter > 2){
    cout << "Q1                  : " << std::fixed << Q1_val << " ms\n";
    cout << "Median              : " << std::fixed << median << " ms\n";
    cout << "Q3                  : " << std::fixed << Q3_val << " ms\n";
  }

  DftiFreeDescriptor(&fft_desc_handle);
  mkl_free(fft_data);
  mkl_free(verify_data);
}