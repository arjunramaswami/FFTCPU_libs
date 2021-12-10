// Author: Arjun Ramaswami
#include <iostream>
#include <cmath>
#include <cfloat> // FLT_EPSILON
#include <omp.h>
#include <fftw3.h>
#include "config.h"

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
 * \param n0, n1, n2  : number of points in each dimension
 * \param local_n0    : number of points in the n0 dim
 * \param local_start : starting point in the n0 dim
 * \param H1, H2, H3  : harmonic to modify frequency of discrete time signal
 * \param how_many    : number of batched implementations of FFTW
 */
void get_sp_mpi_many_input(fftwf_complex *fftw_data, fftwf_complex *verify_data,ptrdiff_t N, ptrdiff_t local_n0, ptrdiff_t local_start, unsigned H1, unsigned H2, unsigned H3, unsigned how_many){

  unsigned index; 
  float TWOPI = 6.2831853071795864769;
  float phase, phase1, phase2, phase3;
  double re_val = 0.0, img_val = 0.0;

  /*
  * Interleaved data reordering for batched implementation
  *   i.e. elements of multiple transforms are adjacent to each other
  *   e.g. first element of batches 1, 2, 3 are in adjacent positions
  *        meaning, stride of 3 for second element of same batch 
  *                 distance of 1 between batch data
  * Creating distinct input data to different batches to stress CPU
  *   by modifying the phase by the index (many)
  */
  for (ptrdiff_t i = 0; i < local_n0; i++) {
    for (ptrdiff_t j = 0; j < N; j++) {
      for (ptrdiff_t k = 0; k < N; k++) {

        for(ptrdiff_t many = 0; many < how_many; many++){

          index = (i * N * N * how_many) + (j * N * how_many) + (k * how_many) + many;

          // considering the H1, H2, H3 are inverse for backward FFT
          //   multiply with the index
          phase1 = moda(i + local_start, H1, N) / N;
          phase2 = moda(j, H2, N) / N;
          phase3 = moda(k, H3, N) / N;
          phase = phase1 + phase2 + phase3;

          re_val = cosf( TWOPI * phase ) / (N*N*N);
          img_val = sinf( TWOPI * phase ) / (N*N*N);

          /*
          re_val = ((double) rand() / (RAND_MAX));
          img_val = ((double) rand() / (RAND_MAX));
          */

          verify_data[index][0] = fftw_data[index][0] = re_val;
          verify_data[index][1] = fftw_data[index][1] = img_val;

#ifdef VERBOSE
          cout << many << ": " << i << " " << j << " " << k << " : fftw[" << index << "] = ";
          cout <<"(" << fftw_data[index][0] << ", " << fftw_data[index][1] << ")";
          cout <<" = (" << verify_data[index][0] << ", " << verify_data[index][1] << ")";
          cout << endl;
#endif
        }
      }
    }
  }
}

/**
 * \brief create single precision floating points values for FFT computation for each process level block
 * \param fftw_data   : pointer to 3d number of sp points for FFTW
 * \param verify_data : pointer to 3d number of sp points for verification
 * \param N           : number of points in each dimension
 * \param how_many    : number of batched implementations of FFTW
 */
void get_data(fftwf_complex *fftw_data, fftwf_complex *verify_data, size_t N, unsigned how_many){

  double re_val = 0.0, img_val = 0.0;

  for(unsigned i = 0; i < how_many * N * N * N; i++){
    re_val = ((double) rand() / (RAND_MAX));
    img_val = ((double) rand() / (RAND_MAX));

    verify_data[i][0] = fftw_data[i][0] = re_val;
    verify_data[i][1] = fftw_data[i][1] = img_val;

#ifndef NDEBUG          
  printf("fftw[%u]: (%f, %f)\n", i, fftw_data[i][0], fftw_data[i][1]);
#endif
  }
}


/**
 * \brief  Verify single precision batched FFT3d computation using FFTW
 * \param  fftw_data   : pointer to 3D number of sp points after FFTW
 * \param  verify_data : pointer to 3D number of sp points for verification
 * \param  N1, N2, N3  : fft size
 * \param  H1, H2, H3  : harmonic to modify frequency of discrete time signal
 * \param  how_many    : number of batched implementations of FFTW
 * \return true if successful, false otherwise
 */
bool verify_fftw(fftwf_complex *fftw_data, fftwf_complex *verify_data, unsigned N, unsigned how_many){

  double magnitude = 0.0, noise = 0.0, mag_sum = 0.0, noise_sum = 0.0;

  for(size_t i = 0; i < how_many * N * N * N; i++){

    // FFT -> iFFT is scaled by dimensions (N*N*N)
    verify_data[i][0] = verify_data[i][0] * N * N * N;
    verify_data[i][1] = verify_data[i][1] * N * N * N;

    magnitude = verify_data[i][0] * verify_data[i][0] + \
                      verify_data[i][1] * verify_data[i][1];
    noise = (verify_data[i][0] - fftw_data[i][0]) \
        * (verify_data[i][0] - fftw_data[i][0]) + 
        (verify_data[i][1] - fftw_data[i][1]) * (verify_data[i][1] - fftw_data[i][1]);

    mag_sum += magnitude;
    noise_sum += noise;

#ifndef NDEBUG
    cout << i << ": fftw_out[" << i << "] = (" << fftw_data[i][0] << ", " << fftw_data[i][1] << ")";
    cout << " = (" << verify_data[i][0] << ", " << verify_data[i][1] << ")";
    cout << endl;
#endif
  }

  float db = 10 * log(mag_sum / noise_sum) / log(10.0);

    // if SNR greater than 120, verification passes
  if(db > 120){
    return true;
  }
  else{ 
    cout << "Signal to noise ratio on output sample: " << db << " --> FAILED \n\n";
    return false;
  }
}

/**
 * \brief  OpenMP Multithreaded Single precision FFTW 3D execution
 * \param  N          - Size of one dimension of FFT
 * \param  how_many   - number of batches
 * \param  nthreads   - number of threads
 * \param  inverse    - true if backward transform
 * \param  iter       - number of iterations of execution
 */
void fftwf_openmp_many(unsigned N, unsigned how_many, unsigned nthreads, bool inverse, unsigned iter){
    
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
  const int n[3] = {N, N, N};
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

  // Make Plan
  double plan_start = getTimeinMilliSec();

  plan = fftwf_plan_many_dft(3, n, how_many, fftw_data, inembed, istride, idist, fftw_data, onembed, ostride, odist, direction, fftw_plan);

  double plan_time = getTimeinMilliSec() - plan_start;

  // Make plan for verification
  plan_verify = fftwf_plan_many_dft(3, n, how_many, fftw_data, inembed, istride, idist, fftw_data, onembed, ostride, odist, direction_inv, FFTW_ESTIMATE);

  /* every iteration: FFT followed by inverse for verification */
  double start = 0.0, stop = 0.0, exec_diff = 0.0;
  double exec_t[iter];
  cout << "Iteration: ";
  for(size_t it = 0; it < iter; it++){
    cout << it << ", ";

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

    exec_t[it] = stop - start;
    exec_diff += stop - start;
  }
  cout << endl;

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

  cout << "Printing individual runtimes:\n";
  for(unsigned i = 0; i < iter; i++)
    printf(" %u: %lfms\n", i, exec_t[i]);
  cout << endl;

  cout << "\nMeasurements\n" << "--------------------------\n";
  cout << "FFT Size            : " << N << "^3\n";
  cout << "Threads             : " << nthreads << endl;
  cout << "Batch               : " << how_many << endl;
  cout << "Iterations          : " << iter << endl;
  cout << "Avg Tot Runtime     : " << std::fixed << (mean * 1e3)<< " ms\n";
  cout << "Runtime per batch   : " << ((mean / how_many) * 1e3) << " ms\n";
  cout << "SD                  : " << sd * 1e3 << " ms\n";
  cout << "Throughput          : " << (flops * 1e-9) << " GFLOPs\n";
  cout << "Plan Time           : " << plan_time << "sec\n";

  cleanup_openmp(fftw_data, verify_data);
}
