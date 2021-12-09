// Author: Arjun Ramaswami
#include <iostream>
#include <cmath>
#include <cfloat> // FLT_EPSILON
#include <omp.h>
#include <mpi.h>
#include <fftw3.h>
#include <fftw3-mpi.h>
#include "config.h"

#include "cxxopts.hpp" // Cmd-Line Args parser
#include "fftw_many_sp.hpp"
#include "helper.hpp"
using namespace std;

static fftwf_plan plan, plan_verify;

static void cleanup_mpi(fftwf_complex *per_process_data, fftwf_complex *verify_per_process){
  // Cleanup 
  fftwf_free(per_process_data);
  fftwf_free(verify_per_process);
  //fftwf_cleanup_threads(); // Thread Cleanup
  fftwf_mpi_cleanup();
  fftwf_destroy_plan(plan);
  fftwf_destroy_plan(plan_verify);
}

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
          /*
          phase1 = moda(i + local_start, H1, N) / N;
          phase2 = moda(j, H2, N) / N;
          phase3 = moda(k, H3, N) / N;
          phase = phase1 + phase2 + phase3;

          re_val = cosf( TWOPI * phase ) / (N*N*N);
          img_val = sinf( TWOPI * phase ) / (N*N*N);
          */

          re_val = ((double) rand() / (RAND_MAX));
          img_val = ((double) rand() / (RAND_MAX));

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
 * \param H1, H2, H3  : harmonic to modify frequency of discrete time signal
 * \param how_many    : number of batched implementations of FFTW
 */
void get_sp_many_input(fftwf_complex *fftw_data, fftwf_complex *verify_data,size_t N, unsigned H1, unsigned H2, unsigned H3, unsigned how_many){

  unsigned index; 
  float TWOPI = 6.2831853071795864769;
  float phase, phase1, phase2, phase3;
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

  #ifdef DEBUG          
          printf(" %d %d %d : fftw[%d] = (%f, %f) \n", i, j, k, index, fftw_data[index][0], fftw_data[index][1]);
  #endif
        }
      }
    }
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
bool verify_fftw(fftwf_complex *fftw_data, fftwf_complex *verify_data, unsigned N, unsigned H1, unsigned H2, unsigned H3, unsigned how_many){

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

#ifdef VERBOSE
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
 * \brief  Hybrid Single precision FFTW execution
 * \param  dim        - number of dimensions of FFT (supports only 3)
 * \param  N          - Size of one dimension of FFT
 * \param  how_many   - number of batches
 * \param  nthreads   - number of threads
 * \param  inverse    - true if backward transform
 * \param  iter       - number of iterations of execution
 */
void fftwf_hybrid_many(unsigned dim, unsigned N, unsigned how_many, unsigned nthreads, bool inverse, unsigned iter, std::string wisfile, bool noverify){
    
  if(dim != 3){
    throw "Currently supports only 3D FFT!";
  }
  // else if ((N & N-1) != 0){
  //   throw "Invalid N value, should be a power of 2!";
  // }
  else if ( (how_many == 0) || (nthreads == 0) || (iter == 0) ){
    throw "Invalid value, should be >=1!";
  }

  unsigned H1 = 1, H2 = 1, H3 = 1;
  double gather_diff = 0.0, exec_diff = 0.0;
  double exec[iter];

  int world_size, myrank, namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Comm_size(MPI_COMM_WORLD, &world_size); 
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Get_processor_name(processor_name, &namelen);
  srand(myrank +1);
  
  // One time initialization to use threads
  int threads_ok = fftwf_init_threads();
  if(threads_ok == 0){
    throw "Something went wrong with SP Multithreaded Many FFTW! Exiting..";
  }
  else{
    fftwf_plan_with_nthreads(nthreads); // not thread-safe, single thread call
  }

  // One time initialization to use MPI processes
  fftwf_mpi_init();

#ifndef NDEBUG
  #pragma omp parallel 
  {
      // prints number of threads per rank on machine
      int num_th = omp_get_num_threads();
      int thr_num = omp_get_thread_num();
      cout << "Hybrid: Hello from thread "<< thr_num << " out of "<< num_th;
      cout << " from process " << myrank << " out of " << world_size;
      cout << " on " << processor_name << endl;
  }
#endif

  ptrdiff_t alloc_local, local_n0, local_0_start;
  ptrdiff_t n[3] = {N, N, N};

  // get local data size
  alloc_local = fftwf_mpi_local_size_many(3, n, how_many,   FFTW_MPI_DEFAULT_BLOCK, MPI_COMM_WORLD, &local_n0, &local_0_start);

#ifndef NDEBUG
  cout << endl;
  for(size_t i = 0; i < world_size; i++){
    if(myrank == i){
      cout << "Rank " << myrank << " of " << world_size;
      cout << "\n\t" << alloc_local << "pts in local transform of total: " <<  N*N*N*how_many;
      cout << "\n\t" << local_n0 << " blocks starting from " << local_0_start; cout << "\n\t" << "Each block of " << N*N << "pts \n\n";
     }
  }
#endif

  // allocate
  fftwf_complex *per_process_data = fftwf_alloc_complex(alloc_local);
  fftwf_complex *verify_per_process = fftwf_alloc_complex(alloc_local);

  // direction of FFT for the plan
  int direction = FFTW_FORWARD;
  int direction_inv = FFTW_BACKWARD;
  if(inverse){
    direction = FFTW_BACKWARD;
    direction_inv = FFTW_FORWARD;
  }

  unsigned fftw_plan = FFTW_PLAN;

  if(myrank == 0){
    switch(fftw_plan){
      case FFTW_MEASURE:  cout << "FFTW Plan Measure\n";
                          break;
      case FFTW_ESTIMATE: cout << "FFTW Plan Estimate\n";
                          break;
      case FFTW_PATIENT:  cout << "FFTW Plan Patient\n";
                          break;
      case FFTW_EXHAUSTIVE: cout << "FFTW Plan Exhaustive\n";
                          break;
      default: throw "Incorrect plan\n";
              break;
    }
  }

  int status = MPI_Bcast(&noverify, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
  if(status != MPI_SUCCESS){
    throw "Wisdom broadcast unsuccessful";
  }
  // Planning for verification. Need to do this before importing wisdom.
  if(!noverify){
    plan_verify = fftwf_mpi_plan_many_dft(3, n, how_many, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, per_process_data, per_process_data, MPI_COMM_WORLD, direction_inv, fftw_plan);
  }

  int wis_status = 0;
  if(myrank == 0){
    wis_status = fftwf_import_wisdom_from_filename(wisfile.c_str());
    if(wis_status == 0){
      cout << "-- Cannot import wisdom from " << wisfile << endl;
    }
    else{
      cout << "-- Importing wisdom from " << wisfile << endl;
    }
  }
  // BCast wis_status to MPI processes i.e. 0 if wisdom not imported
  status = MPI_Bcast(&wis_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if(status != MPI_SUCCESS){
    throw "Wisdom broadcast unsuccessful";
  }

  if(wis_status != 0){
    fftwf_mpi_broadcast_wisdom(MPI_COMM_WORLD);
    fftw_plan = FFTW_WISDOM_ONLY | FFTW_ESTIMATE;
  }

  // plan
  double plan_start = MPI_Wtime();
  plan = fftwf_mpi_plan_many_dft(3, n, how_many, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, per_process_data, per_process_data, MPI_COMM_WORLD, direction, fftw_plan);
  double plan_time = MPI_Wtime() - plan_start;

  if(myrank == 0)
    cout << "\nPlanning completed" << endl;

  if(wis_status == 0){
    fftwf_mpi_gather_wisdom(MPI_COMM_WORLD);
  }
  if(myrank == 0 && wis_status == 0){
    int exp_stat = fftwf_export_wisdom_to_filename(wisfile.c_str()); 
    if(exp_stat == 0){
      cout << "-- Could not export wisdom file to " << wisfile.c_str() << endl;
    }
    else{
      cout << "-- Exporting wisdom file to " << wisfile.c_str() << endl;
    }
  }

  size_t data_sz = N * N * N * how_many;
  fftwf_complex *total_data = fftwf_alloc_complex(data_sz); // gathered data
  fftwf_complex *total_verify = fftwf_alloc_complex(data_sz); // -- || -- 

  size_t num = 256*256*256;
  double *test_res, *temp1, *temp2;
  test_res = new double [num];
  temp1 = new double [num];
  temp2 = new double [num];

  for(unsigned i = 0; i < num; i++){
    test_res[i] = 0.0; 
    temp1[i] = ((double) rand() / (RAND_MAX)); 
    temp2[i] = ((double) rand() / (RAND_MAX)); 
  }

  double scale = 2.0;

  /*
  * iterate iter times 
  * time the required planning and transform phase
  * verify the transformation by performing its inverse
  * verify whether the inverse matches the original data after scaling correctly
  */
  for(size_t it = 0; it < iter; it++){
    // fill data
    if(inverse){
      get_sp_mpi_many_input(per_process_data, verify_per_process, N, local_n0, local_0_start, -H1, -H2, -H3, how_many);
    }
    else{
      get_sp_mpi_many_input(per_process_data, verify_per_process, N, local_n0, local_0_start, H1, H2, H3, how_many);
    }

    if(myrank == 0)
      cout << "Starting execution.." << endl;

    MPI_Barrier(MPI_COMM_WORLD);

    // execute
    double exec_start = MPI_Wtime();
    fftwf_execute(plan); 
    double exec_end = MPI_Wtime();

    if(myrank == 0)
      cout << "Completed execution.." << endl;

    // Inverse transform to verify
    if(!noverify){
      fftwf_execute(plan_verify);
      if(myrank == 0)
        cout << "Completed verification.." << endl;
    }

    // Gather transformed data from all processes to master 
    double gather_start = MPI_Wtime();

    MPI_Gather(&per_process_data[0], alloc_local, MPI_COMPLEX, &total_data[0], alloc_local, MPI_COMPLEX, 0, MPI_COMM_WORLD);

    double gather_end = MPI_Wtime();
   
    if(myrank == 0)
      cout << "Gathered results.." << endl;

    for(unsigned i = 0; i < num; i++){
      test_res[i] = (temp1[i] * temp2[i]) + scale;
    }

    if(myrank == 0)
      cout << "Completed multiplication.." << endl;

    // Gather initial data from all processes to master
    if(!noverify){
      MPI_Gather(&verify_per_process[0], alloc_local, MPI_COMPLEX, &total_verify[0], alloc_local, MPI_COMPLEX, 0, MPI_COMM_WORLD);
    }
    
    // verify transformed and original data
    if(myrank == 0){
      if(!noverify){
        bool status = verify_fftw(total_data, total_verify, N, H1, H2, H3, how_many);
        if(!status){
          cerr << "Error in transformation\n";
          cleanup_mpi(per_process_data, verify_per_process);
          throw "Error in verification transformation";
        }
      }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
//#ifndef NDEBUG
    for(size_t i = 0; i < world_size; i++){
      if(myrank == i){
        const double exec_temp = (exec_end - exec_start) * 1e3;
        const double gather_temp = (gather_end - gather_start)*1e3;
        printf("Rank: %d \n\titer %lu \n\texec time: %lf ms\n\tgather time: %lf ms\n\n", myrank, it, exec_temp, gather_temp);
        fflush (stdout);
        MPI_Barrier(MPI_COMM_WORLD);
      }
      else{
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
//#endif

    exec_diff += (exec_end - exec_start);
    gather_diff += (gather_end - gather_start);
    exec[it] = (exec_end - exec_start);
  }

  free(test_res);
  free(temp1);
  free(temp2);
  
  // Gathered data
  free(total_data); // large sz
  free(total_verify); // large sz

  double mean = exec_diff / iter;
  gather_diff = gather_diff / iter;

  double variance = 0.0;
  for(unsigned i = 0; i < iter; i++){
    variance += pow(exec[i] - mean, 2);
  }
  double sq_sd = variance / iter;
  double sd = sqrt(variance / iter);

  double add, mul, fma, flops;
  fftwf_flops(plan, &add, &mul, &fma);
  flops = add + mul + fma; 

  // pool average
  double avg;
  MPI_Allreduce(&mean, &avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  avg = avg / world_size;

  // pool sd
  double pool_sd = 0.0;
  MPI_Allreduce(&sq_sd, &pool_sd, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  pool_sd = sqrt(pool_sd / world_size);

  // calculating total flops
  double tot_flops;
  MPI_Reduce(&flops, &tot_flops, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if(myrank == 0){
    // Print to console the configuration chosen to execute during runtime
    print_config(N, false, world_size, nthreads, how_many, inverse, iter);
    cout << "\nTime to plan: " << plan_time << "sec\n";
    bool status = print_results(avg, gather_diff, tot_flops, sd, N, world_size, nthreads, iter, how_many);
    if(!status){
      cleanup_mpi(per_process_data, verify_per_process);
      throw "Printing Results function failed!";
    }
  }
}