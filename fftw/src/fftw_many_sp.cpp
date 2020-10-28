// Author: Arjun Ramaswami
#include <iostream>
#include <cmath>
#include <cfloat> // FLT_EPSILON
#include <omp.h>
#include <mpi.h>
#include <fftw3.h>
#include <fftw3-mpi.h>

#include "cxxopts.hpp" // Cmd-Line Args parser
#include "fftw_many_sp.h"
#include "helper.h"
using namespace std;

static fftwf_plan plan, plan_verify;

static void cleanup(fftwf_complex *per_process_data, fftwf_complex *verify_per_process){
  // Cleanup 
  fftwf_free(per_process_data);
  fftwf_free(verify_per_process);
  //fftwf_cleanup_threads(); // Thread Cleanup
  fftwf_mpi_cleanup();
  fftwf_destroy_plan(plan);
  fftwf_destroy_plan(plan_verify);
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
void get_sp_input_data(fftwf_complex *fftw_data, fftwf_complex *verify_data,ptrdiff_t N, ptrdiff_t local_n0, ptrdiff_t local_start, unsigned H1, unsigned H2, unsigned H3, unsigned how_many){

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
          phase1 = moda(i + local_start, H1 * many, N) / N;
          phase2 = moda(j, H2 * many, N) / N;
          phase3 = moda(k, H3 * many, N) / N;
          phase = phase1 + phase2 + phase3;

          re_val = cosf( TWOPI * phase ) / (N*N*N);
          img_val = sinf( TWOPI * phase ) / (N*N*N);

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
    cout << " = (" << total_verify[i][0] << ", " << total_verify[i][1] << ")";
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
 * \brief  Distributed Single precision FFTW execution
 * \param  dim        - number of dimensions of FFT (supports only 3)
 * \param  N          - Size of one dimension of FFT
 * \param  how_many   - number of batches
 * \param  nthreads   - number of threads
 * \param  inverse    - true if backward transform
 * \param  iter       - number of iterations of execution
 */
void fftwf_many_sp(unsigned dim, unsigned N, unsigned how_many, unsigned nthreads, bool inverse, unsigned iter){
    
  if(dim != 3){
    throw "Currently supports only 3D FFT!";
  }
  else if ((N & N-1) != 0){
    throw "Invalid N value, should be a power of 2!";
  }
  else if ( (how_many == 0) || (nthreads == 0) || (iter == 0) ){
    throw "Invalid value, should be >=1!";
  }

  unsigned H1 = 1, H2 = 1, H3 = 1;
  double gather_diff = 0.0, exec_diff = 0.0, inp_data_diff = 0.0;

  // One time initialization to use threads
  int threads_ok = fftwf_init_threads();
  if(threads_ok == 0){
    throw "Something went wrong with SP Multithreaded Many FFTW! Exiting..";
  }

  // include threads in fftw plan
  if(threads_ok){
#ifdef VERBOSE
    if(myrank == 0)
      cout << "Threads per rank: " nthreads << endl;
#endif
    fftwf_plan_with_nthreads(nthreads); // not thread-safe, single thread call
  }

  // One time initialization to use MPI processes
  fftwf_mpi_init();

  int world_size, myrank, namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Comm_size(MPI_COMM_WORLD, &world_size); 
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Get_processor_name(processor_name, &namelen);
  
#ifdef VERBOSE
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

#ifdef VERBOSE
  for(size_t i = 0; i < world_size; i++){
    if(myrank == i){
      cout << "Rank " << myrank << " of " << world_size;
      cout << "\n\t" << alloc_local << "pts in local transform of total: " <<  N*N*N*how_many;
      cout << "\n\t" << local_n0 << " blocks starting from " << local_0_start; cout << "\n\t" << " each block of " << N*N << "pts \n\n"; }
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

#ifdef VERBOSE
  if(myrank == 0)
    cout << "Configuring plan for single precision FFT" << endl;
#endif
    
  // plan
  MPI_Barrier(MPI_COMM_WORLD);
  double plan_start = MPI_Wtime();
#ifdef MEASURE
  plan = fftwf_mpi_plan_many_dft(3, n, how_many, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, per_process_data, per_process_data, MPI_COMM_WORLD, direction, FFTW_MEASURE);
#elif PATIENT
  plan = fftwf_mpi_plan_many_dft(3, n, how_many, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, per_process_data, per_process_data, MPI_COMM_WORLD, direction, FFTW_PATIENT);
#elif EXHAUSTIVE
  plan = fftwf_mpi_plan_many_dft(3, n, how_many, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, per_process_data, per_process_data, MPI_COMM_WORLD, direction, FFTW_EXHAUSTIVE);
#else
  plan = fftwf_mpi_plan_many_dft(3, n, how_many, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, per_process_data, per_process_data, MPI_COMM_WORLD, direction, FFTW_ESTIMATE);
#endif
  MPI_Barrier(MPI_COMM_WORLD);
  double plan_time = MPI_Wtime() - plan_start;

  // Planning for verification
  plan_verify = fftwf_mpi_plan_many_dft(3, n, how_many, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, per_process_data, per_process_data, MPI_COMM_WORLD, direction_inv, FFTW_MEASURE);

  /*
  * iterate iter times 
  * time the required planning and transform phase
  * verify the transformation by performing its inverse
  * verify whether the inverse matches the original data after scaling correctly
  */
  for(size_t it = 0; it < iter; it++){
    size_t data_sz = N * N * N * how_many;
    fftwf_complex *total_data = fftwf_alloc_complex(data_sz); //-- large sz
    fftwf_complex *total_verify = fftwf_alloc_complex(data_sz); //-- large sz

    // fill data
    MPI_Barrier(MPI_COMM_WORLD);
    double inp_data_start = MPI_Wtime();
    if(inverse){
      get_sp_input_data(per_process_data, verify_per_process, N, local_n0, local_0_start, -H1, -H2, -H3, how_many);
    }
    else{
      get_sp_input_data(per_process_data, verify_per_process, N, local_n0, local_0_start, H1, H2, H3, how_many);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double inp_data_end = MPI_Wtime();

    // execute
    MPI_Barrier(MPI_COMM_WORLD);
    double exec_start = MPI_Wtime();
    fftwf_execute(plan); 
    MPI_Barrier(MPI_COMM_WORLD);
    double exec_end = MPI_Wtime();

    // Inverse transform to verify
    fftwf_execute(plan_verify);

    // Gather transformed data from all processes to master 
    double gather_start = MPI_Wtime();
    // TODO: exception
    MPI_Gather(&per_process_data[0], alloc_local, MPI_COMPLEX, &total_data[0], alloc_local, MPI_COMPLEX, 0, MPI_COMM_WORLD);
    double gather_end = MPI_Wtime();
   
    // Gather initial data from all processes to master
    MPI_Gather(&verify_per_process[0], alloc_local, MPI_COMPLEX, &total_verify[0], alloc_local, MPI_COMPLEX, 0, MPI_COMM_WORLD);
    
    /*
    if(checkStatus(status)){
      cleanup(per_process_data, verify_per_process);
      return 1;
    }
    */

    // verify transformed and original data
    if(myrank == 0){
      bool status = verify_fftw(total_data, total_verify, N, H1, H2, H3, how_many);
      if(!status){
        cerr << "Error in transformation\n";
        cleanup(per_process_data, verify_per_process);
        throw "Error in verification transformation";
      }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
#ifdef VERBOSE
    for(size_t i = 0; i < world_size; i++){
      if(myrank == i){
        cout << "Rank: " << myrank;
        cout << "\n\titer " << it;
        cout << "\n\texec time: " << exec_end - exec_start << "sec";
        cout << "\n\tgather time: " << gather_end - gather_start << "sec";
        cout << "\n\n";
      }
    }
#endif

    inp_data_diff += (inp_data_end - inp_data_start);
    exec_diff += (exec_end - exec_start);
    gather_diff += (gather_end - gather_start);

    free(total_data); // large sz
    free(total_verify); // large sz
  }

  inp_data_diff = inp_data_diff / iter;
  exec_diff = exec_diff / iter;
  gather_diff = gather_diff / iter;

  double add, mul, fma, flops;
  fftwf_flops(plan, &add, &mul, &fma);
  flops = add + mul + fma; 

#ifdef VERBOSE
  for(size_t i = 0; i < world_size; i++){
    if(myrank == i){
      cout << "Rank " << myrank << ": Flops - " << flops;
      cout << "Exec Time - " << exec_diff << " Gather Time - " << gather_diff;
      cout << endl;
    }
  }
#endif

  // calculating total flops
  double tot_flops;
  // TODO: throw exception, C++ does not return error values but throws exception
  MPI_Reduce(&flops, &tot_flops, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  /*
  if(checkStatus(status)){
    cleanup(per_process_data);
    throw "MPI Reduce failed!";
  }
  */

  if(myrank == 0){
    // Print to console the configuration chosen to execute during runtime
    print_config(N, false, world_size, nthreads, how_many, inverse, iter);
    cout << "\nTime to plan: " << plan_time << "sec\n\n";
    cout << "\nTime to gen reordered data: " << inp_data_diff << "sec\n\n";
    bool status = print_results(exec_diff, gather_diff, tot_flops, N, world_size, nthreads, iter);
    if(!status){
      cleanup(per_process_data, verify_per_process);
      throw "Printing Results function failed!";
    }
  }
}