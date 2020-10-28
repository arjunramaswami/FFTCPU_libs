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

static fftwf_plan plan;

static void cleanup(fftwf_complex *per_process_data){
  // Cleanup 
  fftwf_free(per_process_data);
  //fftwf_cleanup_threads(); // Thread Cleanup
  fftwf_mpi_cleanup();
  fftwf_destroy_plan(plan);
}

/**
 * \brief  create single precision floating points values for FFT computation for each process level block
 * 
 * \param  fftw_data   : pointer to 3d number of sp points 
 * \param  n0, n1, n2  : number of points in each dimension
 * \param  local_n0    : number of points in the n0 dim
 * \param  local_start : starting point in the n0 dim
 * \param  H1, H2, H3  : harmonic to modify frequency of discrete time signal
 */
void get_sp_input_data(fftwf_complex *fftw_data, ptrdiff_t N, ptrdiff_t local_n0, ptrdiff_t local_start, unsigned H1, unsigned H2, unsigned H3, unsigned how_many){

  unsigned index; //S1 = N * S2, S2 = N * S3, S3 = how_many;

  float TWOPI = 6.2831853071795864769;
  float phase, phase1, phase2, phase3;

  cout << "Input data: " << endl;
  for (ptrdiff_t i = 0; i < local_n0; i++) {
    for (ptrdiff_t j = 0; j < N; j++) {
      for (ptrdiff_t k = 0; k < N; k++) {

        phase1 = moda(i + local_start, H1, N) / N;
        phase2 = moda(j, H2, N) / N;
        phase3 = moda(k, H3, N) / N;
        phase = phase1 + phase2 + phase3;

        double re_val = cosf( TWOPI * phase ) / (N*N*N);
        double img_val = sinf( TWOPI * phase ) / (N*N*N);

        for(ptrdiff_t many = 0; many < how_many; many++){

          index = (i * N * N * how_many) + (j * N * how_many) + (k * how_many) + many;
          //index = (many * local_n0 * S1) + i*S1 + j*S2 + k*S3;

          fftw_data[index][0] = re_val;
          fftw_data[index][1] = img_val;

          cout << many << ": " << i << " " << j << " " << k << " : fftw[" << index << "] = ";
          cout <<"(" << fftw_data[index][0] << ", " << fftw_data[index][1] << ")";
          cout << endl;
        }
      }
    }
  }
  cout << endl;
}


/**
 * \brief  Verify single precision FFT3d computation
 * \param  x : fftwf_complex - 3d FFT data after transformation
 * \param  N1, N2, N3 - fft size
 * \param  H1, H2, H3 : harmonic to modify frequency of discrete time signal
 * \return true if successful, false otherwise
 */
bool verify_fftw(fftwf_complex *x, unsigned N, unsigned H1, unsigned H2, unsigned H3, unsigned how_many){
  /* Verify that x(n1,n2,n3) is a peak at H1,H2,H3 */
  double err;

  /* Generalized strides for row-major addressing of x */
  unsigned S1 = N*N, S2 = N, S3 = 1, index;

  /*
    * Note, this simple error bound doesn't take into account error of
    * input data
    */
  double errthr = 5.0 * log( (float)N*N*N ) / log(2.0) * FLT_EPSILON;
#ifdef DEBUG
  cout << "Verify the result, errthr = " << errthr << endl;
#endif
  /*
  double maxerr = 0.0;
  for (size_t n1 = 0; n1 < N; n1++){
    for (size_t n2 = 0; n2 < N; n2++){
      for (size_t n3 = 0; n3 < N; n3++){
        float re_exp = 0.0, im_exp = 0.0, re_got, im_got;

        if ((n1-H1)%N==0 && (n2-H2)%N==0 && (n3-H3)%N==0) {
          re_exp = 1;
        }

        index = n1*S1 + n2*S2 + n3*S3;
        re_got = x[index][0];
        im_got = x[index][1];
        err  = fabs(re_got - re_exp) + fabs(im_got - im_exp);
        if (err > maxerr) maxerr = err;
        if (!(err < errthr)){       
            cerr << " x["<< n1 << "]["<< n2 << "]["<< n3 << "]";
            cerr << " expected (" << re_exp << "," << im_exp << ")";
            cerr << " got (" << re_got << "," << im_got << ")";
            cerr << " err " << err << endl;
            cerr << " Verification FAILED\n";
            return false;
        }
      }
    }
  }
*/
#ifdef VERBOSE
  cerr << "Verification: Maximum error was " << maxerr << endl;
#endif
  cout << "\n\nOutput Frequencies: \n";
  for(size_t many = 0; many < how_many; many++){
    for(size_t i = 0; i < N; i++){
        for(size_t j = 0; j < N; j++){
            for(size_t k = 0; k < N; k++){
              index = (many * N * N * N) + (i * N * N) + (j * N) + k;
              cout << many << ": " << i << " " << j << " " << k << ": fftw[" << index << "] = (" << x[index][0] << ", " << x[index][1] << ")" << endl;

            }
        }
    }
  }

  return true;
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
  double gather_diff = 0.0, exec_diff = 0.0;

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
  
    #pragma omp parallel 
    {
        // prints number of threads per rank on machine
        int num_th = omp_get_num_threads();
        int thr_num = omp_get_thread_num();
        cout << "Hybrid: Hello from thread "<< thr_num << " out of "<< num_th;
        cout << " from process " << myrank << " out of " << world_size;
        cout << " on " << processor_name << endl;
    }

  ptrdiff_t alloc_local, local_n0, local_0_start;
  ptrdiff_t n[3] = {N, N, N};

  // get local data size
  alloc_local = fftwf_mpi_local_size_many(3, n, how_many,   FFTW_MPI_DEFAULT_BLOCK, MPI_COMM_WORLD, &local_n0, &local_0_start);

  //alloc_local = fftwf_mpi_local_size_3d(n0, n1, n2, MPI_COMM_WORLD, &local_n0, &local_0_start);
  for(size_t i = 0; i < world_size; i++){
    if(myrank == i){
      cout << "Rank " << myrank << " of " << world_size;
      cout << "\n\t" << alloc_local << "pts in local transform of total: " <<  N*N*N*how_many;
      cout << "\n\t" << local_n0 << " blocks starting from " << local_0_start; cout << "\n\t" << " each block of " << N*N << "pts \n\n"; }
     }

  // allocate
  fftwf_complex *per_process_data = fftwf_alloc_complex(alloc_local);

  // direction of FFT for the plan
  int direction = FFTW_FORWARD;
  if(inverse){
    direction = FFTW_BACKWARD;
  }

  if(myrank == 0)
    cout << "Configuring plan for single precision FFT" << endl;
    
  // plan
  MPI_Barrier(MPI_COMM_WORLD);
  double plan_start = MPI_Wtime();
#ifdef MEASURE
  plan = fftwf_mpi_plan_many_dft(3, n, how_many, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, per_process_data, per_process_data, MPI_COMM_WORLD, direction, FFTW_MEASURE);
#elif PATIENT
  plan = fftwf_mpi_plan_many_dft(3, n, how_many, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, per_process_data, per_process_data, MPI_COMM_WORLD, direction, FFTW_PATIENT);
  //plan = fftwf_mpi_plan_dft_3d(n0, n1, n2, per_process_data, per_process_data, MPI_COMM_WORLD, direction, FFTW_PATIENT);
#elif EXHAUSTIVE
  plan = fftwf_mpi_plan_many_dft(3, n, how_many, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, per_process_data, per_process_data, MPI_COMM_WORLD, direction, FFTW_EXHAUSTIVE);
  //plan = fftwf_mpi_plan_dft_3d(n0, n1, n2, per_process_data, per_process_data, MPI_COMM_WORLD, direction, FFTW_EXHAUSTIVE);
#else
  plan = fftwf_mpi_plan_many_dft(3, n, how_many, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, per_process_data, per_process_data, MPI_COMM_WORLD, direction, FFTW_ESTIMATE);
  //plan = fftwf_mpi_plan_dft_3d(n0, n1, n2, per_process_data, per_process_data, MPI_COMM_WORLD, direction, FFTW_ESTIMATE);
#endif
  MPI_Barrier(MPI_COMM_WORLD);
  double plan_time = MPI_Wtime() - plan_start;

  // iterate iter times 
  for(size_t it = 0; it < iter; it++){
    size_t data_sz = N * N * N * how_many;
    fftwf_complex *total_data = fftwf_alloc_complex(data_sz); //-- large sz

    // fill data
    if(inverse){
      get_sp_input_data(per_process_data, N, local_n0, local_0_start, -H1, -H2, -H3, how_many);
    }
    else{
      get_sp_input_data(per_process_data, N, local_n0, local_0_start, H1, H2, H3, how_many);
    }

    // execute
    MPI_Barrier(MPI_COMM_WORLD);
    double exec_start = MPI_Wtime();
    fftwf_execute(plan); 
    MPI_Barrier(MPI_COMM_WORLD);
    double exec_end = MPI_Wtime();

    // gather transformed result to master process
    /* --large sz
    */
    //double gather_start = MPI_Wtime();
    // TODO: exception
    /*
    MPI_Gather(&per_process_data[0], alloc_local, MPI_COMPLEX, &total_data[0], alloc_local, MPI_COMPLEX, 0, MPI_COMM_WORLD);
    double gather_end = MPI_Wtime();
    */
    /*
    if(checkStatus(status)){
      cleanup(per_process_data);
      return 1;
    }
    */

    cout << "Output Frequencies: " << endl;
    unsigned index;
    for(size_t many = 0; many < how_many; many++){
      for(size_t i = 0; i < N; i++){
          for(size_t j = 0; j < N; j++){
              for(size_t k = 0; k < N; k++){
                index = (many * N*N*N) + (i * N * N) + (j * N) + k;
                cout << many << ": " << i << " " << j << " " << k << ": fftw_out[" << index << "] = (" << per_process_data[index][0] << ", " << per_process_data[index][1] << ")" << endl;

              }
          }
      }
    }

    // verify result 
    /*
    if(myrank == 0){
      bool status = verify_fftw(total_data, N, H1, H2, H3, how_many);
      if(!status){
        cerr << "Error in transformation\n";
        cleanup(per_process_data);
        throw "Error in verification transformation";
      }
    }
    */
    
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
#ifdef DEBUG
    for(size_t d = 0; d < data_sz; d++){
      cout << d << ": " << total_data[d][0] << " " << total_data[d][1]) << endl;
    }
#endif
    double gather_end = 0.0, gather_start = 0.0;
    exec_diff += (exec_end - exec_start);
    gather_diff += (gather_end - gather_start);

    //free(total_data); // large sz
  }

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
    bool status = print_results(exec_diff, gather_diff, tot_flops, N, world_size, nthreads, iter);
    if(!status){
      cleanup(per_process_data);
      throw "Printing Results function failed!";
    }
  }
}