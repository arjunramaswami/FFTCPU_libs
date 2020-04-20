//  Author: Arjun Ramaswami

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include <math.h>
#include <float.h> // DBL_EPSILON

#include <fftw3.h>
#include <fftw3-mpi.h>

#include "helper.h"

static fftw_plan plan;

/**
 * \brief  create double precision floating points values for FFT computation for each process level block
 * 
 * \param  fftw_data : pointer to 3d number of dp points
 * \param  n0, n1, n2  : number of points in each dimension
 * \param  local_n0    : number of points in the n0 dim
 * \param  local_start : starting point in the n0 dim
 * \param  H1, H2, H3 : harmonic to modify frequency of discrete time signal
 */
void get_mpi_dp_input_data(fftw_complex *fftw_data, ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2, ptrdiff_t local_n0, ptrdiff_t local_start, int H1, int H2, int H3){
  int i, j, k, index;
  int N1 = n0, N2 = n1, N3 = n2;
  int S1 = N2*N3, S2 = N3, S3 = 1;

  double TWOPI = 6.2831853071795864769;
  double phase, phase1, phase2, phase3;

  for (i = 0; i < local_n0; i++) {
    for (j = 0; j < N2; j++) {
      for (k = 0; k < N3; k++) {
        phase1 = moda(i + local_start, H1, N1) / N1;  // phase values between 0 and 1
        phase2 = moda(j, H2, N2) / N2;
        phase3 = moda(k, H3, N3) / N3;
        phase = phase1 + phase2 + phase3;

        index = i*S1 + j*S2 + k*S3;

        fftw_data[index][0] = cos( TWOPI * phase ) / (N1*N2*N3); 
        fftw_data[index][1] = sin( TWOPI * phase ) / (N1*N2*N3);

#ifdef DEBUG    
        printf(" %d %d %d : fftw[%d] = (%lf, %lf) \n", i, j, k, index, fftw_data[index][0], fftw_data[index][1]);
#endif
      }
    }
  }
}

/**
 * \brief  Verify double precision FFT3d computation
 * \param  x : fftw_complex - 3d FFT data after transformation
 * \param  N1, N2, N3 - fft size
 * \param  H1, H2, H3 : harmonic to modify frequency of discrete time signal
 * \return 0 if successful, 1 otherwise
 */
int verify_dp(fftw_complex *x, int N1, int N2, int N3, int H1, int H2, int H3){
  /* Verify that x(n1,n2,n3) is a peak at H1,H2,H3 */
  double err, errthr, maxerr;
  int index;

  /* Generalized strides for row-major addressing of x */
  int S1 = N2*N3, S2 = N3, S3 = 1;

  /*
    * Note, this simple error bound doesn't take into account error of
    * input data
    */
  errthr = 5.0 * log( (double)N1*N2*N3 ) / log(2.0) * DBL_EPSILON;
#ifdef DEBUG
  printf("Verify the result, errthr = %.3lg\n", errthr);
#endif

  maxerr = 0;
  for (int n1 = 0; n1 < N1; n1++){
      for (int n2 = 0; n2 < N2; n2++){
          for (int n3 = 0; n3 < N3; n3++){
              double re_exp = 0.0, im_exp = 0.0, re_got, im_got;

              if ((n1-H1)%N1==0 && (n2-H2)%N2==0 && (n3-H3)%N3==0) {
                  re_exp = 1;
              }

              index = n1*S1 + n2*S2 + n3*S3;
              re_got = x[index][0];
              im_got = x[index][1];
              err  = fabs(re_got - re_exp) + fabs(im_got - im_exp);
              if (err > maxerr) maxerr = err;
              if (!(err < errthr)){
                  fprintf(stderr," x[%i][%i][%i]: ",n1,n2,n3);
                  fprintf(stderr," expected (%.17lg,%.17lg), ",re_exp,im_exp);
                  fprintf(stderr," got (%.17lg,%.17lg), ",re_got,im_got);
                  fprintf(stderr," err %.3lg\n", err);
                  fprintf(stderr," Verification FAILED\n");
                  return 1;
              }
          }
      }
  }

#ifdef VERBOSE
  printf("Verification: Maximum error was %.3lg\n\n", maxerr);
#endif

#ifdef DEBUG
  printf("\n\nOutput Frequencies: \n");
  for(int i = 0; i < N1; i++){
      for(int j = 0; j < N2; j++){
          for(int k = 0; k < N3; k++){
              index = (i * N1 * N2) + (j * N3) + k;
              printf(" %d %d %d : fftw[%d] = (%lf, %lf) \n", i, j, k, index, x[index][0], x[index][1]);
          }
      }
  }
#endif

  return 0;
}

static void cleanup(fftw_complex *per_process_data){
  // Cleanup 
  fftw_free(per_process_data);
  fftw_destroy_plan(plan);
  //fftwf_cleanup_threads();
  fftw_mpi_cleanup();
}

/**
 * \brief  Distributed Double precision FFTW execution
 * \param  N1, N2, N3 - fft size
 * \param  nthreads   - number of threads
 * \param  inverse    - 1 if backward transform
 * \param  iter       - number of iterations of execution
 * \return 0 if successful, 1 otherwise
 */
int fftw_mpi(int N1, int N2, int N3, int nthreads, int inverse, int iter){

  int H1 = 1, H2 = 1, H3 = 1, status;
  double gather_diff = 0.0, exec_diff = 0.0;

  int threads_ok = fftw_init_threads();
  if(threads_ok == 0){
    fprintf(stderr, "Something went wrong with DP Multithreaded FFTW! Exiting... \n");
    return 1;
  }
  fftw_mpi_init();

  int world_size, myrank, namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Comm_size(MPI_COMM_WORLD, &world_size); 
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Get_processor_name(processor_name, &namelen);

#ifdef VERBOSE
    #pragma omp parallel 
    {
        int num_th = omp_get_num_threads();
        int thr_num = omp_get_thread_num();
        printf("Hybrid: Hello from thread %d out of %d from process %d out of %d on %s\n",
                thr_num, num_th, myrank, world_size, processor_name);
    }
#endif

  ptrdiff_t alloc_local, local_n0, local_0_start;
  ptrdiff_t n0 = N1;
  ptrdiff_t n1 = N2;
  ptrdiff_t n2 = N3;

  // get local data size
  alloc_local = fftw_mpi_local_size_3d(n0, n1, n2, MPI_COMM_WORLD,
    &local_n0, &local_0_start);
#ifdef VERBOSE
  for(size_t i = 0; i < world_size; i++){
    if(myrank == i){
      printf("Rank %d of %d:\n\ttotal pts: %d\n\tpts in local transform: %td\n\t%td blocks starting from %td\n\teach block of %d pts \n\n", 
    myrank, world_size, N1*N2*N3, alloc_local, local_n0, local_0_start, N2*N3);
    }
  }
#endif
  
  // allocate
  fftw_complex *per_process_data = fftw_alloc_complex(alloc_local);

  // direction of FFT for the plan
  int direction = FFTW_FORWARD;
  if(inverse){
    direction = FFTW_BACKWARD;
  }

  // include threads in plan
  if(threads_ok){
#ifdef VERBOSE
  if(myrank == 0)
    printf("Threads per rank: %d\n\n", nthreads);
#endif
    fftw_plan_with_nthreads(nthreads);
  }
  
#ifdef VERBOSE
  if(myrank == 0)
    printf("Configuring plan for double precision FFT\n\n");
#endif

  MPI_Barrier(MPI_COMM_WORLD);
  double plan_start = MPI_Wtime();
#ifdef MEASURE
  plan = fftw_mpi_plan_dft_3d(n0, n1, n2, per_process_data, per_process_data, MPI_COMM_WORLD, direction, FFTW_MEASURE);
#elif PATIENT
  plan = fftw_mpi_plan_dft_3d(n0, n1, n2, per_process_data, per_process_data, MPI_COMM_WORLD, direction, FFTW_PATIENT);
#elif EXHAUSTIVE
  plan = fftw_mpi_plan_dft_3d(n0, n1, n2, per_process_data, per_process_data, MPI_COMM_WORLD, direction, FFTW_EXHAUSTIVE);
#else
  plan = fftw_mpi_plan_dft_3d(n0, n1, n2, per_process_data, per_process_data, MPI_COMM_WORLD, direction, FFTW_ESTIMATE);
#endif
  MPI_Barrier(MPI_COMM_WORLD);
  double plan_time = MPI_Wtime() - plan_start;

  // iterate iter times
  for(size_t it = 0; it < iter; it++){

    size_t data_sz = N1 * N2 * N3;
    fftw_complex *total_data = fftw_alloc_complex(data_sz);
    // fill data
    if(inverse){
      get_mpi_dp_input_data(per_process_data, n0, n1, n2, local_n0, local_0_start, -H1, -H2, -H3);
    }
    else{
      get_mpi_dp_input_data(per_process_data, n0, n1, n2, local_n0, local_0_start, H1, H2, H3);
    }

    // execute
    MPI_Barrier(MPI_COMM_WORLD);
    double exec_start = MPI_Wtime();
    fftw_execute(plan);
    MPI_Barrier(MPI_COMM_WORLD);
    double exec_end = MPI_Wtime();

    // gather to master process
    double gather_start = MPI_Wtime();
    status = MPI_Gather(&per_process_data[0], alloc_local, MPI_C_DOUBLE_COMPLEX, &total_data[0], alloc_local, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    double gather_end = MPI_Wtime();
    if(checkStatus(status)){
      cleanup(per_process_data);
      return 1;
    }

    // verify gathered transformed data
    if(myrank == 0){
      status = verify_dp(total_data, N1, N2, N3, H1, H2, H3);
      if(status == 1){
        fprintf(stderr, "Error in transformation\n");
        cleanup(per_process_data);
        return 1;
      }
    }

#ifdef VERBOSE
    for(size_t i = 0; i < world_size; i++){
      if(myrank == i){
        printf("Rank: %d\n\titer %zu\n\texec time: %lfsec\n\tgather time: %lfsec\n\n", myrank, it, (exec_end - exec_start), (gather_end - gather_start));
      }
    }
#endif
#ifdef DEBUG
    for(size_t d = 0; d < data_sz; d++){
      printf("%zu - %lf %lf\n", d, total_data[d][0], total_data[d][1]);
    }
#endif

    exec_diff += (exec_end - exec_start);
    gather_diff += (gather_end - gather_start);

    free(total_data);
  }

  exec_diff = exec_diff / iter;
  gather_diff = gather_diff / iter;

  double add, mul, fma, flops;
  fftw_flops(plan, &add, &mul, &fma);
  flops = add + mul + fma; 

#ifdef VERBOSE
  for(size_t i = 0; i < world_size; i++){
    if(myrank == i){
      printf("Rank %d: Flops - %lf Exec Time - %lf Gather Time - %lf\n", myrank, flops, exec_diff, gather_diff);
    }
  }
#endif

  // calculating total flops
  double tot_flops;
  status = MPI_Reduce(&flops, &tot_flops, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if(checkStatus(status)){
    cleanup(per_process_data);
    return 1;
  }

  if(myrank == 0){
    // Print to console the configuration chosen to execute during runtime
    print_config(N1, N2, N3, 0, world_size, nthreads, inverse, iter);
    printf("\nTime to plan: %lfsec\n\n", plan_time);
    status = print_results(exec_diff, gather_diff, tot_flops, N1, N2, N3, world_size, nthreads, iter);
    if(status){
      cleanup(per_process_data);
      return 1;
    }
  }

  return 0;
}