//  Author: Arjun Ramaswami

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include <math.h>
#include <float.h> // FLT_EPSILON

#include <fftw3.h>
#include <fftw3-mpi.h>

#include "helper.h"

/**
 * \brief  create single precision floating points values for FFT computation for each process level block
 * 
 * \param  fftw_data   : pointer to 3d number of sp points 
 * \param  n0, n1, n2  : number of points in each dimension
 * \param  local_n0    : number of points in the n0 dim
 * \param  local_start : starting point in the n0 dim
 * \param  H1, H2, H3  : harmonic to modify frequency of discrete time signal
 */
void get_mpi_sp_input_data(fftwf_complex *fftw_data, ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2, ptrdiff_t local_n0, ptrdiff_t local_start, int H1, int H2, int H3){

  int i, j, k, index;
  int N1 = n0, N2 = n1, N3 = n2;
  int S1 = N2*N3, S2 = N3, S3 = 1;

  float TWOPI = 6.2831853071795864769;
  float phase, phase1, phase2, phase3;

  for (i = 0; i < local_n0; i++) {
    for (j = 0; j < N2; j++) {
      for (k = 0; k < N3; k++) {
        phase1 = moda(i + local_start, H1, N1) / N1;
        phase2 = moda(j, H2, N2) / N2;
        phase3 = moda(k, H3, N3) / N3;
        phase = phase1 + phase2 + phase3;

        index = i*S1 + j*S2 + k*S3;

        fftw_data[index][0] = cosf( TWOPI * phase ) / (N1*N2*N3);
        fftw_data[index][1] = sinf( TWOPI * phase ) / (N1*N2*N3);

#ifdef DEBUG          
        printf(" %d %d %d : fftw[%d] = (%f, %f) \n", i, j, k, index, fftw_data[index][0], fftw_data[index][1]);
#endif
      }
    }
  }
}

/**
 * \brief  Verify single precision FFT3d computation
 * \param  x : fftwf_complex - 3d FFT data after transformation
 * \param  N - fft size
 * \param  H1, H2, H3 : harmonic to modify frequency of discrete time signal
 */
int verify_sp(fftwf_complex *x, int N1, int N2, int N3, int H1, int H2, int H3){
  /* Verify that x(n1,n2,n3) is a peak at H1,H2,H3 */
  double err, errthr, maxerr;
  int index;

  /* Generalized strides for row-major addressing of x */
  int S1 = N2*N3, S2 = N3, S3 = 1;

  /*
    * Note, this simple error bound doesn't take into account error of
    * input data
    */
  errthr = 5.0 * log( (float)N1*N2*N3 ) / log(2.0) * FLT_EPSILON;
#ifdef DEBUG
  printf("Verify the result, errthr = %.3lg\n", errthr);
#endif

  maxerr = 0;
  for (int n1 = 0; n1 < N1; n1++){
    for (int n2 = 0; n2 < N2; n2++){
      for (int n3 = 0; n3 < N3; n3++){
        float re_exp = 0.0, im_exp = 0.0, re_got, im_got;

        if ((n1-H1)%N1==0 && (n2-H2)%N2==0 && (n3-H3)%N3==0) {
          re_exp = 1;
        }

        index = n1*S1 + n2*S2 + n3*S3;
        re_got = x[index][0];
        im_got = x[index][1];
        err  = fabs(re_got - re_exp) + fabs(im_got - im_exp);
        if (err > maxerr) maxerr = err;
        if (!(err < errthr)){       
            printf(" x[%i][%i][%i]: ",n1,n2,n3);
            printf(" expected (%.17lg,%.17lg), ",re_exp,im_exp);
            printf(" got (%.17lg,%.17lg), ",re_got,im_got);
            printf(" err %.3lg\n", err);
            printf(" Verification FAILED\n");
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
            printf(" %d %d %d : fftw[%d] = (%f, %f) \n", i, j, k, index, x[index][0], x[index][1]);
          }
      }
  }
#endif

  return 0;
}

void fftwf_mpi(int N1, int N2, int N3, int nthreads, int inverse, int iter){
    
  int H1 = 1, H2 = 1, H3 = 1, status;
  double gather_diff = 0.0, exec_diff = 0.0;

  int provided, threads_ok;
  MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided); // instead of MPI_Init
  threads_ok = provided >= MPI_THREAD_FUNNELED;
  if (threads_ok){ 
    threads_ok = fftwf_init_threads();
    if(threads_ok == 0){
      printf("Something went wrong with SP Multithreaded FFTW! Exiting... \n");
      exit(EXIT_FAILURE);
    }
  }
  fftwf_mpi_init();

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
  alloc_local = fftwf_mpi_local_size_3d(n0, n1, n2, MPI_COMM_WORLD,
    &local_n0, &local_0_start);
#ifdef VERBOSE
  printf("Rank %d of %d:\n\ttotal pts: %d\n\tpts in local transform: %td\n\t%td blocks starting from %td\n\teach block of %d pts \n\n", 
    myrank, world_size, N1*N2*N3, alloc_local, local_n0, local_0_start, N2*N3);
#endif

  // allocate
  fftwf_complex *per_process_data = fftwf_alloc_complex(alloc_local);
  size_t data_sz = N1 * N2 * N3;
  fftwf_complex *total_data = fftwf_alloc_complex(data_sz);

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
    fftwf_plan_with_nthreads(nthreads);
  }

#ifdef VERBOSE
  if(myrank == 0)
    printf("Configuring plan for single precision FFT\n\n");
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  double plan_start = MPI_Wtime();
#ifdef MEASURE
  fftwf_plan plan = fftwf_mpi_plan_dft_3d(n0, n1, n2, per_process_data, per_process_data, MPI_COMM_WORLD, direction, FFTW_MEASURE);
#elif PATIENT
  fftwf_plan plan = fftwf_mpi_plan_dft_3d(n0, n1, n2, per_process_data, per_process_data, MPI_COMM_WORLD, direction, FFTW_PATIENT);
#elif EXHAUSTIVE
  fftwf_plan plan = fftwf_mpi_plan_dft_3d(n0, n1, n2, per_process_data, per_process_data, MPI_COMM_WORLD, direction, FFTW_EXHAUSTIVE);
#else
  fftwf_plan plan = fftwf_mpi_plan_dft_3d(n0, n1, n2, per_process_data, per_process_data, MPI_COMM_WORLD, direction, FFTW_ESTIMATE);
#endif
  MPI_Barrier(MPI_COMM_WORLD);
  double plan_end = MPI_Wtime();

  // iterate iter times
  for(size_t it = 0; it < iter; it++){
    // fill data
    if(inverse){
      get_mpi_sp_input_data(per_process_data, n0, n1, n2, local_n0, local_0_start, -H1, -H2, -H3);
    }
    else{
      get_mpi_sp_input_data(per_process_data, n0, n1, n2, local_n0, local_0_start, H1, H2, H3);
    }

    // execute
    MPI_Barrier(MPI_COMM_WORLD);
    double exec_start = MPI_Wtime();
    fftwf_execute(plan);
    MPI_Barrier(MPI_COMM_WORLD);
    double exec_end = MPI_Wtime();

    // gather to master process
    MPI_Barrier(MPI_COMM_WORLD);
    double gather_start = MPI_Wtime();
    status = MPI_Gather(&per_process_data[0], alloc_local, MPI_C_FLOAT_COMPLEX, &total_data[0], alloc_local, MPI_C_FLOAT_COMPLEX, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double gather_end = MPI_Wtime();

    if(!checkStatus(status)){
      exit(EXIT_FAILURE);
    }

    // verify gathered transformed data
    if(myrank == 0){
      status = verify_sp(total_data, N1, N2, N3, H1, H2, H3);
      if(status == 1){
        fprintf(stderr, "Error in transformation\n");
        exit(EXIT_FAILURE);
      }
    }

#ifdef VERBOSE
    printf("Rank: %d\n\titer %zu\n\texec time: %lfsec\n\tgather time: %lfsec\n\n", myrank, it, (exec_end - exec_start), (gather_end - gather_start));
#endif
#ifdef DEBUG
    for(size_t d = 0; d < data_sz; d++){
      printf("%zu - %f %f\n", d, total_data[d][0], total_data[d][1]);
    }
#endif

    exec_diff += (exec_end - exec_start);
    gather_diff += (gather_end - gather_start);
  }

  double add, mul, fma, flops;
  fftwf_flops(plan, &add, &mul, &fma);
  flops = add + mul + fma; 

#ifdef VERBOSE
  printf("Rank %d: Flops - %lf\n", myrank, flops);
#endif

  double tot_flops;
  status = MPI_Reduce(&flops, &tot_flops, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if(!checkStatus(status)){
    exit(EXIT_FAILURE);
  }

  if(myrank == 0){
    // Print to console the configuration chosen to execute during runtime
    print_config(N1, N2, N3, 1, world_size, nthreads, inverse, iter);
    printf("\nTime to plan: %lfsec\n\n", plan_end - plan_start);
    print_results(exec_diff, gather_diff, tot_flops, N1, N2, N3, world_size, nthreads, iter);
  }

  // Cleanup 
  fftwf_free(per_process_data);
  fftwf_free(total_data);
  fftwf_destroy_plan(plan);
  fftwf_cleanup_threads();
  fftwf_mpi_cleanup();

  MPI_Finalize();
}
