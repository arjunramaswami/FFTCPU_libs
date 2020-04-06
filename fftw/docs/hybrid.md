# Configuring FFTW with MPI and OpenMP

Steps to enable distributed execution.

## Link

- `-lfftw3 -lfftw3_mpi -lfftw3_omp -lm` for double precision FFT

- `-lfftw3f -lfftw3f_mpi -lfftw3f_omp -lm` for single precision FFT

## Initialize environment

```C
  #include <mpi.h>
  #include <omp.h>
  #include <fftw3.h>
  #include <fftw3-mpi.h>

  // initialize multithreaded MPI
  // funneled dictates all MPI calls to be funneled through the master thread
  MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);  
  ...
  // initialize threading
  fftw_init_threads()

  // initialize MPI execution environment
  fftw_mpi_init();

  // cleanup
  fftw_mpi_cleanup();
  fftw_cleanup_threads();
  MPI_Finalize();
```


## Divide 3d points to processes and allocate

```C

  // collective call that returns number of points transformed per process
  // distribution is in the first dimension n0, in blocks of n1xn2
  // starting block number is returned in local_0_start
  // the number of blocks per process is returned in local_n0
  alloc_local = fftw_mpi_local_size_3d(n0, n1, n2, MPI_COMM_WORLD,
  &local_n0, &local_0_start);

  // alocate complex data of size returned per process
  data = fftw_alloc_complex(alloc_local);
```

## Plan

```C
  // enable threading in plan
  fftw_plan_with_nthreads(nthreads);

  // MPI plan passing the world as argument
  fftw_plan plan = fftw_mpi_plan_dft_3d(n0, n1, n2, per_process_data, per_process_data, MPI_COMM_WORLD, direction, FFTW_MEASURE);
```

## Execute

  `fftw_execute(plan)` collective call.

## Additional

- Transformed data is gathered to the master rank to be verfied.
- `fftw_flops()` to calculate the throughput. Reduced to find the total throughput at the master rank. Collective call

> `fftwf_` for respective single precision calls
> `fftw_execute() fftw_flops()` are collective calls without MPI prefix.
