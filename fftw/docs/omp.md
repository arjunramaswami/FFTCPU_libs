# Configuring FFTW with OpenMP

Steps to configure multithreaded execution of FFTW with OpenMPI

- Link with `-fopenmp` to enable threads if compiling with GCC.

- To use parallel transforms, link with:
  - `-lfftw3_omp -lfftw3 -lm` for double precision FFT
  - ` -lfftw3f -lfftw3f_omp -lm` for single precision FFT

- Initialize the environment before calling any FFTW functions

  ```C
  #include<omp.h>
  int fftw_init_threads(void);   // dp
  int fftwf_init_threads(void);  // sp
  ```

- To make all subsequent plans use threads.

  ```C
  void fftw_plan_with_nthreads(int nthreads);   // dp
  void fftwf_plan_with_nthreads(int nthreads);  // sp
  ```

- Create a 3d fft plan

- Execute with the normal API call

  ```C
  fftw_execute(plan)            // dp
  fftwf_execute(plan)           // sp
  ```

- Cleanup plan and threads after execution

  ```C
  fftw_destroy_plan()
  fftw_cleanup_threads();

  fftwf_destroy_plan()
  fftwf_cleanup_threads();
  ```

- Configure the number of threads to use by setting it using `OMP_NUM_THREADS` or `omp_set_num_threads()` during execution.
