# FFTW 
This folder contains the code to execute 3d FFT using FFTW. Currently executes 
single threaded, multi threaded configurations on single and double precision 
complex numbers.

Link to the official FFTW [website](http://www.fftw.org/).

## Builds

### Prerequisites
- easyload FFTW `module load numlib/FFTW`
- C compiler that implements OpenMPI (tested with gcc 8.3.0)

### Targets
1. Single precision single threaded FFTW 
  - `make` and `make sp` : Creates an executable named `host_sp` in the FFTW 
directory





2. For double precision single threaded FFTW
  - `make dp` : Creates an executable named `host_dp` in the FFTW directory




3. Multi threaded FFTW (always uses double precision) 
  - `make omp` : Creates an executable named `host_omp` in the FFTW directory


* Compiling with DEBUG macro set say, `make DEBUG=1 omp`, prints data input to
  FFTW

## Execution

Running the program with `-h` to print the options available.

Options:

```
  -m : FFT 1st Dim Size
  -n : FFT 2nd Dim Size
  -p : FFT 3rd Dim Size

  -i : Number of iterations of the FFT computation
  -b : toggles backward FFT

  -t : number of threads in a multithreaded execution
```



### Example multithreaded execution

```
  module load numlib/FFTW
  make omp
  ./host_omp -m 256 -n 256 -p 256 -t 40
```

## Interpreting Results

The output shows configuration of execution and the series of steps the program
executes followed by the consolidation of some performance metrics such as: 

```
Number of runs : 100

        FFT Size    Total Runtime(ms)   Avg Runtime(ms)     Throughput(GFLOPS)
fftw      32<sup>3<sup>     30.051          0.300               23.51
```

- FFT Size is the size of the 3d FFT
- Total Runtime (milliseconds) : total amount of time to execute the given
  number of iterations.
- Avg Runtime (milliseconds) : average runtime for a single iteration of execution i.e.,
  total runtime by the number of iterations
- Throughput (GFLOPS): Calculated by *3 * 5 * N * logN / (time for one FFT)* as described
  by the [FFTW Benchmark Methodology](http://www.fftw.org/speed/method.html).
  This is not the actual flop count rather an asymptotic measurement using the
  radix-2 Cooley Tukey algorithm. 

### Important Points 
- Runtime only measures the walltime of the FFTW execution, not the
  initialization and plan creation. Measured using `clock_gettime` to provide
  nanosecond resolution.
- Iterations are made on the same input data. Input data is $[0, N^3 - 1]$ where
  N is the number of data points in a dimension.
 
#  Results






# Configuring FFTW with OpenMP

Steps to configure multithreaded execution of FFTW with OpenMPI

1. To compile with OpenMPI, link this additional flag `-lfftw3_omp` along with 
`-fopenmp` other than the regular `fftw` flags. These are added to the makefile.


2. Modify the code

*  Initialize the environment:
    ```
    #include<omp.h>
    int fftw_init_threads(void); 
    ```


*  Make the plan with the necessary number of threads to execute
    ```
    void fftw_plan_with_nthreads(int nthreads);
    ```


*  Execute with the normal 
    ```
    fftw_execute(plan) 
    ```
    
*  Cleanup plan and threads after execution
    ```
    fftw_destroy_plan()
    void fftw_cleanup_threads(void);
    ```

3. FFTW Api for multithreading works only with double precision 

*  Creating a single precision plan after initializing threads produces single 
   threaded outcomes


*  Creating fftwf alternatives throw linker error due to lack of such
   functionalities

#### Note
Execution is thread-safe but not plan creation and destruction, therefore use a
single thread for the latter. 

