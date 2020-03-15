# FFTW

This folder contains the code to execute 3d FFT using [FFTW](http://www.fftw.org/) that executes single process multi-threaded variant of single and double precision complex 3FFT.

## Build

#### Prerequisites

The following libraries have to be loaded in noctua in order to build the target

- FFTW:
  
   `module load numlib/FFTW/3.3.8-gompi-2019b`

- C compiler that implements OpenMPI (tested with gcc 8.3.0):

    `module load toolchain/gompi/2019b`

#### Targets

Use the makefile along with the targets mentioned below to build different configurations of the FFTW library.

| Target | Description                            |
|:------:|----------------------------------------|
|   all  | builds multithreaded single and double precision binary                      |

#### Build Parameters

| Target    | Description                            |
|:---------:|----------------------------------------|
| DEBUG     | output debug information on execution  |
| VERBOSE   | output runtime of every iteration      |
| MEASURE   | Plan FFT with type FFTW_MEASURE        |
| PATIENT   | Plan FFT with type FFTW_PATIENT        |
| EXHAUSTIVE| Plan FFT with type FFTW_EXHAUSTIVE     |

The default FFTW plan is **FFTW_ESTIMATE**.

The program is compiled to the `bin` folder. The following is an example of compilation:

```bash
module load toolchain/gompi/2019b
module load numlib/FFTW/3.3.8-gompi-2019b
make DEBUG=1 VERBOSE=1
make PATIENT=1
```

## Execution

The arguments available to the program:

|   Argument  | Default | Description                                      |
|:-----------:|---------|--------------------------------------------------|
| -h / --help | -       | displays the cmd line parameter options          |
|      -m     | 64      | number of points in the first dim of 3d fft      |
|      -n     | 64      | number of points in the second dim of 3d fft     |
|      -p     | 64      | number of points in the third dim of 3d fft      |
|      -b     | -       | compute backward 3d fft                          |
|      -s     | -       | call single precision functions instead of double|
|      -t     | 1       | number of threads if multithreading is available |
|      -i     | 1       | number of iterations of the application          |

To execute:

```bash
# executing a single threaded dp backward fftw. Note '-t' is 1
bin/fftw -m 16 -n 16 -p 16 -t 1 -i 2

# executing a 40 threaded dp fftw
bin/fftw -m 256 -n 256 -p 256 -t 40 -i 2

# executing a 20 threaded sp fftw
bin/fftw -m 256 -n 256 -p 256 -t 20 -s -i 2
```

## Interpreting Results

The following metrics are measured:

1. Runtime of the fftw execution
2. Throughput based on the plan

### Runtime

Runtime is measured for the following:

1. `fftw(f)_execute()` method over a number of iterations, then its average runtime is considered.

2. `fftw_plan_dft_3d()` method that creates a plan for the fft configuration.

### Throughput

The `fftw(f)_flops` method is used to obtain the number of add, mul and fused-multiply-accumulate floating point operations performed for the specific plan. Their total would be total flops. This is divided by the runtime to find the throughput.

### Console Output

The console output shows the configuration of execution followed by the following results:

```bash
Threads 3: time to plan - 0.319790 sec

       Threads  FFTSize  AvgRuntime(ms)  Throughput(GFLOPS)  
fftw:     3       64³       0.6434            2.56 
```

### Note

- Runtime only measures the walltime of the FFTW execution, measured using `clock_gettime` to provide nanosecond resolution.
- Iterations are made on the same input data. 

## Results

### Best Runtime

| # points | Best Runtime (ms) SP | Best Runtime (ms) DP | FPGA Execution (ms) | FPGA PCIe Transfer (ms) | FPGA Total (ms) | SVM Transfer (ms) |
|:--------:|:--------------------:|:--------------------:|:-------------------:|-------------------------|-----------------|-------------------|
| 32^3 | 0.0289 | 0.05 | 0.22 | 0.215 | 0.43 | 0.110 |
| 64^3 | 0.141 | 0.22 | 0.74 | 0.87 | 1.61 | 0.227 |
| 128^3 | 0.711 | 1.16 | - | 5.5 |  | 1.60 |
| 256^3 | 6.94 | 17.23 | - | 42.6 |  | 12.62 |
| 512^3 | 109.63 | 203.66 |  |  |  | 98.71 |
| 1024^3 | 717.13 |  |  |  |  | |

SP - Complex Single Precision Points
DP - Complex Double Precision Points

### Plans

Just for fun: Given below is the time taken to plan and the plans used to obtain the best runtime given above.

| # points | Plan Time (sec) SP | Plan Time (ms) DP |
|:--------:|:------------------:|-------------------|
| 32^3     | 4.23 (patient)     | 2.17 (patient)    |
| 64^3     | 0.0012 (estimate)  | 0.0018 (estimate) |
| 128^3    | 53.3 (patient)     | 65.5 (patient)    |
| 256^3    | 360 (patient)      | 6.60 (measure)    |
| 512^3    | 1449 (patient)     | 2240 (patient)    |
| 1024^3    | 17320 (patient)     |    |

The `common/fftw_plans` directory has illustrations on the comparison of different runtimes of plans.

## Additional Scripts

- Bash scripts for different precisions and plans.

```
sbatch omp_fftw_dp_run.sh <array of sizes of fft3d>

sbatch omp_fftw_dp_run.sh 16 32 64
```

- Bash script `create_csv.sh` can be used to create a csv output from the reports generated by the above bash file.

```
./create_csv.sh <generated_report> <output.csv>
./create_csv.sh ../raw/measure/sp_128_* ../csv/measure/sp_128.csv
```

- Bash script `plan.sh` can be used to tabulate the time to plan using different planning schemes for different number of threads.

```
./plan.sh <generate report> <output.csv>
./plan.sh ../raw/measure/sp_128_* ../csv/measure/plans/sp_128_plan.csv
```
