# FFTW

This folder contains code to execute 3d FFT using [FFTW](http://www.fftw.org/) that executes hybrid (MPI + OpenMP) single and double precision complex configurations.

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
|   all  | builds multiprocess multithreaded single and double precision binary                      |

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
mpirun -n 2 ./bin/fftw -m 16 -n 16 -p 16 -t 1 -i 2

# executing a 40 threaded dp fftw
mpirun -n 16 ./bin/fftw -m 256 -n 256 -p 256 -t 40 -i 2

# executing a 20 threaded sp fftw
mpirun -n 8 ./bin/fftw -m 256 -n 256 -p 256 -t 20 -s -i 2
```

## Interpreting Results

The following metrics are measured:

1. Runtime of the fftw execution
2. Throughput based on the plan
3. Time to transfer data to master node

### Runtime

Runtime is measured for the following:

1. `fftw(f)_execute()` collective routine over a number of iterations, then its average runtime is considered.

2. `fftw_plan_dft_3d()` method that creates a plan for the fft configuration.

### Throughput

The `fftw(f)_flops` collective routine is used to obtain the number of add, mul and fused-multiply-accumulate floating point operations performed for the specific plan in every process. Their total would be total flops aggregated at the master node. This is divided by the runtime to find the throughput.

### Transfer time

In a distributed FFT, each process contains its respective transformed subset of points that could be gathered to the master node to form the complete set of transformed points. This transfer is performed using `MPI_Gather` and is timed using `MPI_Wtime()`.

### Console Output

The console output shows the configuration of execution followed by the following results:

```bash
Time to plan: 0.002379sec


       Processes  Threads  FFTSize  AvgRuntime(ms)  Throughput(GFLOPS) AvgTimetoTransfer(ms)  
fftw:       2        2       16³       0.1071            1.1476              0.0515 
```

### Note

- Execution and transfer times are measured using `MPI_Wtime()`.
- Iterations are made on the same input data.

## Measurements

### Runtime: Single Precision FFTW

#### MPI only

| # points | 1 Node | 4 Nodes | 8 Nodes | 16 Nodes | 32 Nodes | FPGA Total<br> PCIe Transfer | FPGA Kernel<br>+ PCIe | FPGA <br>SVM Transfer |
|:--------:|:------:|:-------:|:-------:|:--------:|:--------:|:----------------------------:|:---------------------:|:---------------------:|
| 32^3 | 0.0289 | - |  |  |  | 0.24 | 0.46 | 0.110 |
| 64^3 | 0.141 | - |  |  |  | 0.86 | 1.6 | 0.227 |
| 128^3 | 0.711 | 1.07 |  |  |  | 5.73 |  | 1.60 |
| 256^3 | 6.94 | 7.89 |  |  |  | 44.42 |  | 12.62 |
| 512^3 | 109.63 | 72.56 | 50.57 |  |  | 352.11 |  | 98.71 |
| 1024^3 | 717.14 | 607.33 | 324.87 | 211.89 | 115.06 | 2822.96 |  |  |

- Runtime is in milliseconds.
- Best runtime:
  - 1 Node with 1 process and 1-40 threads per node, best of all the plans.
  - 4, 8, 16, 32 nodes with 32 processes per node using patient plan. No multithreading.
- PCIe Transfer is the summation of read and write average over 100 iterations.
- SVM Transfer is full duplex average over 100 iterations i.e. parallel reads and writes.

#### MPI + OpenMP

| # points | 1 Node | 4 Nodes | 8 Nodes | FPGA Total<br> PCIe Transfer | FPGA Kernel<br>+ PCIe | FPGA <br>SVM Transfer |
|:--------:|:------:|:-------:|:-------:|:----------------------------:|:---------------------:|:---------------------:|
| 32^3 | 0.0289 | 0.093 | 0.07 | 0.24 (0.12 + 0.12) | 0.46 | 0.110 |
| 64^3 | 0.141 | 0.46 | 0.35 | 0.86 (0.43 + 0.43) | 1.6 | 0.227 |
| 128^3 | 0.711 | 3.07 | 1.64 | 5.73 (2.78 + 2.95) |  | 1.60 |
| 256^3 | 6.94 | 30.41 | 13.72 | 44.42 (21.38 + 23.04) |  | 12.62 |
| 512^3 | 109.63 | 327.60 | 184.5 | 352.11 (170.08 + 183.03) |  | 98.71 |
| 1024^3 | 717.14 | 2356.55 | 1253.83 | 2822.96 (1359.86 + 1463.10) |  |  ||

- Runtime is in milliseconds.
- Best runtime:
  - 1 Node with 1 process and 1-40 threads per node, best of all the plans.
  - 4 and 8 Nodes with 1 process and 32-40 threads per node. Using patient plan.
- PCIe Transfer is the summation of read and write average over 100 iterations.
- SVM Transfer is full duplex average over 100 iterations i.e. parallel reads and writes.

| # points | 1 Node | 2x MPI Collective <br>Data transfers | 4 Nodes |
|:--------:|:------:|:------------------------------------:|:-------:|
| 32^3 | 0.0289 | 0.23 | 0.093 |
| 64^3 | 0.141 | 0.4 | 0.46 |
| 128^3 | 0.711 | 2.94 | 3.07 |
| 256^3 | 6.94 | 38.26 | 30.41 |
| 512^3 | 109.63 | 333.74 | 327.60 |
| 1024^3 | 717.14 | 2660.14 | 2356.55 |

- All measurements in milliseconds
- Time for data transfer using MPI_Gather collective routine from 4 nodes to a single node.
- The cumulative time to transfer data and execute a faster single node FFTW is worse than performing a distributed 4 nodal Hybrid FFTW.

### Runtime: Double Precision FFTW

| # points | 1 Node |
|:--------:|:------:|
| 32^3 | 0.05 |
| 64^3 | 0.22 |
| 128^3 | 1.16 |
| 256^3 | 17.23 |
| 512^3 | 203.66 |
| 1024^3 |  |

- Runtime is in milliseconds.
- Best runtime using 1 Node with 1 process and 1-40 threads per node, best of all the plans.

### Plans

Just for fun: Given below is the time taken to plan and the plans used to obtain the best runtime given above for single node configuration. SP is single precision, DP is double precision.

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
