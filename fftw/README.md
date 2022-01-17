# FFTW

Here, you find the code along with several helper scripts required to execute and collect performance results of FFT3D using [FFTW](http://www.fftw.org/). FFTW can be executed either using OpenMP multi-threaded only or hybrid (MPI + OpenMP) single precision configurations.

## Build

#### Prerequisites

The following libraries are required to be loaded before building:

- FFTW
- Intel MKL
- CMake

#### Targets

| Target | Description                            |  Linking Libraries
|:------:|----------------------------------------|---------------------|
| all         | builds the below two binaries     | 
| openmp_many | single node openmp multi-threaded binary | `-lfftw3f_omp -lfftw3f` |
| hybrid_many | distributed mpi+openmpi hybrid binary | `-lfftw3f_mpi  fftw3f_omp -lfftw3f`|

#### Build Parameters

| Target    | Description                            |
|:---------:|----------------------------------------|
| CMAKE_BUILD_TYPE | Debug, Release                  |
| FFTW_PLAN  | FFTW_ESTIMATE, FFTW_MEASURE, FFTW_PATIENT, FFTW_EXHAUSTIVE |

The default FFTW plan is **FFTW_ESTIMATE**.

#### How to build in Noctua

```bash
module load devel/CMake
module load toolchain/intel/2021a
module load numlib/FFTW/3.3.10-gompi-2021b

mkdir build
cmake ..
ccmake .. # to change params in gui
make 
```

## Execution

For: `openmp_many`:

```bash
Parse FFTW input params
Usage:
  FFTW [OPTION...]

  -n, --num arg         Size of FFT dim (default: 64)
  -t, --threads arg     Number of threads (default: 1)
  -c, --batch arg       Number of batch (default: 1)
  -i, --iter arg        Number of iterations (default: 1)
  -b, --inverse         Backward FFT
  -w, --wisdomfile arg  File to wisdom (default: test.wisdom)
  -e, --expm arg        Expm number (default: 1)
  -h, --help            Print usage
```

To execute:

```bash
# executing openmp multithreaded FFTW
./openmp_many --num=64 --threads=36 --iter=100
```

## Interpreting Results

The following metrics are measured:

1. Runtime of the fftw execution
2. Throughput based on the plan
3. Time to transfer data to master node

### Runtime

Runtime is measured for the following:

1. `fftw(f)_execute()` collective routine over a number of iterations, then its average runtime is considered.

2. `fftwf_plan_many_dft()` method that creates a plan for the fft configuration.

### Throughput

The `fftw(f)_flops` collective routine is used to obtain the number of add, mul and fused-multiply-accumulate floating point operations performed for the specific plan in every process. Their total would be total flops aggregated at the master node. This is divided by the runtime to find the throughput.

### Transfer time

In a distributed FFT, each process contains its respective transformed subset of points that could be gathered to the master node to form the complete set of transformed points. This transfer is performed using `MPI_Gather` and is timed using `MPI_Wtime()`.

### Console Output

The console output shows the configuration of execution followed by the following results:

```bash
Measurements
--------------------------
FFT Size            : 16^3
Threads             : 40
Batch               : 1
Iterations          : 50
Avg Tot Runtime     : 0.026052 ms
Runtime per batch   : 0.026052 ms
SD                  : 0.008196 ms
Throughput          : 0.000026 GFLOPs
Plan Time           : 1.226069 sec
```

### Note

- Execution and transfer times are measured using `MPI_Wtime()`.
- Iterations are made on the same input data.

## Measurements

### Runtime: Single Precision FFTW

#### MPI only

| # points | 1 Node | 4 Nodes | 8 Nodes | 16 Nodes | 32 Nodes | FPGA Total<br> PCIe Transfer | FPGA Kernel<br>+ PCIe | FPGA <br>SVM Transfer |
|:--------:|:------:|:-------:|:-------:|:--------:|:--------:|:----------------------------:|:---------------------:|:---------------------:|
|   32^3   | 0.0289 |    -    |         |          |          |             0.24             |          0.46         |         0.110         |
|   64^3   |  0.141 |    -    |         |          |          |             0.86             |          1.6          |         0.227         |
|   128^3  |  0.711 |   1.07  |         |          |          |             5.73             |                       |          1.60         |
|   256^3  |  6.94  |   7.89  |         |          |          |             44.42            |                       |         12.62         |
|   512^3  | 109.63 |  72.56  |  50.57  |          |          |            352.11            |                       |         98.71         |
|  1024^3  | 717.14 |  607.33 |  324.87 |  211.89  |  115.06  |            2822.96           |                       |                       |
|  4096^3  |        |         |         | 13414.41 |          |                              |                       |                       |

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

The `images/fftw_plans` directory has illustrations on the comparison of different runtimes of plans.

## Additional Scripts

Performance analysis of OpenMP FFTW FFT3D:

1. `scripts/fftw_runs/omp_fftw_patient.sh`:  run OpenMP configurations in Noctua. Outputs measurements are saved in `data/` folder and wisdoms in `wisdom` folder. In `data`, performance is logged per FFT size. 

2. `scripts/evaluation/perf_fftw.sh`: collects performance based on the folder structure in `data` to produce a `csv` file with the format `fftsize, perf(ms)`
    - `./perf_fftw.sh 16 671 perf_17.01.csv`

3. `scripts/evaluation/fftw_plot_perf.ipynb` is a JupterLab file that can be used to plot graphs using the previous `csv` file. The output images are stored in `images` folder.