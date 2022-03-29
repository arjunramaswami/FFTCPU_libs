# MKL FFT

This folder contains the along with helper scripts required to execute FFT using [MKL](https://software.intel.com/en-us/node/521955). It can be executed for MPI + OpenMP multi-threaded single precision configurations.

## Builds

### Prerequisites

The following libraries are required to be loaded before building:

- FFTW
- Intel MKL
- CMake

## Target

| Target | Description                            |  Linking Libraries
|:------:|----------------------------------------|---------------------|
| all         | builds the below binaries     | 
| openmp_many | single node openmp multi-threaded binary | `lp64, thread, core, libiomp5` |

### Build Parameters

| Target    | Description                            |
|:---------:|----------------------------------------|
| CMAKE_BUILD_TYPE | Debug, Release                  |

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

For `openmp_many`:

```bash
Parse FFTW input params
Usage:
  FFTW [OPTION...]

  -n, --num arg         Size of FFT dim (default: 64)
  -d, --dim arg         Number of dim (default: 3)
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
# executing openmp multithreaded MKL FFT
./mkl_openmp_many --num=64 --iter=100 --expm=1
```

## Interpreting Results

- Runtime of the fft execution. It is measured by timing `DftiComputeForward` routine over a number of iterations, then the following metrics are computed:
  - Average runtime, standard deviation
  - Median, Q1, Q3 : dispersion without outlier influence

## Note

- icc vs gcc: gcc showed a lot more variance in runtime, so icc is recommended.
- the first run of an mkl fft is atleast 3x slower than the median performance.
  - Perhaps due to the lack of a specific planning phase
