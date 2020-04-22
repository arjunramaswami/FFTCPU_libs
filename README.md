# FFT3d Library Comparison

This repository contains implementations of different FFT libraries in
order to compare their performance with the FPGA implementation.

## Libraries 
These libraries are self composed within their respective folders. The READMEs
to configure and execute them are also available.

1. FFTW
2. MKL FFT

### Configurations

Different configurations when using the code:

- 3d FFT points
- Forward / Backward FFT
- Precision of floating point numbers : single precision, double precision
- Hybrid (MPI + OpenMP)
- Plans: estimate, measure, patient, exhaustive

## FFTW Measurements

### Runtime: Single Precision FFTW

| # points | 1 Node | 4 Nodes | 8 Nodes | 16 Nodes | 32 Nodes | FPGA Total<br> PCIe Transfer | FPGA Kernel<br>+ PCIe | FPGA <br>SVM Transfer |
|:--------:|:------:|:-------:|:-------:|:--------:|:--------:|:----------------------------:|:---------------------:|:---------------------:|
| 32^3 | 0.0289 | - |  |  |  | 0.24 | 0.46 | 0.110 |
| 64^3 | 0.141 | - |  |  |  | 0.86 | 1.6 | 0.227 |
| 128^3 | 0.711 | 1.07 |  |  |  | 5.73 |  | 1.60 |
| 256^3 | 6.94 | 7.89 |  |  |  | 44.42 |  | 12.62 |
| 512^3 | 109.63 | 72.56 | 50.57 |  |  | 352.11 |  | 98.71 |
| 1024^3 | 717.14 | 607.33 | 324.87 | 211.89 | 115.06 | 2822.96 |  |  ||

- Runtime is in milliseconds.
- Best runtime:
  - 1 Node with 1 process and 1-40 threads per node, best of all the plans.
  - 4,8,16,32 nodes with 32 processes per node using patient plan. No multithreading.
- PCIe Transfer is the summation of read and write average over 100 iterations.
- SVM Transfer is full duplex average over 100 iterations i.e. parallel reads and writes.

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

## Environment

### CPU used

2x Intel Xeon Gold "Skylake" 6148, 2.4 GHz, each with 20 cores, hyperthreading disabled

Cache Hierarchy:

- L0, L1I (32KB), L1D (32KB) private per core.
- L2 private - 1MB/core
- L3 non inclusive - 1.375 MB/core or 27.5 MB per CPU

### Library Versions

- FFTW 3.3.8 linked with GCC v8.3.0
- Intel MKL as part of Intel Parallel Studio XE 2020, linked with icc v19.1

## Analysis

|  sz |  sp cmplex (MB) | dp cmplex (MB) |
|:---:|:---------------:|:--------------:|
|  32 |       0.25      |       0.5      |
|  64 |        2        |        4       |
| 128 |        16       |       32       |
| 256 |       128       |       256      |
| 512 |       1024      |       2048     |
| 1024 |     8192       |     16384      |

Reason for loss in performance when scaling from 128 to 256 cube sp FFT - 128 cube requires only 16 MB of memory whereas 256 cube FFT uses 128 MB of memory. The latter cannot be stored in the L3 cache, which has only 27.5 MB of memory per CPU. Thereby, the loss in performance.

### Justifying the performance loss using FFT and cache sizes

To clearly understand if the loss in performance is due to cache misses, one can estimate the maximum FFT size that can fit into the L3 cache and obtain its performance. Then, calculate the performance of the FFT that is just larger than the cache.

- In this case, FFT size `153^3` should completely fit into 1 L3 cache but has a best performance of 28GFLOPS, whereas `154^3` which shouldn't, has a performance of 105GFLOPS.

Perhaps, FFT sizes that are powers of 2 could justify performance as the implementations of other FFT configurations could vary.

1. FFT Size `128 * 128 * 256` is larger than 1 L3 cache of the 2 in the node and has the best throughput of approximately 100 GFLOPS. Considering the L3 caches are non-inclusive and cache coherent, this should provide enough memory for the FFT to be stored in both caches, but the performance is lowered due to NUMA latency and cache coherency.

2. FFT size `128 * 256 * 256` is however, larger than the total L3 cache size but performs close to 75 GFLOPS as compared to `256 * 128 * 256`, which has a throughput of 25 GFLOPS but are identical in the number of points.

This makes it difficult to create a coherent reason to justify performance by correlating FFT sizes and cache due to possible differences in FFTW implementations and plans.
