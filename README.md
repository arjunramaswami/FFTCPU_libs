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

- FFT3d sizes
- Forward / Backward FFT
- Precision of floating point numbers : single precision, double precision
- Single Threaded, multi threaded using OpenMP

### Metrics measured

- Runtime (milliseconds)
- Throughput (GFLOPS)

## Performance

### Runtime Comparison

Runtime is reported in milliseconds for single precision complex floating points.

The FPGA runtime includes PCIe transfer latencies.

| FFT3d Size |   FFTW   |    MKL   | FPGA Obtained |
|:----------:|:--------:|:--------:|:-------------:|
|     32     |   0.02   |  0.056   |    0.43       |
|     64     |   0.14   |  0.197   |    1.61       |
|     128    |   0.71   |  1.393   |               |
|     256    |   6.94   |  23.19   |               |
|     512    |   109.63 |          |               |
|    1024    |   717.14 |          |               |

### Environment

#### CPU used 

2x Intel Xeon Gold "Skylake" 6148, 2.4 GHz, each with 20 cores, hyperthreading disabled

Cache Hierarchy:

- L0, L1I (32KB), L1D (32KB) private per core.
- L2 private - 1MB/core
- L3 non inclusive - 1.375 MB/core or 27.5 MB per CPU

#### Library Versions

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

#### Justifying the performance loss using FFT and cache sizes

To clearly understand if the loss in performance is due to cache misses, one can estimate the maximum FFT size that can fit into the L3 cache and obtain its performance. Then, calculate the performance of the FFT that is just larger than the cache.

- In this case, FFT size `153^3` should completely fit into 1 L3 cache but has a best performance of 28GFLOPS, whereas `154^3` which shouldn't, has a performance of 105GFLOPS.

Perhaps, FFT sizes that are powers of 2 could justify performance as the implementations of other FFT configurations could vary.

1. FFT Size `128 * 128 * 256` is larger than 1 L3 cache of the 2 in the node and has the best throughput of approximately 100 GFLOPS. Considering the L3 caches are non-inclusive and cache coherent, this should provide enough memory for the FFT to be stored in both caches, but the performance is lowered due to NUMA latency and cache coherency.

2. FFT size `128 * 256 * 256` is however, larger than the total L3 cache size but performs close to 75 GFLOPS as compared to `256 * 128 * 256`, which has a throughput of 25 GFLOPS but are identical in the number of points.

This makes it difficult to create a coherent reason to justify performance by correlating FFT sizes and cache due to possible differences in FFTW implementations and plans.
