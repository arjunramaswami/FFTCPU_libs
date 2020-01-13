# FFT3d Library Comparison

This repository contains implementations of different FFT libraries in
order to compare their performance with the FPGA implementation.

# Libraries 
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

![Performance Comparison](common/fftw_mkl_perf.png)

#### Runtime Comparison

Runtime is reported in milliseconds.

| FFT3d Size |   FFTW   |    MKL   |
|:----------:|:--------:|:--------:|
|     16     |   0.022  |  0.012   |
|     32     |   0.060  |  0.056   |
|     64     |   0.270  |  0.197   |
|     128    |   1.591  |  1.393   |
|     256    |  90.415  |  23.19   |

#### Throughput Comparison

Throughput is reported in GFLOPS.

| FFT3d Size |   FFTW   |    MKL   |
|:----------:|:--------:|:--------:|
|     16     |   11.13  |  19.25   |
|     32     |   40.56  |  43.58   |
|     64     |   87.09  | 119.80   |
|     128    |  138.39  | 158.08   |
|     256    |   22.26  |  86.80   |

## Analysis

1. Reason for loss in performance when scaling from 128 to 256 cube FFT
