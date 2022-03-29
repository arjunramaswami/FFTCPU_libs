# FFT Libraries for CPU

This repository contains implementations of different FFT libraries. The libraries are self-composed within their respective folders with specific READMEs to configure and execute them.

1. FFTW
2. MKL FFT

## Configurations

- 1D, 2D and 3D FFT
- Forward / Backward FFT
- Single precision floating point
- Hybrid (MPI + OpenMP), OpenMP only

## Environment

These libraries have been tested using Intel Xeon Gold "Skylake" 6148 CPUs present in the [Noctua](https://pc2.uni-paderborn.de/hpc-services/available-systems/noctua1/) cluster of the Paderborn Center for Parallel Computing (PC2) at Paderborn University.

## Publications

Measurements using the libraries have been used in the following publications:

1. Evaluating the Design Space for Offloading 3D FFT Calculations to an FPGA for High-Performance Computing : https://doi.org/10.1007/978-3-030-79025-7_21

2. Efficient Ab-Initio Molecular Dynamic Simulations by Offloading Fast Fourier Transformations to FPGAs : https://doi.org/10.1109/FPL50879.2020.00065

## Related Repositories

- [FFTFPGA](https://github.com/pc2/fft3d-fpga) - OpenCL based library for Fast Fourier Transformations for FPGAs.

## Contact

- [Arjun Ramaswami](https://github.com/arjunramaswami)
- [Tobias Kenter](https://www.uni-paderborn.de/person/3145/)
- [Thomas D. Kühne](https://chemie.uni-paderborn.de/arbeitskreise/theoretische-chemie/kuehne/)
- [Christian Plessl](https://github.com/plessl)