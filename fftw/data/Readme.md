# Experiments and resulting data

The experiments are categorised based on the FFTW plan type:

- Patient Plan
  - all_wisonly_expm1, _expm2: 3D FFT from 16^3 using existing wisdom
  - prime: 1D & 3D mainly to understand the prime algorithm used for each FFT
    - Algorithm in 1D prime is the same as in 3D prime, hence the corroboration
    - required a custom installation of FFTW, with print stms in respective `apply()` functions of `dft/rader.c`, `dft/bluestein.c`, `dft/rank-geq2.c` and `dft/generic.c`. The latter 2 can be interpreted as hard-coded and dft (not fft) implementations.
  - hybrid for mpi+openmp runs

Other experiments and their respective folders are self-explanatory.

# Environment

- FFTW 3.3.9 with Gcc 10.3.0 and OpenMPI 4.1.1
- Intel MKL 2021a with icc 2021.2.0
- CMake 3.20