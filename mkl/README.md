# MKL FFT3d

This folder contains the code to execute 3d FFT using MKL. Currently executes
multi threaded configurations on double precision floating point data.  

Link to the official [MKL FFT](https://software.intel.com/en-us/node/521955).

## Builds

### Prerequisites

`module load intel` : loads icc and mkl libraries

## Target

## Execution

## Interpreting Results

## Results

## Configuring FFT with MKL

- Create descriptor
  - creates a configuration for the FFT to be computed.
  - inputs : precision, data type, dim, size

- Set additional configuration values
  - number of threads, in-place / not in-place placement

- Commit Descriptor

- Initialize input to transform

- Compute Transform