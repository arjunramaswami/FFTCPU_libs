# MKL FFT3d

This folder contains the code to execute 3d FFT using MKL. Currently executes
multi threaded configurations on double precision floating point data.  

Link to the official [MKL FFT](https://software.intel.com/en-us/node/521955).

## Builds

### Prerequisites

`module load intel` : loads icc and mkl libraries

## Target

## Execution

Running the program with `-h` to print the options available.
Options:

```bash
  -m : FFT 1st Dim Size
  -n : FFT 2nd Dim Size
  -p : FFT 3rd Dim Size

  -i : Number of iterations of the FFT computation

  -t : number of threads in a multithreaded execution
```

### Example

```bash
  module load intel
  make
  ./host_dp -m 16 -n 16 -p 16 
  ./host_dp -m 256 -n 256 -p 256 -i 100 -t 20
```

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