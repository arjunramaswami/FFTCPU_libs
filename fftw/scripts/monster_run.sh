#!/bin/bash
#SBATCH -A pc2-mitarbeiter
#SBATCH -J fftw_ref
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 11:29:00

#./omp_fftw_sp_run.sh 64 128 256 
#./omp_fftw_dp_run.sh 64 128 256 

./omp_fftw_sp_measure.sh 64 128
./omp_fftw_dp_measure.sh 64 128

./omp_fftw_sp_patient.sh 64 128
./omp_fftw_dp_patient.sh 64 128

./omp_fftw_sp_exhaustive.sh 64 128
./omp_fftw_dp_exhaustive.sh 64 128
