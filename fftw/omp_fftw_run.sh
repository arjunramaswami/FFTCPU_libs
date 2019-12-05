#!/bin/bash
#SBATCH -A pc2-mitarbeiter
#SBATCH -J fftw_ref
#SBATCH -p batch
#SBATCH -N 1

## Execute fftw multithreaded code 
##   Arg : Sizes of FFT to execute
##   e.g. omp_fftw_run.sh 16 32 64
##   
## Each size is scaled from 1 to 40 threads, each performed iteration times
## Performance results are output to a timestamped file in data/ folder

module load toolchain/gompi
module load numlib/FFTW

current_time=$(date "+%Y.%m.%d-%H.%M")
iteration=1

echo "Passed $# FFT3d Sizes"
for arg in $@
do
    echo "Executing FFT Size : $arg $arg $arg"
    for thread in {1..40}
    do
        #echo " Num of threads : $thread " >> omp_${arg}_fft_${current_time}
        ./host_omp -m $arg -n $arg -p $arg -i ${iteration} -t ${thread} >> data/omp_${arg}_fft_${current_time}
    done
done

