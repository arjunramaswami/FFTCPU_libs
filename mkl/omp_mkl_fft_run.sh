#!/bin/bash
#SBATCH -A pc2-mitarbeiter
#SBATCH -J mkl_fft_ref
#SBATCH -p short
#SBATCH -N 1
#SBATCH -t 29:00

## Execute mkl fft multithreaded code 
##   Arg : Sizes of FFT to execute
##   e.g. omp_mkl_fft_run.sh 16 32 64
##   
## Each size is scaled from 1 to 40 threads, each performed iteration times
## Performance results are output to a timestamped file in data/ folder

module load intel/20.0.0

current_time=$(date "+%Y.%m.%d-%H.%M")
iteration=10

## Set OMP Environment Variables
export OMP_DISPLAY_AFFINITY=TRUE
export OMP_DISPLAY_ENV=TRUE

export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Execute applications from 1 to 40 threads one after another
echo "Passed $# FFT3d Sizes"
for arg in "$@"
do
    echo "Executing FFT Size : $arg $arg $arg"
    for thread in {1..40}
    do
        echo "Running with number of threads : $thread"
        #echo " Num of threads : $thread " >> omp_${arg}_fft_${current_time}
        ./host_dp -m $arg -n $arg -p $arg -i ${iteration} -t ${thread} >> data/omp_${arg}_fft_${current_time}
    done
done

