#!/bin/bash
#SBATCH -A pc2-mitarbeiter
#SBATCH -J est_dp_fftw_ref
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 11:29:00

## Execute fftw multithreaded code 
##   Arg : Sizes of FFT to execute
##   e.g. omp_fftw_run.sh 16 32 64
##   
## Each size is scaled from 1 to 40 threads, each performed iteration times
## Performance results are output to a timestamped file in data/ folder

module load toolchain/gompi/2019b
module load numlib/FFTW/3.3.8-gompi-2019b

ctime=$(date "+%Y.%m.%d-%H.%M")
outdir="../data/estimate/"
iter=100

make -C ../ 

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
    echo "Running with number of threads : ${thread}"
    outfile="${outdir}dp_${arg}_${ctime}"
    echo "Writing to file : ${outfile}"

    ../bin/fftw -m $arg -n $arg -p $arg -i ${iter} -t ${thread} >> ${outfile}
  done
done
