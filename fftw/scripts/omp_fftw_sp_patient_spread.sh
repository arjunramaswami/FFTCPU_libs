#!/bin/bash
#SBATCH --account=pc2-mitarbeiter
#SBATCH --job-name=fftw_patient
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH -t 3-00:00:00

## Execute fftw multithreaded code 
##   Arg : Sizes of FFT to execute
##   e.g. omp_fftw_run.sh 16 32 64
##   
## Each size is scaled from 1 to 40 threads, each performed iteration times
## Performance results are output to a timestamped file in data/ folder

module reset
module load devel/CMake
module load toolchain/gompi
module load numlib/FFTW

#make PATIENT=1 -C ../debug

## OMP Environment Variables
export OMP_DISPLAY_AFFINITY=TRUE

# every thread is pinned to a core, Options: Sockets, Cores, Threads
export OMP_PLACES=cores     

# Can threads be moved between procs: True / False (no affinity)
# Thread Affinity: close (successive cores) / spread (successive sockets)
export OMP_PROC_BIND=spread

export OMP_DISPLAY_ENV=TRUE

ctime=$(date "+%Y.%m.%d-%H.%M")
outdir="../data/patient/mpi1nodes/cores_spread/"
iter=100
batch=1

# Execute applications from 1 to 40 threads one after another
echo "Passed $# FFT3d Sizes"
for arg in "$@"
do
  echo "Executing FFT Size : $arg $arg $arg"
  for thread in {1..40}
  do
    echo "Running with number of threads : $thread"
    outfile="${outdir}sp_${arg}_${thread}_${ctime}_batch${batch}"
    echo "Writing to file : ${outfile}"

    ../build/openmp_many --num=$arg -i ${iter} -t ${thread} --batch=${batch} > ${outfile}
  done
done