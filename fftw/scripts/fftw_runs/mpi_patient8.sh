#!/bin/bash
#SBATCH --account=pc2-mitarbeiter
#SBATCH --job-name=mpi_patient
#SBATCH --partition=all
#SBATCH --nodes=2
#SBATCH -t 3:00:00
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --switches=1

module reset
module load devel/CMake
module load toolchain/gompi
module load numlib/FFTW

#export OMP_DISPLAY_AFFINITY=TRUE
export OMP_PLACES=cores     
export OMP_PROC_BIND=close

outdir="../data/patient/mpi2nodes/result"
wisdir="../wisdom/mpi2nodes"
iter=100
batch=1

echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "N         : $@"
echo "Threads per process   : 8"
echo "Processes : $SLURM_NTASKS"
echo "Iter   : ${iter}"
echo "Batch  : ${batch}"
# Execute applications from 1 to 40 threads one after another
echo ""
echo "Passed $# FFT3d Sizes"
for arg in "$@"
do
  echo "FFT Size : $arg $arg $arg"
  for thread in {8..8}
  do
    echo "Running with number of threads : $thread"
    outfile="${outdir}/fftw_${arg}_p8_t${thread}_b${batch}.log"
    wisdomfile="${wisdir}/fftw_${arg}_p8_t${thread}_b${batch}.patient"
    echo "Writing to file : ${outfile}"

    #srun ../build/hybrid_many --num=$arg -i ${iter} -t ${thread} --batch=${batch} --wisdomfile=${wisdomfile} > ${outfile}
    mpirun --map-by ppr:2:socket --bind-to socket ../build/hybrid_many --num=$arg -i ${iter} -t ${thread} --batch=${batch} --wisdomfile=${wisdomfile} > ${outfile}
    #mpirun --map-by ppr:2:socket --bind-to socket --report-bindings ../build/hybrid_many --num=$arg -i ${iter} -t ${thread} --batch=${batch} --wisdomfile=${wisdomfile} > ${outfile}
  done
done
